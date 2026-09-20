//! What a data frame's payload is, when it is not the bytes themselves.
//!
//! A payload that is not raw is one self-contained block. Its header names the
//! element type, the shape of the piece of the output array it holds and the
//! error bound it was given, so a receiver can expand it knowing nothing about
//! the transfer it belongs to — which is what lets blocks arrive out of order,
//! on any connection, and be re-fetched on their own.
//!
//! Shape is the point of the header. Every error-bounded compressor predicts
//! from neighbours in every axis, so handing it a flat run of elements throws
//! away much of what it could do. End to end on the same 256 MiB of float32,
//! SZ3 at a bound of 1e-3 reached 9.70x told the array is 32768 x 2048 and
//! 5.85x told the same bytes are one long row.

use crate::dtype::DType;
use crate::error::{AexError, Result};
use crate::quality::Codec;

#[cfg(feature = "sz")]
mod sz;

/// Bytes of block header before the codec's own stream.
const BLOCK_HEADER_LEN: usize = 24;

/// The most axes a block is described with.
///
/// SZ3 takes at most four, and the gain from the fourth is small next to the
/// gain from the first two, so the fold to three keeps one rule for every
/// array rank.
const MAX_BLOCK_DIMS: usize = 3;

const VERSION: u8 = 1;

/// One block: what it holds, and how faithfully.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockSpec {
    pub dtype: DType,
    /// C order, fastest axis last. Only `dims[..ndim]` is meaningful.
    dims: [u32; MAX_BLOCK_DIMS],
    ndim: u8,
    /// Absolute error bound. Every element comes back within this of the
    /// original.
    pub eps: f64,
}

impl BlockSpec {
    /// Describe the block covering `[offset, offset + len)` of a logical byte
    /// stream whose output array is `out_shape`.
    ///
    /// A byte range of a C-order stream is a run of innermost rows, so a range
    /// that starts and ends on a row boundary is an array slab, full in every
    /// axis but the slowest. One that does not is described as a single row,
    /// which costs ratio and nothing else — so nothing here has to be
    /// guaranteed by the caller.
    pub fn for_range(
        out_shape: &[u64],
        dtype: DType,
        eps: f64,
        offset: u64,
        len: u64,
    ) -> Result<Self> {
        let itemsize = dtype.itemsize();
        if len == 0 || len % itemsize != 0 {
            return Err(AexError::BadBlock(format!(
                "a block of {len} bytes does not hold whole {dtype} elements"
            )));
        }
        let count = len / itemsize;

        let axes = slab_axes(out_shape, itemsize, offset, len).unwrap_or_else(|| vec![count]);
        debug_assert_eq!(axes.iter().product::<u64>(), count);

        let mut dims = [0u32; MAX_BLOCK_DIMS];
        for (slot, axis) in dims.iter_mut().zip(&axes) {
            *slot = u32::try_from(*axis).map_err(|_| {
                AexError::BadBlock(format!("a block axis of {axis} elements is too long"))
            })?;
        }
        Ok(BlockSpec {
            dtype,
            dims,
            ndim: axes.len() as u8,
            eps,
        })
    }

    /// Elements in the block.
    pub fn count(&self) -> u64 {
        self.dims[..self.ndim as usize]
            .iter()
            .map(|d| *d as u64)
            .product()
    }

    /// Logical bytes the block expands to.
    pub fn bytes(&self) -> u64 {
        self.count() * self.dtype.itemsize()
    }

    /// The axes, C order, fastest last.
    pub fn dims(&self) -> &[u32] {
        &self.dims[..self.ndim as usize]
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.push(VERSION);
        out.push(self.dtype.as_i32() as u8);
        out.push(self.ndim);
        out.push(0);
        out.extend_from_slice(&self.eps.to_le_bytes());
        for dim in self.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
    }

    fn decode(src: &[u8]) -> Result<Self> {
        let bad = |what: String| AexError::BadBlock(what);
        if src.len() < BLOCK_HEADER_LEN {
            return Err(bad(format!(
                "a header is {BLOCK_HEADER_LEN} bytes, and only {} arrived",
                src.len()
            )));
        }
        if src[0] != VERSION {
            return Err(bad(format!(
                "block format {} is not one this build knows",
                src[0]
            )));
        }
        let dtype = DType::from_i32(src[1] as i32)?;
        let ndim = src[2];
        if ndim == 0 || ndim as usize > MAX_BLOCK_DIMS {
            return Err(bad(format!(
                "a block has 1 to {MAX_BLOCK_DIMS} axes, not {ndim}"
            )));
        }
        let eps = f64::from_le_bytes(src[4..12].try_into().expect("8 bytes"));
        if !eps.is_finite() || eps <= 0.0 {
            return Err(bad(format!("{eps} is not an error bound")));
        }
        let mut dims = [0u32; MAX_BLOCK_DIMS];
        for (i, dim) in dims.iter_mut().enumerate() {
            let at = 12 + i * 4;
            *dim = u32::from_le_bytes(src[at..at + 4].try_into().expect("4 bytes"));
            // A length-1 axis is folded away when the block is described, so
            // one here means the header disagrees with itself.
            let want_used = i < ndim as usize;
            if want_used != (*dim > 1) && !(want_used && ndim == 1 && *dim == 1) {
                return Err(bad(format!("axis {i} of a {ndim}-axis block is {dim}")));
            }
        }
        Ok(BlockSpec {
            dtype,
            dims,
            ndim,
            eps,
        })
    }
}

/// The block's axes, when the range is a slab of whole innermost rows.
///
/// `None` when it is not, and the caller falls back to one flat axis.
fn slab_axes(out_shape: &[u64], itemsize: u64, offset: u64, len: u64) -> Option<Vec<u64>> {
    let &inner = out_shape.last()?;
    let row = inner.checked_mul(itemsize)?;
    if row == 0 || offset % row != 0 || len % row != 0 {
        return None;
    }
    let rows = len / row;
    let first = offset / row;

    // Three axes when the slab is whole planes starting on a plane boundary;
    // two otherwise.
    let plane = if out_shape.len() >= 2 {
        out_shape[out_shape.len() - 2]
    } else {
        1
    };
    let axes = if plane > 1 && rows % plane == 0 && first % plane == 0 {
        vec![rows / plane, plane, inner]
    } else {
        vec![rows, inner]
    };

    // Dropping a length-1 axis leaves the element order alone, and the
    // compressors refuse one anywhere but on their own.
    let axes: Vec<u64> = axes.into_iter().filter(|axis| *axis > 1).collect();
    (!axes.is_empty()).then_some(axes)
}

/// Compress one block into `dst`, which is replaced.
///
/// `false` means the block did not shrink — small blocks cost more in codec
/// header than they save — and the caller should send the raw bytes instead.
/// `dst` is left empty in that case.
pub fn compress(codec: Codec, spec: &BlockSpec, src: &[u8], dst: &mut Vec<u8>) -> Result<bool> {
    if spec.bytes() != src.len() as u64 {
        return Err(AexError::BadBlock(format!(
            "a block of {:?} {} elements is {} bytes, not {}",
            spec.dims(),
            spec.dtype,
            spec.bytes(),
            src.len()
        )));
    }
    dst.clear();
    spec.encode(dst);
    let written: Result<()> = match codec {
        #[cfg(feature = "sz")]
        Codec::Sz => sz::compress(spec, src, dst),
        other => Err(AexError::BadBlock(format!(
            "{other:?} is not a codec this build can produce"
        ))),
    };
    if let Err(e) = written {
        dst.clear();
        return Err(e);
    }
    if dst.len() >= src.len() {
        dst.clear();
        return Ok(false);
    }
    Ok(true)
}

/// Expand one block into `dst`, which must be exactly the block's logical size.
pub fn decompress_into(codec: Codec, src: &[u8], dst: &mut [u8]) -> Result<()> {
    let spec = BlockSpec::decode(src)?;
    if spec.bytes() != dst.len() as u64 {
        return Err(AexError::BadBlock(format!(
            "a block of {:?} {} elements expands to {} bytes, not the {} asked for",
            spec.dims(),
            spec.dtype,
            spec.bytes(),
            dst.len()
        )));
    }
    match codec {
        #[cfg(feature = "sz")]
        Codec::Sz => sz::decompress_into(&spec, &src[BLOCK_HEADER_LEN..], dst),
        other => Err(AexError::BadBlock(format!(
            "{other:?} is not a codec this build can expand"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn spec(out_shape: &[u64], dtype: DType, offset: u64, len: u64) -> BlockSpec {
        BlockSpec::for_range(out_shape, dtype, 1e-3, offset, len).expect("a well-formed range")
    }

    #[test]
    fn a_row_aligned_range_is_described_as_the_slab_it_is() {
        // 1000 x 1024 float32: a 512 KiB piece is 128 whole rows.
        let piece = 512 * 1024;
        let s = spec(&[1000, 1024], DType::Float32, 0, piece);
        assert_eq!(s.dims(), [128, 1024]);
        assert_eq!(s.bytes(), piece);

        // The same slab further in.
        assert_eq!(
            spec(&[1000, 1024], DType::Float32, piece * 3, piece).dims(),
            [128, 1024]
        );
    }

    #[test]
    fn whole_planes_on_a_plane_boundary_get_a_third_axis() {
        // 100 x 8 x 1024 float32: one plane is 32 KiB, so 512 KiB is 16 planes.
        let s = spec(&[100, 8, 1024], DType::Float32, 0, 512 * 1024);
        assert_eq!(s.dims(), [16, 8, 1024]);

        // Half a plane in, the planes no longer line up, so two axes it is.
        let s = spec(&[100, 8, 1024], DType::Float32, 16 * 1024, 512 * 1024);
        assert_eq!(s.dims(), [128, 1024]);
    }

    #[test]
    fn a_range_that_is_not_whole_rows_is_one_flat_axis() {
        // Starts mid-row.
        assert_eq!(spec(&[1000, 1024], DType::Float32, 8, 4096).dims(), [1024]);
        // Ends mid-row.
        assert_eq!(
            spec(&[1000, 1024], DType::Float32, 0, 4096 + 8).dims(),
            [1026]
        );
        // A row longer than the whole block.
        assert_eq!(spec(&[4, 1 << 20], DType::Float32, 0, 4096).dims(), [1024]);
    }

    #[test]
    fn a_length_one_axis_is_folded_away() {
        // The compressors refuse one, and dropping it leaves element order alone.
        assert_eq!(spec(&[1000, 1], DType::Float64, 0, 8 * 64).dims(), [64]);
        assert_eq!(
            spec(&[64, 1, 1024], DType::Float32, 0, 4 * 1024).dims(),
            [1024]
        );
        assert_eq!(spec(&[1000, 1024], DType::Float32, 0, 4096).dims(), [1024]);
    }

    #[test]
    fn a_scalar_and_an_empty_shape_still_describe_a_block() {
        assert_eq!(spec(&[], DType::Float32, 0, 4).dims(), [1]);
        assert_eq!(spec(&[1], DType::Float32, 0, 4).dims(), [1]);
        assert_eq!(spec(&[8, 8], DType::Float64, 0, 8).dims(), [1]);
    }

    #[test]
    fn a_range_that_is_not_whole_elements_is_refused() {
        assert!(BlockSpec::for_range(&[64], DType::Float32, 1e-3, 0, 6).is_err());
        assert!(BlockSpec::for_range(&[64], DType::Float32, 1e-3, 0, 0).is_err());
    }

    #[test]
    fn a_header_is_twenty_four_bytes_with_the_documented_layout() {
        let s = spec(&[1000, 1024], DType::Float64, 0, 1024 * 1024);
        let mut out = Vec::new();
        s.encode(&mut out);
        assert_eq!(out.len(), BLOCK_HEADER_LEN);
        assert_eq!(out[0], VERSION);
        assert_eq!(out[1], DType::Float64.as_i32() as u8);
        assert_eq!(out[2], 2);
        assert_eq!(out[3], 0);
        assert_eq!(f64::from_le_bytes(out[4..12].try_into().unwrap()), 1e-3);
        assert_eq!(u32::from_le_bytes(out[12..16].try_into().unwrap()), 128);
        assert_eq!(u32::from_le_bytes(out[16..20].try_into().unwrap()), 1024);
        assert_eq!(u32::from_le_bytes(out[20..24].try_into().unwrap()), 0);
    }

    #[test]
    fn a_header_this_build_cannot_trust_is_reported_rather_than_guessed() {
        let s = spec(&[1000, 1024], DType::Float32, 0, 512 * 1024);
        let mut good = Vec::new();
        s.encode(&mut good);
        assert_eq!(BlockSpec::decode(&good).unwrap(), s);

        let mutate = |at: usize, to: u8| {
            let mut bytes = good.clone();
            bytes[at] = to;
            BlockSpec::decode(&bytes)
        };
        assert!(mutate(0, VERSION + 1).is_err(), "a newer format");
        assert!(mutate(1, 200).is_err(), "an unknown dtype");
        assert!(mutate(2, 0).is_err(), "no axes");
        assert!(
            mutate(2, MAX_BLOCK_DIMS as u8 + 1).is_err(),
            "too many axes"
        );
        assert!(mutate(2, 3).is_err(), "an axis count the dims do not fill");
        assert!(mutate(12, 0).is_err(), "an empty axis");
        assert!(mutate(20, 4).is_err(), "an axis past the count");
        assert!(
            BlockSpec::decode(&good[..BLOCK_HEADER_LEN - 1]).is_err(),
            "a short header"
        );

        let mut nan = good.clone();
        nan[4..12].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(
            BlockSpec::decode(&nan).is_err(),
            "an error bound that is not one"
        );
        let mut zero = good.clone();
        zero[4..12].copy_from_slice(&0f64.to_le_bytes());
        assert!(BlockSpec::decode(&zero).is_err(), "a zero error bound");
    }

    proptest! {
        /// Whatever the shape and the range, the block describes exactly the
        /// elements the range holds, and survives a header roundtrip.
        #[test]
        fn any_range_is_described_and_survives_a_roundtrip(
            shape in prop::collection::vec(1u64..40, 1..4),
            dtype in prop::sample::select(vec![DType::Float32, DType::Float64]),
            start in 0usize..64,
            take in 1usize..64,
        ) {
            let itemsize = dtype.itemsize();
            let total = shape.iter().product::<u64>() * itemsize;
            let offset = (start as u64 * itemsize) % total;
            let len = ((take as u64 * itemsize) % (total - offset)) + itemsize;

            let s = BlockSpec::for_range(&shape, dtype, 1e-3, offset, len)?;
            prop_assert_eq!(s.bytes(), len);
            prop_assert!(!s.dims().is_empty() && s.dims().len() <= MAX_BLOCK_DIMS);
            prop_assert!(s.dims().iter().all(|d| *d > 1) || s.dims().len() == 1);

            let mut bytes = Vec::new();
            s.encode(&mut bytes);
            prop_assert_eq!(BlockSpec::decode(&bytes)?, s);
        }
    }
}
