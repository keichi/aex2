//! SZ3, through the `sz3` crate.
//!
//! That crate vendors SZ3 and builds it with cmake and a C++ compiler, which is
//! why this is behind a feature and not simply always here.
//!
//! SZ3 takes and returns typed arrays while the logical stream is bytes, so
//! every block goes through a scratch buffer. The copy is a few percent of what
//! the compressor itself costs, and it means nothing here depends on a byte
//! buffer happening to be aligned for floats.

use std::cell::RefCell;
use std::mem::size_of;

use super::BlockSpec;
use crate::dtype::DType;
use crate::error::{AexError, Result};

thread_local! {
    /// Element-typed scratch, so a transfer in progress never allocates.
    static F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static F64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn compress(spec: &BlockSpec, src: &[u8], dst: &mut Vec<u8>) -> Result<()> {
    match spec.dtype {
        DType::Float32 => F32.with(|s| compress_as::<f32>(spec, src, dst, &mut s.borrow_mut())),
        DType::Float64 => F64.with(|s| compress_as::<f64>(spec, src, dst, &mut s.borrow_mut())),
        other => Err(unsupported(other)),
    }
}

pub(super) fn decompress_into(spec: &BlockSpec, src: &[u8], dst: &mut [u8]) -> Result<()> {
    match spec.dtype {
        DType::Float32 => F32.with(|s| expand_as::<f32>(spec, src, dst, &mut s.borrow_mut())),
        DType::Float64 => F64.with(|s| expand_as::<f64>(spec, src, dst, &mut s.borrow_mut())),
        other => Err(unsupported(other)),
    }
}

fn compress_as<T: sz3::SZ3Compressible + Copy + Default>(
    spec: &BlockSpec,
    src: &[u8],
    dst: &mut Vec<u8>,
    scratch: &mut Vec<T>,
) -> Result<()> {
    load(src, scratch);
    let values: &[T] = scratch;
    let mut builder = sz3::DimensionedData::<T, &[T]>::build(&values);
    for dim in spec.dims() {
        builder = builder.dim(*dim as usize).map_err(refused)?;
    }
    let data = builder.finish().map_err(refused)?;
    let config = sz3::Config::new(sz3::ErrorBound::Absolute(spec.eps));
    // Appends, which is what puts the stream straight after the block header
    // the caller has already written.
    sz3::compress_into_with_config(&data, &config, dst).map_err(refused)
}

fn expand_as<T: sz3::SZ3Compressible + Copy + Default>(
    spec: &BlockSpec,
    src: &[u8],
    dst: &mut [u8],
    scratch: &mut Vec<T>,
) -> Result<()> {
    scratch.clear();
    scratch.resize(spec.count() as usize, T::default());
    let mut values: &mut [T] = scratch;
    let mut builder = sz3::DimensionedData::<T, &mut [T]>::build_mut(&mut values);
    for dim in spec.dims() {
        builder = builder.dim(*dim as usize).map_err(refused)?;
    }
    let mut data = builder.finish().map_err(refused)?;
    // Checks the stream's own shape against ours, so a block that belongs to
    // another transfer is refused rather than written somewhere.
    sz3::decompress_into_dimensioned(src, &mut data).map_err(refused)?;
    store(scratch, dst);
    Ok(())
}

/// Read the block's bytes as elements.
///
/// The logical stream is little-endian and so is every target this builds for,
/// so the elements are already laid out the way they are wanted.
fn load<T: Copy + Default>(src: &[u8], scratch: &mut Vec<T>) {
    let count = src.len() / size_of::<T>();
    scratch.clear();
    scratch.resize(count, T::default());
    // SAFETY: both sides span exactly count * size_of::<T>() bytes, and a
    // float has no invalid bit pattern.
    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr(),
            scratch.as_mut_ptr().cast::<u8>(),
            count * size_of::<T>(),
        );
    }
}

fn store<T: Copy>(scratch: &[T], dst: &mut [u8]) {
    debug_assert_eq!(std::mem::size_of_val(scratch), dst.len());
    // SAFETY: as in `load`, and the lengths are equal by the caller's check.
    unsafe {
        std::ptr::copy_nonoverlapping(scratch.as_ptr().cast::<u8>(), dst.as_mut_ptr(), dst.len());
    }
}

fn unsupported(dtype: DType) -> AexError {
    AexError::BadBlock(format!("SZ3 compresses floats, not {dtype}"))
}

fn refused(e: sz3::SZ3Error) -> AexError {
    AexError::BadBlock(format!("SZ3: {e}"))
}

#[cfg(test)]
mod tests {
    use super::super::{compress, decompress_into, BlockSpec};
    use crate::dtype::DType;
    use crate::quality::Codec;

    /// Compress and expand one block, and report the worst error seen.
    fn roundtrip_f32(out_shape: &[u64], eps: f64, values: &[f32]) -> (f64, usize) {
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let spec =
            BlockSpec::for_range(out_shape, DType::Float32, eps, 0, src.len() as u64).unwrap();
        let mut packed = Vec::new();
        let shrank = compress(Codec::Sz, &spec, &src, &mut packed).unwrap();
        if !shrank {
            return (0.0, src.len());
        }
        let mut out = vec![0u8; src.len()];
        decompress_into(Codec::Sz, &packed, &mut out).unwrap();
        let worst = out
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .zip(values)
            .filter(|(_, want)| want.is_finite())
            .map(|(got, want)| (got as f64 - *want as f64).abs())
            .fold(0.0f64, f64::max);
        (worst, packed.len())
    }

    fn smooth(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                (c as f32 / 32.0).sin() * 100.0 + r as f32 * 0.25
            })
            .collect()
    }

    #[test]
    fn every_element_comes_back_within_the_bound() {
        let values = smooth(128, 1024);
        for eps in [1e-6, 1e-3, 1e-1, 1.0] {
            let (worst, _) = roundtrip_f32(&[128, 1024], eps, &values);
            assert!(worst <= eps, "worst {worst:e} over a bound of {eps:e}");
        }
    }

    #[test]
    fn float64_comes_back_within_the_bound_too() {
        let values: Vec<f64> = (0..64 * 512)
            .map(|i| (i as f64 / 32.0).sin() * 100.0)
            .collect();
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let eps = 1e-5;
        let spec =
            BlockSpec::for_range(&[64, 512], DType::Float64, eps, 0, src.len() as u64).unwrap();
        let mut packed = Vec::new();
        assert!(compress(Codec::Sz, &spec, &src, &mut packed).unwrap());
        let mut out = vec![0u8; src.len()];
        decompress_into(Codec::Sz, &packed, &mut out).unwrap();
        let worst = out
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .zip(&values)
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f64, f64::max);
        assert!(worst <= eps, "worst {worst:e} over a bound of {eps:e}");
    }

    #[test]
    fn telling_it_the_shape_is_what_earns_the_ratio() {
        // The same bytes, described as the slab they are and as one long row.
        let values = smooth(128, 1024);
        let eps = 1e-2;
        let (_, slab) = roundtrip_f32(&[128, 1024], eps, &values);
        let (_, flat) = roundtrip_f32(&[128 * 1024], eps, &values);
        assert!(
            slab * 2 < flat,
            "a slab took {slab} bytes and a flat run {flat}; the shape should be worth far more"
        );
    }

    #[test]
    fn non_finite_elements_leave_their_neighbours_alone() {
        let mut values = smooth(64, 64);
        values[100] = f32::NAN;
        values[200] = f32::INFINITY;
        values[300] = f32::NEG_INFINITY;
        let eps = 1e-3;
        let (worst, _) = roundtrip_f32(&[64, 64], eps, &values);
        assert!(worst <= eps, "worst {worst:e} over a bound of {eps:e}");
    }

    #[test]
    fn a_block_too_small_to_pay_for_itself_is_reported_rather_than_grown() {
        // SZ3 writes tens of bytes of its own header, so a handful of elements
        // never shrinks. The caller sends those raw.
        let src = 1.5f32.to_le_bytes().to_vec();
        let spec = BlockSpec::for_range(&[1], DType::Float32, 1e-3, 0, 4).unwrap();
        let mut packed = Vec::new();
        assert!(!compress(Codec::Sz, &spec, &src, &mut packed).unwrap());
        assert!(packed.is_empty());
    }

    #[test]
    fn a_block_of_one_value_repeated_still_roundtrips() {
        let (worst, _) = roundtrip_f32(&[64, 64], 1e-3, &vec![7.0f32; 4096]);
        assert_eq!(worst, 0.0);
    }

    #[test]
    fn an_integer_block_is_refused_rather_than_mangled() {
        let src = vec![0u8; 4096];
        let spec = BlockSpec::for_range(&[64, 16], DType::Int32, 1e-3, 0, 4096).unwrap();
        let mut packed = Vec::new();
        assert!(compress(Codec::Sz, &spec, &src, &mut packed).is_err());
    }

    #[test]
    fn a_block_that_does_not_fill_the_destination_is_refused() {
        let values = smooth(64, 64);
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let spec =
            BlockSpec::for_range(&[64, 64], DType::Float32, 1e-3, 0, src.len() as u64).unwrap();
        let mut packed = Vec::new();
        assert!(compress(Codec::Sz, &spec, &src, &mut packed).unwrap());
        let mut too_small = vec![0u8; src.len() - 4];
        assert!(decompress_into(Codec::Sz, &packed, &mut too_small).is_err());
    }
}
