//! zfp, through the raw `zfp-sys` bindings.
//!
//! That crate vendors zfp and builds it with cmake, binding it with bindgen,
//! which is why this is behind a feature and not simply always here. It is
//! linked statically, which also leaves zfp's OpenMP out: a connection thread
//! compresses on its own core, as SZ3 does, so the number of connections stays
//! the only thing that decides how many cores compress.
//!
//! **zfp does not survive NaN or infinity.** A block holding one comes back
//! with the rest of that block's values wrong by far more than the bound, and
//! there is no cheap way for zfp to say so. SZ3 has no such limit. Nothing
//! here scans for them, because the scan would cost a pass over every block to
//! defend against data zfp's own documentation tells you not to give it.
//!
//! Nothing writes zfp's own stream header. Element type, shape and error bound
//! are in the AEX block header the receiver has already read, so a second copy
//! would only be a second thing to disagree with.

use std::cell::RefCell;
use std::os::raw::c_void;
use std::ptr;

use zfp_sys::{
    bitstream, stream_close, stream_open, zfp_compress, zfp_decompress, zfp_field, zfp_field_1d,
    zfp_field_2d, zfp_field_3d, zfp_field_free, zfp_stream, zfp_stream_close,
    zfp_stream_maximum_size, zfp_stream_open, zfp_stream_rewind, zfp_stream_set_accuracy,
    zfp_stream_set_bit_stream, zfp_type, zfp_type_zfp_type_double, zfp_type_zfp_type_float,
};

use super::{load, store, BlockSpec, F32, F64};
use crate::dtype::DType;
use crate::error::{AexError, Result};

thread_local! {
    /// Room for the bit stream, in words rather than bytes: zfp reads and
    /// writes it 64 bits at a time and a `Vec<u8>` is not guaranteed to start
    /// on a word boundary.
    static WORDS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn compress(spec: &BlockSpec, src: &[u8], dst: &mut Vec<u8>) -> Result<()> {
    match spec.dtype {
        DType::Float32 => with_scratch(|values: &mut Vec<f32>, words| {
            compress_as(spec, src, dst, values, words, zfp_type_zfp_type_float)
        }),
        DType::Float64 => with_scratch(|values: &mut Vec<f64>, words| {
            compress_as(spec, src, dst, values, words, zfp_type_zfp_type_double)
        }),
        other => Err(unsupported(other)),
    }
}

pub(super) fn decompress_into(spec: &BlockSpec, src: &[u8], dst: &mut [u8]) -> Result<()> {
    match spec.dtype {
        DType::Float32 => with_scratch(|values: &mut Vec<f32>, words| {
            expand_as(spec, src, dst, values, words, zfp_type_zfp_type_float)
        }),
        DType::Float64 => with_scratch(|values: &mut Vec<f64>, words| {
            expand_as(spec, src, dst, values, words, zfp_type_zfp_type_double)
        }),
        other => Err(unsupported(other)),
    }
}

fn compress_as<T: Copy + Default>(
    spec: &BlockSpec,
    src: &[u8],
    dst: &mut Vec<u8>,
    values: &mut Vec<T>,
    words: &mut Vec<u64>,
    ty: zfp_type,
) -> Result<()> {
    load(src, values);
    let mut session = Session::open(spec, values.as_mut_ptr().cast(), ty)?;
    let room = session.maximum_size();
    words.clear();
    words.resize(room.div_ceil(8), 0);
    session.attach(words);

    // SAFETY: the field spans exactly `values`, and the bit stream exactly
    // `words`, which zfp's own bound says is enough for the field.
    let written = unsafe { zfp_compress(session.zfp, session.field) };
    if written == 0 {
        return Err(refused("compressing a block"));
    }
    // Appends after the caller's block header.
    dst.extend_from_slice(&as_bytes(words)[..written]);
    Ok(())
}

fn expand_as<T: Copy + Default>(
    spec: &BlockSpec,
    src: &[u8],
    dst: &mut [u8],
    values: &mut Vec<T>,
    words: &mut Vec<u64>,
    ty: zfp_type,
) -> Result<()> {
    values.clear();
    values.resize(spec.count() as usize, T::default());
    // The received bytes start wherever the frame buffer put them, so they are
    // copied onto a word boundary before zfp reads them. It is the compressed
    // size that is copied, not the block's, which is why it does not show up.
    words.clear();
    words.resize(src.len().div_ceil(8), 0);
    // SAFETY: `words` holds at least src.len() bytes, and both are plain data.
    unsafe {
        ptr::copy_nonoverlapping(src.as_ptr(), words.as_mut_ptr().cast::<u8>(), src.len());
    }

    let mut session = Session::open(spec, values.as_mut_ptr().cast(), ty)?;
    session.attach(words);
    // SAFETY: as in `compress_as`. A stream that does not describe this field
    // is caught by zfp returning zero, since it reads a fixed number of
    // blocks and runs off the end of the buffer.
    if unsafe { zfp_decompress(session.zfp, session.field) } == 0 {
        return Err(refused("expanding a block"));
    }
    store(values, dst);
    Ok(())
}

/// The three C objects a zfp call needs, freed on every path out.
struct Session {
    field: *mut zfp_field,
    zfp: *mut zfp_stream,
    bits: *mut bitstream,
}

impl Session {
    /// A field over `values` and a stream set to the block's error bound.
    ///
    /// The bit stream is attached later: how much room it needs is not known
    /// until zfp has both of these.
    fn open(spec: &BlockSpec, values: *mut c_void, ty: zfp_type) -> Result<Self> {
        // zfp's axes run fastest first, and a block's run fastest last.
        let d: Vec<usize> = spec.dims().iter().rev().map(|d| *d as usize).collect();
        // SAFETY: every argument is a length or a pointer to `values`, which
        // the caller sized from the same spec.
        let field = unsafe {
            match d[..] {
                [nx] => zfp_field_1d(values, ty, nx),
                [nx, ny] => zfp_field_2d(values, ty, nx, ny),
                [nx, ny, nz] => zfp_field_3d(values, ty, nx, ny, nz),
                _ => ptr::null_mut(),
            }
        };
        let mut session = Session {
            field,
            zfp: ptr::null_mut(),
            bits: ptr::null_mut(),
        };
        if session.field.is_null() {
            return Err(refused("describing a block"));
        }
        // SAFETY: a null bit stream is what zfp_stream_open takes when the
        // buffer is not known yet.
        session.zfp = unsafe { zfp_stream_open(ptr::null_mut()) };
        if session.zfp.is_null() {
            return Err(refused("opening a stream"));
        }
        // SAFETY: the stream is live, and the bound is finite and positive by
        // the block header's own check.
        unsafe { zfp_stream_set_accuracy(session.zfp, spec.eps) };
        Ok(session)
    }

    /// How many bytes this field can take at worst.
    fn maximum_size(&self) -> usize {
        // SAFETY: both are live for the life of the session.
        unsafe { zfp_stream_maximum_size(self.zfp, self.field) }
    }

    /// Point the stream at `words` and rewind it.
    fn attach(&mut self, words: &mut [u64]) {
        // SAFETY: the bit stream borrows `words` for as long as the session
        // lives, and the session is dropped before the caller reuses them.
        self.bits = unsafe { stream_open(words.as_mut_ptr().cast(), std::mem::size_of_val(words)) };
        unsafe {
            zfp_stream_set_bit_stream(self.zfp, self.bits);
            zfp_stream_rewind(self.zfp);
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: each is either null or was handed back by its opener, and
        // nothing else frees them.
        unsafe {
            if !self.bits.is_null() {
                stream_close(self.bits);
            }
            if !self.zfp.is_null() {
                zfp_stream_close(self.zfp);
            }
            if !self.field.is_null() {
                zfp_field_free(self.field);
            }
        }
    }
}

fn as_bytes(words: &[u64]) -> &[u8] {
    // SAFETY: any bit pattern is a valid u8, and the span is the same.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words)) }
}

/// Run `go` with this thread's element and bit-stream scratch.
fn with_scratch<T, R>(go: impl FnOnce(&mut Vec<T>, &mut Vec<u64>) -> R) -> R
where
    Vec<T>: ScratchOf,
{
    <Vec<T> as ScratchOf>::with(|values| WORDS.with(|words| go(values, &mut words.borrow_mut())))
}

/// Which of the parent module's element buffers holds a given element type.
trait ScratchOf: Sized {
    fn with<R>(go: impl FnOnce(&mut Self) -> R) -> R;
}

impl ScratchOf for Vec<f32> {
    fn with<R>(go: impl FnOnce(&mut Self) -> R) -> R {
        F32.with(|s| go(&mut s.borrow_mut()))
    }
}

impl ScratchOf for Vec<f64> {
    fn with<R>(go: impl FnOnce(&mut Self) -> R) -> R {
        F64.with(|s| go(&mut s.borrow_mut()))
    }
}

fn unsupported(dtype: DType) -> AexError {
    AexError::BadBlock(format!("zfp compresses floats, not {dtype}"))
}

fn refused(what: &str) -> AexError {
    AexError::BadBlock(format!("zfp refused {what}"))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{roundtrip_f32, smooth};
    use super::super::{compress, decompress_into, BlockSpec};
    use crate::dtype::DType;
    use crate::quality::Codec;

    fn roundtrip(out_shape: &[u64], eps: f64, values: &[f32]) -> (f64, usize) {
        roundtrip_f32(Codec::Zfp, out_shape, eps, values)
    }

    #[test]
    fn every_element_comes_back_within_the_bound() {
        let values = smooth(128, 1024);
        for eps in [1e-6, 1e-3, 1e-1, 1.0] {
            let (worst, _) = roundtrip(&[128, 1024], eps, &values);
            assert!(worst <= eps, "worst {worst:e} over a bound of {eps:e}");
        }
    }

    #[test]
    fn a_bound_is_met_by_rounding_it_down_to_a_power_of_two() {
        // zfp drops whole bit planes, so what it honours is the largest power
        // of two at or below the bound. Asking for 1e-1 gets 2^-4, and the
        // ratio is the one that bound would have earned -- worth knowing
        // before comparing a ratio against SZ3's at the same number.
        let values = smooth(128, 1024);
        let (worst, _) = roundtrip(&[128, 1024], 0.1, &values);
        assert!(
            worst <= 0.0625,
            "worst {worst:e} over the 2^-4 zfp rounds to"
        );
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
        assert!(compress(Codec::Zfp, &spec, &src, &mut packed).unwrap());
        let mut out = vec![0u8; src.len()];
        decompress_into(Codec::Zfp, &packed, &mut out).unwrap();
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
        // zfp's blocks are 4 wide in each axis, so a flat run sees a quarter
        // of the neighbours a 2-d block does.
        let values = smooth(128, 1024);
        let eps = 1e-2;
        let (_, slab) = roundtrip(&[128, 1024], eps, &values);
        let (_, flat) = roundtrip(&[128 * 1024], eps, &values);
        assert!(
            slab < flat,
            "a slab took {slab} bytes and a flat run {flat}; the shape should be worth something"
        );
    }

    #[test]
    fn three_axes_roundtrip_as_well_as_two() {
        let values = smooth(64, 1024);
        let (worst, _) = roundtrip(&[16, 4, 1024], 1e-3, &values);
        assert!(worst <= 1e-3, "worst {worst:e} over a bound of 1e-3");
    }

    #[test]
    fn a_block_too_small_to_pay_for_itself_is_reported_rather_than_grown() {
        // zfp pads a partial block out to 4 values, so a single one cannot
        // shrink. The caller sends those raw.
        let src = 1.5f32.to_le_bytes().to_vec();
        let spec = BlockSpec::for_range(&[1], DType::Float32, 1e-3, 0, 4).unwrap();
        let mut packed = Vec::new();
        assert!(!compress(Codec::Zfp, &spec, &src, &mut packed).unwrap());
        assert!(packed.is_empty());
    }

    #[test]
    fn a_block_of_one_value_repeated_still_roundtrips() {
        let (worst, _) = roundtrip(&[64, 64], 1e-3, &vec![7.0f32; 4096]);
        assert_eq!(worst, 0.0);
    }

    #[test]
    fn an_integer_block_is_refused_rather_than_mangled() {
        let src = vec![0u8; 4096];
        let spec = BlockSpec::for_range(&[64, 16], DType::Int32, 1e-3, 0, 4096).unwrap();
        let mut packed = Vec::new();
        assert!(compress(Codec::Zfp, &spec, &src, &mut packed).is_err());
    }

    #[test]
    fn a_block_that_does_not_fill_the_destination_is_refused() {
        let values = smooth(64, 64);
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let spec =
            BlockSpec::for_range(&[64, 64], DType::Float32, 1e-3, 0, src.len() as u64).unwrap();
        let mut packed = Vec::new();
        assert!(compress(Codec::Zfp, &spec, &src, &mut packed).unwrap());
        let mut too_small = vec![0u8; src.len() - 4];
        assert!(decompress_into(Codec::Zfp, &packed, &mut too_small).is_err());
    }
}
