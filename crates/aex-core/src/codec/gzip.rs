//! Deflate, through `flate2`.
//!
//! The one codec here that does not look at the elements: it takes the block's
//! bytes as bytes. The block header still goes in front of the stream, so that
//! a receiver checks the same things for every codec and the shape is there if
//! a later reader wants it.

use std::io::Write;

use super::BlockSpec;
use crate::error::{AexError, Result};

/// What `gzip -6` uses, and the level every published ratio is quoted at.
const LEVEL: flate2::Compression = flate2::Compression::new(6);

pub(super) fn compress(_spec: &BlockSpec, src: &[u8], dst: &mut Vec<u8>) -> Result<()> {
    // Appends after the caller's block header.
    let mut encoder = flate2::write::DeflateEncoder::new(dst, LEVEL);
    encoder.write_all(src).map_err(refused)?;
    encoder.finish().map_err(refused)?;
    Ok(())
}

pub(super) fn decompress_into(_spec: &BlockSpec, src: &[u8], dst: &mut [u8]) -> Result<()> {
    let mut decoder = flate2::write::DeflateDecoder::new(dst);
    decoder.write_all(src).map_err(refused)?;
    // Checks that the stream ended where it said it would, and that it filled
    // the block exactly: a short stream leaves the tail of `dst` untouched,
    // which would otherwise pass for data.
    let dst = decoder.finish().map_err(refused)?;
    if !dst.is_empty() {
        return Err(AexError::BadBlock(format!(
            "a deflate stream left {} bytes of its block unwritten",
            dst.len()
        )));
    }
    Ok(())
}

fn refused(e: std::io::Error) -> AexError {
    AexError::BadBlock(format!("deflate: {e}"))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{roundtrip_f32, smooth};
    use super::super::{compress, decompress_into, BlockSpec, BLOCK_HEADER_LEN};
    use crate::dtype::DType;
    use crate::quality::Codec;

    #[test]
    fn a_block_comes_back_byte_for_byte() {
        let values = smooth(128, 1024);
        // The bound is in the header and nothing reads it, so any value will
        // do; what matters is that lossless means lossless.
        let (worst, packed) = roundtrip_f32(Codec::Gzip, &[128, 1024], 1e-3, &values);
        assert_eq!(worst, 0.0, "deflate is lossless");
        assert!(packed < values.len() * 4, "a smooth field should shrink");
    }

    #[test]
    fn a_block_that_does_not_shrink_is_left_to_the_caller() {
        // Bytes with no structure: deflate can only add its own framing.
        let src: Vec<u8> = (0..4096u32)
            .flat_map(|i| i.wrapping_mul(2654435761).to_le_bytes())
            .collect();
        let spec = BlockSpec::for_range(&[4096], DType::Uint32, 0.0, 0, src.len() as u64).unwrap();
        let mut packed = Vec::new();
        assert!(!compress(Codec::Gzip, &spec, &src, &mut packed).unwrap());
        assert!(packed.is_empty(), "a refused block leaves nothing behind");
    }

    #[test]
    fn a_truncated_stream_is_refused_rather_than_half_written() {
        let values = smooth(16, 256);
        let src: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let spec =
            BlockSpec::for_range(&[16, 256], DType::Float32, 0.0, 0, src.len() as u64).unwrap();
        let mut packed = Vec::new();
        assert!(compress(Codec::Gzip, &spec, &src, &mut packed).unwrap());

        packed.truncate(packed.len() - 8);
        let mut out = vec![0u8; src.len()];
        // Either the stream itself complains or the block comes up short; the
        // point is that neither leaves half a block passing for data.
        decompress_into(Codec::Gzip, &packed, &mut out).unwrap_err();
        // The header alone decodes cleanly to nothing, so only the length
        // check catches it.
        let err = decompress_into(Codec::Gzip, &packed[..BLOCK_HEADER_LEN], &mut out).unwrap_err();
        assert!(err.to_string().contains("unwritten"), "{err}");
    }
}
