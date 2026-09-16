//! Write a 1-d float32 `.npy` counting up from zero.
//!
//! numpy can do this too, but not at the sizes the disk-resident case needs
//! without pushing the whole array through memory first.
//!
//! Counting up makes every element say where it came from, so a transfer that
//! lands one chunk off is a visible mismatch rather than a plausible number.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(about = "Write a 1-d float32 .npy counting up from zero")]
struct Cli {
    path: PathBuf,
    /// Number of float32 elements; the file is four times this, plus a header.
    elements: u64,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        cli.elements
    );
    // A v1.0 header: magic, version, a 2-byte length, then text padded so that
    // the data starts on a 64-byte boundary.
    let padding = (64 - (10 + dict.len() + 1) % 64) % 64;
    let header_len = dict.len() + padding + 1;

    let mut out = BufWriter::with_capacity(8 << 20, File::create(&cli.path)?);
    out.write_all(b"\x93NUMPY\x01\x00")?;
    out.write_all(&(header_len as u16).to_le_bytes())?;
    out.write_all(dict.as_bytes())?;
    out.write_all(&vec![b' '; padding])?;
    out.write_all(b"\n")?;

    const BATCH: usize = 1 << 20;
    let mut buf: Vec<u8> = Vec::with_capacity(BATCH * 4);
    let mut written = 0u64;
    while written < cli.elements {
        buf.clear();
        let n = BATCH.min((cli.elements - written) as usize);
        for i in 0..n as u64 {
            buf.extend_from_slice(&((written + i) as f32).to_le_bytes());
        }
        out.write_all(&buf)?;
        written += n as u64;
    }
    out.flush()?;

    println!(
        "{}: {} float32, {} bytes of data",
        cli.path.display(),
        cli.elements,
        cli.elements * 4
    );
    Ok(())
}
