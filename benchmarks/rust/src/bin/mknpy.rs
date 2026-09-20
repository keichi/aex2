//! Write a float32 `.npy` fixture.
//!
//! numpy can do this too, but not at the sizes the disk-resident case needs
//! without pushing the whole array through memory first.
//!
//! Counting up makes every element say where it came from, so a transfer that
//! lands one chunk off is a visible mismatch rather than a plausible number.
//! It is useless for measuring compression, though: a ramp is exactly what a
//! predictor is built for and the ratio it produces says nothing about real
//! data. `--field wave` writes something a compressor has to work at.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// What the elements are.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Field {
    /// Element i is i. Says where every element came from; read with
    /// `--pattern counting-f32`.
    Counting,
    /// A smooth 2-d field with noise on top: smooth enough that predicting
    /// from neighbours pays, noisy enough that it does not pay unboundedly.
    /// Read with `--pattern none`.
    Wave,
}

#[derive(Parser)]
#[command(about = "Write a float32 .npy fixture")]
struct Cli {
    path: PathBuf,
    /// Number of float32 elements; the file is four times this, plus a header.
    elements: u64,
    /// Elements in a row. 0 writes a 1-d array.
    #[arg(long, default_value_t = 0)]
    row_elements: u64,
    #[arg(long, value_enum, default_value_t = Field::Counting)]
    field: Field,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    if cli.row_elements > 0 && cli.elements % cli.row_elements != 0 {
        eprintln!(
            "{} elements is not a whole number of {} element rows",
            cli.elements, cli.row_elements
        );
        std::process::exit(2);
    }

    let shape = match cli.row_elements {
        0 => format!("{},", cli.elements),
        row => format!("{}, {row}", cli.elements / row),
    };
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape}), }}");
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
            buf.extend_from_slice(&value(&cli, written + i).to_le_bytes());
        }
        out.write_all(&buf)?;
        written += n as u64;
    }
    out.flush()?;

    println!(
        "{}: shape ({shape}) float32, {} bytes of data",
        cli.path.display(),
        cli.elements * 4
    );
    Ok(())
}

/// The element at flat position `at`.
fn value(cli: &Cli, at: u64) -> f32 {
    match cli.field {
        Field::Counting => at as f32,
        Field::Wave => {
            let row = cli.row_elements.max(1);
            let (y, x) = ((at / row) as f32, (at % row) as f32);
            // Two scales of smooth structure, and a deterministic wiggle in the
            // last few bits so that the field is not exactly representable by
            // its own predictor.
            let smooth =
                (x / 37.0).sin() * 100.0 + (y / 53.0).cos() * 40.0 + ((x + y) / 211.0).sin() * 10.0;
            let hash = at
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            smooth + ((hash >> 40) as f32 / (1 << 24) as f32 - 0.5)
        }
    }
}
