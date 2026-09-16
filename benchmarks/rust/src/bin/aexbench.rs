//! End to end: what a client actually gets out of AEX2.
//!
//! The selection is a contiguous range of a 1-d array, read into a buffer
//! allocated once, so that what is timed is the transfer and not an allocator.

use std::time::Instant;

use aex_bench::{cpu_seconds, report, Run};
use aex_client::{Client, ClientConfig, Index};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Time reading a contiguous selection over AEX2")]
struct Cli {
    /// Control plane endpoint, e.g. http://127.0.0.1:50051
    url: String,
    /// File to open, relative to one of the server's data roots.
    file: String,
    /// Bytes to read per run. The file is float32, so this is four times the
    /// number of elements the selection covers.
    #[arg(long)]
    bytes: u64,
    /// Bytes per fetch. 0 follows the server's recommendation.
    #[arg(long, default_value_t = 0)]
    chunk: u64,
    #[arg(long, default_value_t = 5)]
    reps: usize,
    #[arg(long)]
    label: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let client = Client::connect(
        &cli.url,
        ClientConfig {
            chunk_bytes: cli.chunk,
            ..ClientConfig::default()
        },
    )?;
    let handle = client.open(&cli.file)?;

    let elements = cli.bytes / 4;
    let indices = [Index::range(0, elements as i64)];
    let mut dst = vec![0u8; cli.bytes as usize];

    let mut runs = Vec::with_capacity(cli.reps);
    for _ in 0..cli.reps {
        let cpu = cpu_seconds();
        let started = Instant::now();
        let result = client.read_selection_into(handle, "array", &indices, &mut dst)?;
        let elapsed = started.elapsed();
        // A retry means some of the time went somewhere the number cannot
        // explain, so the run is not one to average in.
        assert_eq!(result.retries, 0, "a fetch had to be retried");
        runs.push(Run {
            bytes: result.bytes,
            elapsed,
            cpu: cpu_seconds() - cpu,
        });
    }

    // Cheap proof that the right bytes arrived, not merely the right number of
    // them: the file counts up, so the ends say where they came from.
    let value_at = |i: usize| f32::from_le_bytes(dst[i..i + 4].try_into().expect("4 bytes"));
    assert_eq!(value_at(0), 0.0, "the transfer did not start at the start");
    assert_eq!(
        value_at(dst.len() - 4),
        (elements - 1) as f32,
        "the transfer did not end at the end"
    );

    let label = cli.label.unwrap_or_else(|| cli.file.clone());
    report(&format!("aex {label} chunk={}", cli.chunk), &runs);
    client.disconnect()?;
    Ok(())
}
