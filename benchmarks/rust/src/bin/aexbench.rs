//! End to end: what a client actually gets out of AEX2.
//!
//! The selection is a range of a 1-d array, contiguous unless `--step` says
//! otherwise, read into a buffer allocated once, so that what is timed is the
//! transfer and not an allocator.

use std::time::Instant;

use aex_bench::{cpu_seconds, report, Run};
use aex_client::{Client, ClientConfig, Index};
use clap::{Parser, ValueEnum};

/// What the bytes of the fixture are.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Pattern {
    /// float32 counting up from zero: what `mknpy` writes.
    CountingF32,
    /// The byte at position p is p as u8: what the synthetic backend serves.
    Bytes,
    /// Do not check. Only for a fixture whose content is not known.
    None,
}

#[derive(Parser)]
#[command(about = "Time reading a contiguous selection over AEX2")]
struct Cli {
    /// Control plane endpoint, e.g. http://127.0.0.1:50051
    url: String,
    /// File to open, relative to one of the server's data roots. With
    /// `--format null`, a synthetic dataset such as `uint8:4294967296`.
    file: String,
    /// Format to open it as. Empty lets the server infer it from the name.
    #[arg(long, default_value = "")]
    format: String,
    /// Bytes to read per run.
    #[arg(long)]
    bytes: u64,
    /// Size of one element, for turning `--bytes` into a selection.
    #[arg(long, default_value_t = 4)]
    itemsize: u64,
    /// What the data should look like, so that a transfer which moved the
    /// wrong bytes fails rather than posting a good number.
    #[arg(long, value_enum, default_value_t = Pattern::CountingF32)]
    pattern: Pattern,
    /// Bytes per fetch. 0 follows the server's recommendation.
    #[arg(long, default_value_t = 0)]
    chunk: u64,
    /// Data connections.
    #[arg(long, default_value_t = aex_client::config::DEFAULT_STREAMS)]
    streams: u32,
    /// Fetches outstanding per connection.
    #[arg(long, default_value_t = aex_client::config::DEFAULT_CREDIT)]
    credit: u32,
    /// Take every step-th element. Above 1 the selection is not one run, and
    /// every element is read on its own.
    #[arg(long, default_value_t = 1)]
    step: u64,
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
            streams: cli.streams,
            credit: cli.credit,
            ..ClientConfig::default()
        },
    )?;
    let handle = client.open_as(&cli.file, &cli.format)?;

    let elements = cli.bytes / cli.itemsize;
    let indices = [Index::Slice {
        start: Some(0),
        stop: Some((elements * cli.step) as i64),
        step: Some(cli.step as i64),
    }];
    let mut dst = vec![0u8; cli.bytes as usize];

    let mut runs = Vec::with_capacity(cli.reps);
    for _ in 0..cli.reps {
        let cpu = cpu_seconds();
        let started = Instant::now();
        let result = client.read_selection_into(handle, "array", &indices, &mut dst)?;
        let elapsed = started.elapsed();
        assert_eq!(result.streams, cli.streams, "not every connection was used");
        // A retry means some of the time went somewhere the number cannot
        // explain, so the run is not one to average in.
        assert_eq!(result.retries, 0, "a fetch had to be retried");
        runs.push(Run {
            bytes: result.bytes,
            elapsed,
            cpu: cpu_seconds() - cpu,
        });
    }

    // Proof that the right bytes arrived, not merely the right number of them.
    // Outside the timing, so a fast transfer that moved the wrong data is a
    // failure rather than a record.
    match cli.pattern {
        Pattern::CountingF32 => {
            let value_at =
                |i: usize| f32::from_le_bytes(dst[i..i + 4].try_into().expect("4 bytes"));
            assert_eq!(value_at(0), 0.0, "the transfer did not start at the start");
            assert_eq!(
                value_at(dst.len() - 4),
                ((elements - 1) * cli.step) as f32,
                "the transfer did not end at the end"
            );
        }
        Pattern::Bytes => {
            let item = cli.itemsize as usize;
            let source = |i: usize| (i / item) * cli.step as usize * item + i % item;
            let wrong = dst.iter().enumerate().find(|(i, &b)| b != source(*i) as u8);
            assert_eq!(
                wrong.map(|(i, _)| i),
                None,
                "a byte arrived from the wrong place"
            );
        }
        Pattern::None => {}
    }

    let label = cli.label.unwrap_or_else(|| cli.file.clone());
    report(
        &format!(
            "aex {label} chunk={} streams={} credit={} step={}",
            cli.chunk, cli.streams, cli.credit, cli.step
        ),
        &runs,
    );
    client.disconnect()?;
    Ok(())
}
