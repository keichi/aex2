//! End to end: what a client actually gets out of AEX2.
//!
//! The selection is a range of the leading axis, contiguous unless `--step`
//! says otherwise, read into a buffer allocated once, so that what is timed is
//! the transfer and not an allocator.

use std::time::Instant;

use aex_bench::{cpu_seconds, report, Run};
use aex_client::{Client, ClientConfig, Index, Selection};
use aex_core::{Codec, Encoding, QualitySpec};
use clap::{Parser, ValueEnum};

/// Which error-bounded codec to ask for.
///
/// Its own enum rather than `aex_core::Codec`, so the flag offers the codecs
/// that can carry a bound and not the lossless values as well.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CodecArg {
    Sz,
    Zfp,
}

impl From<CodecArg> for Codec {
    fn from(arg: CodecArg) -> Codec {
        match arg {
            CodecArg::Sz => Codec::Sz,
            CodecArg::Zfp => Codec::Zfp,
        }
    }
}

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
    /// Ask for error-bounded lossy transfer with this absolute bound. 0 asks
    /// for exact data.
    #[arg(long, default_value_t = 0.0)]
    abs_error: f64,
    /// Which codec carries the bound. Ignored when `--abs-error` is 0.
    #[arg(long, value_enum, default_value_t = CodecArg::Sz)]
    codec: CodecArg,
    #[arg(long, default_value_t = 5)]
    reps: usize,
    #[arg(long)]
    label: Option<String>,
    /// Touch the output buffer before timing, so a first run does not pay
    /// the page faults a reused buffer would not.
    #[arg(long)]
    prefault: bool,
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

    // A slice of the leading axis: the whole of every other axis comes with
    // it, which is what gives a codec an array slab rather than one long row.
    let elements = cli.bytes / cli.itemsize;
    let indices = [Index::Slice {
        start: Some(0),
        stop: Some((elements * cli.step) as i64),
        step: Some(cli.step as i64),
    }];
    let quality = if cli.abs_error > 0.0 {
        QualitySpec {
            encoding: Encoding::ErrorBound,
            abs_error_bound: Some(cli.abs_error),
            codec: Some(cli.codec.into()),
            ..QualitySpec::default()
        }
    } else {
        QualitySpec::exact()
    };
    let selection = Selection {
        quality: &quality,
        ..Selection::exact(handle, "array", &indices)
    };
    let mut dst = vec![0u8; cli.bytes as usize];
    if cli.prefault {
        dst.fill(1);
    }

    let mut wire_bytes = 0;
    let mut ratio = 0.0;
    let mut runs = Vec::with_capacity(cli.reps);
    for _ in 0..cli.reps {
        // The prepare is inside the timing, as it was when this called
        // read_selection_into, so the numbers stay comparable with the ones
        // already in docs/.
        let cpu = cpu_seconds();
        let started = Instant::now();
        let plan = client.prepare_selection(&selection)?;
        let result = client.fill_many(
            std::slice::from_ref(&plan),
            std::slice::from_ref(&selection),
            &mut [&mut dst],
        )?;
        let elapsed = started.elapsed();
        assert_eq!(
            plan.applied_quality.encoding, quality.encoding,
            "the server did not apply the quality asked for"
        );
        // Without this a server built without the codec asked for would send
        // the other one's bytes under this run's label, and a sweep comparing
        // the two would quietly be comparing one with itself.
        assert_eq!(
            plan.applied_quality.codec(),
            quality.codec(),
            "the server did not use the codec asked for"
        );
        assert_eq!(
            plan.total_bytes, cli.bytes,
            "the selection is not the size asked for"
        );
        assert_eq!(result.streams, cli.streams, "not every connection was used");
        // A retry means some of the time went somewhere the number cannot
        // explain, so the run is not one to average in.
        assert_eq!(result.retries, 0, "a fetch had to be retried");
        wire_bytes = result.wire_bytes;
        ratio = result.compression_ratio();
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
                |i: usize| f32::from_le_bytes(dst[i..i + 4].try_into().expect("4 bytes")) as f64;
            // Lossy transfers are checked against the bound they promised,
            // which is the whole claim being made about them.
            //
            // Through f32 first: past 2^24 the fixture itself cannot hold the
            // index exactly, and the value that arrived is the rounded one.
            let last = (((elements - 1) * cli.step) as f32) as f64;
            let off_by = |got: f64, want: f64| (got - want).abs();
            assert!(
                off_by(value_at(0), 0.0) <= cli.abs_error,
                "the transfer did not start at the start: {}",
                value_at(0)
            );
            assert!(
                off_by(value_at(dst.len() - 4), last) <= cli.abs_error,
                "the transfer did not end at the end: {} for {last}",
                value_at(dst.len() - 4)
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
    if cli.abs_error > 0.0 {
        // Printed only when something was compressed, so every existing sweep
        // produces exactly the output it did before.
        println!(
            "  codec={:?} abs_error={} compressed {ratio:.2}x ({} -> {wire_bytes} bytes on the wire)",
            cli.codec, cli.abs_error, cli.bytes
        );
    }
    client.disconnect()?;
    Ok(())
}
