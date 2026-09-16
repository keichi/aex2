//! Latency: what a small interactive read costs.
//!
//! A selection that fits the server's inline limit comes back with its plan, in
//! one round trip; anything larger costs two. This is where that shows — or, on
//! a loopback link where a round trip is tens of microseconds, where it does
//! not, and the reading is a floor rather than a verdict.

use std::time::Instant;

use aex_bench::percentile;
use aex_client::{Client, ClientConfig, Index};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Measure the latency of a small read, inline and over the data plane")]
struct Cli {
    url: String,
    file: String,
    #[arg(long, default_value_t = 2000)]
    reps: usize,
    /// Selection sizes to sweep, in bytes.
    #[arg(long, value_delimiter = ',', default_values_t = [4u64, 1024, 16384, 65536, 65540, 262144, 1048576])]
    sizes: Vec<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let client = Client::connect(&cli.url, ClientConfig::default())?;
    let handle = client.open(&cli.file)?;

    // A metadata RPC, as the floor for anything that crosses the control plane.
    let mut rpc: Vec<f64> = Vec::with_capacity(cli.reps);
    for _ in 0..cli.reps {
        let started = Instant::now();
        client.get_item(handle, "array")?;
        rpc.push(started.elapsed().as_secs_f64() * 1e6);
    }
    rpc.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    println!(
        "{:<26} p50 {:7.1} us  p99 {:7.1} us",
        "GetItem (metadata only)",
        percentile(&rpc, 0.5),
        percentile(&rpc, 0.99)
    );

    for bytes in cli.sizes {
        let indices = [Index::range(0, (bytes / 4) as i64)];
        let mut dst = vec![0u8; bytes as usize];

        let mut inline = false;
        let mut samples: Vec<f64> = Vec::with_capacity(cli.reps);
        for rep in 0..cli.reps {
            let started = Instant::now();
            let result = client.read_selection_into(handle, "array", &indices, &mut dst)?;
            // The first one pays for whatever the connection was not ready for.
            if rep == 0 {
                inline = result.inline;
                continue;
            }
            samples.push(started.elapsed().as_secs_f64() * 1e6);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        println!(
            "{:<26} p50 {:7.1} us  p99 {:7.1} us   {}",
            format!("read {bytes} B"),
            percentile(&samples, 0.5),
            percentile(&samples, 0.99),
            if inline {
                "inline, 1 round trip"
            } else {
                "data plane, 2 round trips"
            }
        );
    }

    client.disconnect()?;
    Ok(())
}
