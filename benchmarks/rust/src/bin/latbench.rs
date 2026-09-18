//! Latency: what a small interactive read costs.
//!
//! A selection that fits the server's inline limit comes back with its plan, in
//! one round trip; anything larger costs two. This is where that shows — or, on
//! a loopback link where a round trip is tens of microseconds, where it does
//! not, and the reading is a floor rather than a verdict.
//!
//! It also times several selections taken one by one against the same ones
//! gathered: what `gather` is for is turning N round trips into one.

use std::time::Instant;

use aex_bench::percentile;
use aex_client::{Client, ClientConfig, FileHandle, Index, Selection};
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
    /// Selections per gather, swept one batch size at a time.
    #[arg(long, value_delimiter = ',', default_values_t = [2usize, 8, 32, 64])]
    gather: Vec<usize>,
    /// Bytes per selection of a gather.
    #[arg(long, default_value_t = 4096)]
    gather_bytes: u64,
    /// Times each gather batch is timed. Fewer than `reps`: a batch is N reads.
    #[arg(long, default_value_t = 50)]
    gather_reps: usize,
}

/// Time `count` selections taken one at a time against the same ones gathered.
fn gather_batch(
    client: &Client,
    handle: FileHandle,
    count: usize,
    bytes: u64,
    reps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Spread over the array, as an analysis reading scattered rows would be.
    let keys: Vec<Vec<Index>> = (0..count as i64)
        .map(|i| {
            let start = i * (bytes / 4) as i64 * 7;
            vec![Index::range(start, start + (bytes / 4) as i64)]
        })
        .collect();
    let selections: Vec<Selection<'_>> = keys
        .iter()
        .map(|indices| Selection::exact(handle, "array", indices))
        .collect();
    let mut buffers = vec![vec![0u8; bytes as usize]; count];

    let (mut single, mut batched) = (Vec::new(), Vec::new());
    for rep in 0..=reps {
        let started = Instant::now();
        for (selection, out) in selections.iter().zip(&mut buffers) {
            client.read_selection_into(handle, selection.name, selection.indices, out)?;
        }
        let one_by_one = started.elapsed().as_secs_f64() * 1e3;

        let started = Instant::now();
        let plans = client
            .prepare_many(&selections)?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let mut outs: Vec<&mut [u8]> = buffers.iter_mut().map(|b| b.as_mut_slice()).collect();
        client.fill_many(&plans, &selections, &mut outs)?;
        let together = started.elapsed().as_secs_f64() * 1e3;

        // The first pass pays for whatever the connection was not ready for.
        if rep > 0 {
            single.push(one_by_one);
            batched.push(together);
        }
    }
    single.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    batched.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let (one, all) = (percentile(&single, 0.5), percentile(&batched, 0.5));
    println!(
        "{:<26} p50 {:7.2} ms one by one, {:7.2} ms gathered, {:5.1}x",
        format!("gather {count} x {bytes} B"),
        one,
        all,
        one / all
    );
    Ok(())
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

    for count in &cli.gather {
        gather_batch(&client, handle, *count, cli.gather_bytes, cli.gather_reps)?;
    }

    client.disconnect()?;
    Ok(())
}
