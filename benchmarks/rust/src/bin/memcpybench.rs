//! How fast this machine copies memory.
//!
//! The ceiling everything else is measured against: a transfer copies its
//! payload several times over, so knowing what one copy costs says how much of
//! the machine a given throughput is using.
//!
//! The buffers are allocated and touched before the timing starts, so what is
//! measured is the copying rather than the page faults.

use std::time::Instant;

use clap::Parser;

#[derive(Parser)]
#[command(about = "Measure memory copy bandwidth, one thread and several")]
struct Cli {
    /// Bytes per thread.
    #[arg(long, default_value_t = 1 << 30)]
    bytes: usize,
    /// How many times each thread copies its buffer.
    #[arg(long, default_value_t = 4)]
    reps: usize,
    /// Thread counts to sweep.
    #[arg(long, value_delimiter = ',', default_values_t = [1usize, 2, 4, 8])]
    threads: Vec<usize>,
}

fn main() {
    let cli = Cli::parse();
    const MIB: f64 = 1024.0 * 1024.0;

    for &threads in &cli.threads {
        let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..threads)
            .map(|_| (vec![7u8; cli.bytes], vec![1u8; cli.bytes]))
            .collect();
        // Touch every page once, so the timing sees no faults.
        for (src, dst) in pairs.iter_mut() {
            dst.copy_from_slice(src);
        }

        let started = Instant::now();
        std::thread::scope(|scope| {
            for (src, dst) in pairs.iter_mut() {
                scope.spawn(move || {
                    for _ in 0..cli.reps {
                        dst.copy_from_slice(src);
                        std::hint::black_box(&dst);
                    }
                });
            }
        });
        let seconds = started.elapsed().as_secs_f64();
        let moved = (cli.bytes * cli.reps * threads) as f64;

        println!(
            "memcpy {threads:2} thread(s): {:8.0} MiB/s copied ({:6.1} GiB/s of memory traffic)",
            moved / seconds / MIB,
            // Each copy reads one byte and writes another.
            moved * 2.0 / seconds / (MIB * 1024.0),
        );
    }
}
