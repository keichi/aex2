//! How fast this machine moves memory.
//!
//! The ceiling everything else is measured against: a transfer copies its
//! payload several times over, so knowing what one copy costs says how much of
//! the machine a given throughput is using.
//!
//! Reading and writing are measured alongside copying, because a copy on its
//! own does not say whether it is at the limit. Reading is the cleanest measure
//! of what the memory can do — one stream, no ownership traffic — so a copy
//! that reaches half the read figure is at the limit rather than merely slow:
//! it moves two bytes for every one it copies.
//!
//! The buffers are allocated and touched before the timing starts, so what is
//! measured is the moving rather than the page faults.

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
        let mut pairs: Vec<(Vec<u64>, Vec<u64>)> = (0..threads)
            .map(|_| (vec![7u64; cli.bytes / 8], vec![1u64; cli.bytes / 8]))
            .collect();
        // Touch every page once, so the timing sees no faults.
        for (src, dst) in pairs.iter_mut() {
            dst.copy_from_slice(src);
        }
        let bytes = (cli.bytes * cli.reps * threads) as f64;
        let rate = |seconds: f64| bytes / seconds / MIB;

        let started = Instant::now();
        std::thread::scope(|scope| {
            for (src, _) in pairs.iter() {
                scope.spawn(move || {
                    for _ in 0..cli.reps {
                        // Summing is the cheapest way to touch every byte.
                        let sum = src.iter().fold(0u64, |acc, &v| acc.wrapping_add(v));
                        std::hint::black_box(sum);
                    }
                });
            }
        });
        let read = rate(started.elapsed().as_secs_f64());

        let started = Instant::now();
        std::thread::scope(|scope| {
            for (_, dst) in pairs.iter_mut() {
                scope.spawn(move || {
                    for rep in 0..cli.reps {
                        dst.fill(rep as u64);
                        std::hint::black_box(&dst);
                    }
                });
            }
        });
        let write = rate(started.elapsed().as_secs_f64());

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
        let copy = rate(started.elapsed().as_secs_f64());

        println!(
            "{threads:2} thread(s): read {read:8.0}  write {write:8.0}  copy {copy:8.0} MiB/s              (copy moves {:6.1} GiB/s)",
            copy * 2.0 / 1024.0,
        );
    }
}
