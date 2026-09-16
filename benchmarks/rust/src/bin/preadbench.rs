//! What the storage alone can do: `pread` a file into one reused buffer.
//!
//! The same shape as the server's read path, with nothing else in it, so it
//! bounds what any amount of protocol work could deliver. Run it against a file
//! whose pages are resident and it measures the memory bus; run it against one
//! that has been evicted and it measures the disk.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

use aex_bench::{cpu_seconds, report, Run};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Time pread of a range of a file into a reused buffer")]
struct Cli {
    path: PathBuf,
    /// Bytes to read per run.
    #[arg(long)]
    bytes: u64,
    /// Bytes per pread.
    #[arg(long)]
    chunk: usize,
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Where the data starts. 128 is where a `.npy` header usually ends.
    #[arg(long, default_value_t = 128)]
    offset: u64,
    #[arg(long)]
    label: Option<String>,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let file = File::open(&cli.path)?;
    let mut buf = vec![0u8; cli.chunk];

    let mut runs = Vec::with_capacity(cli.reps);
    for _ in 0..cli.reps {
        let cpu = cpu_seconds();
        let started = Instant::now();
        let mut read = 0u64;
        while read < cli.bytes {
            let len = cli.chunk.min((cli.bytes - read) as usize);
            file.read_exact_at(&mut buf[..len], cli.offset + read)?;
            read += len as u64;
        }
        runs.push(Run {
            bytes: cli.bytes,
            elapsed: started.elapsed(),
            cpu: cpu_seconds() - cpu,
        });
    }

    let label = cli.label.unwrap_or_else(|| cli.path.display().to_string());
    report(&format!("pread {label} chunk={}", cli.chunk), &runs);
    Ok(())
}
