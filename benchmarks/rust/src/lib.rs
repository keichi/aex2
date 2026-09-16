//! Timing and reporting, shared by the benchmark binaries.
//!
//! Throughput is reported as a median rather than a mean: one run delayed by
//! something else on the machine should not move the number that gets written
//! down. CPU per gibibyte is reported alongside it, because on a loopback link
//! throughput says more about the memory bus than about the design, and how
//! much of the machine it took to reach that throughput is the honest measure.

use std::time::Duration;

/// User plus system CPU seconds this process has used so far.
pub fn cpu_seconds() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // Cannot fail for RUSAGE_SELF; the zeroed struct is what it fills in.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

/// One timed transfer.
pub struct Run {
    pub bytes: u64,
    pub elapsed: Duration,
    /// CPU seconds this process spent on it.
    pub cpu: f64,
}

const MIB: f64 = 1024.0 * 1024.0;

impl Run {
    pub fn mib_per_sec(&self) -> f64 {
        self.bytes as f64 / self.elapsed.as_secs_f64() / MIB
    }

    pub fn gbit_per_sec(&self) -> f64 {
        self.bytes as f64 * 8.0 / self.elapsed.as_secs_f64() / 1e9
    }

    /// Seconds of CPU per gibibyte moved.
    pub fn cpu_per_gib(&self) -> f64 {
        self.cpu / (self.bytes as f64 / (MIB * 1024.0))
    }
}

/// Print one line for a set of runs: the median, the best, and the CPU it cost.
pub fn report(label: &str, runs: &[Run]) {
    if runs.is_empty() {
        println!("{label:<44} no runs");
        return;
    }
    let mut sorted: Vec<&Run> = runs.iter().collect();
    sorted.sort_by(|a, b| {
        a.mib_per_sec()
            .partial_cmp(&b.mib_per_sec())
            .expect("no NaN")
    });

    let median = sorted[sorted.len() / 2];
    let best = sorted.last().expect("not empty");
    let cpu: f64 = runs.iter().map(Run::cpu_per_gib).sum::<f64>() / runs.len() as f64;
    println!(
        "{label:<44} median {:8.0} MiB/s ({:5.1} Gbit/s)  best {:8.0}  cpu {cpu:5.2} s/GiB  n={}",
        median.mib_per_sec(),
        median.gbit_per_sec(),
        best.mib_per_sec(),
        runs.len(),
    );
}

/// Percentile of an already sorted sample.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index]
}
