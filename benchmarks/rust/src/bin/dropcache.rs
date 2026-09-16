//! Evict a file's pages from the page cache, without privileges.
//!
//! A cold-cache measurement needs the pages gone, and dropping the whole cache
//! wants root. `posix_fadvise(POSIX_FADV_DONTNEED)` drops one file's clean
//! pages and needs nothing, which is both less privileged and more precise.
//!
//! Linux only. macOS has no equivalent that works from outside the process
//! doing the reading, which is why the measurements there evict by reading
//! something larger than memory instead.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(about = "Drop a file's pages from the page cache")]
struct Cli {
    /// Files to evict.
    paths: Vec<PathBuf>,
}

#[cfg(target_os = "linux")]
fn drop_pages(path: &std::path::Path) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;

    let file = std::fs::File::open(path)?;
    // Written pages have to reach the disk before they can be dropped; a dirty
    // page stays cached however politely it is asked to leave.
    file.sync_all()?;

    // SAFETY: the descriptor is open for the call. 0 length means to the end.
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if rc != 0 {
        return Err(std::io::Error::from_raw_os_error(rc));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn drop_pages(_path: &std::path::Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "posix_fadvise(DONTNEED) is a Linux facility",
    ))
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    for path in &cli.paths {
        drop_pages(path)?;
    }
    Ok(())
}
