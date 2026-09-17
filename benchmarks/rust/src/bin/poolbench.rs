//! How should the server's reading be threaded across connections?
//!
//! The server's send path, cut out of the server: read pieces of a file, send
//! each with a 32-byte header, over N connections to a sink that discards them.
//! Three ways to arrange the reading:
//!
//! - `serial`: one thread per connection reads and writes in turn
//! - `pair`: each connection has a reader thread and a writer thread
//! - `pool`: P reader threads shared by every connection, one writer each
//!
//! `pair` doubles the threads with the connections; `pool` keeps the readers
//! at a count chosen for the machine or the storage.

use std::fs::File;
use std::io::{IoSlice, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use aex_bench::{cpu_seconds, report, Run};
use clap::{Parser, Subcommand, ValueEnum};

const HEADER_LEN: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    Serial,
    Pair,
    Pool,
}

#[derive(Parser)]
#[command(about = "Compare ways of threading the server's reads")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Accept connections and discard what arrives, until killed.
    Sink {
        #[arg(long, default_value = "0.0.0.0:50399")]
        listen: String,
    },
    /// Send `--bytes` of `path`, split across `--streams` connections.
    Send {
        path: PathBuf,
        #[arg(long)]
        to: String,
        #[arg(long)]
        bytes: u64,
        #[arg(long)]
        streams: usize,
        #[arg(long, value_enum)]
        mode: Mode,
        /// Shared reader threads, for `pool`.
        #[arg(long, default_value_t = 4)]
        readers: usize,
        /// Buffers per connection, for `pair` and `pool`.
        #[arg(long, default_value_t = 3)]
        buffers: usize,
        #[arg(long, default_value_t = 524288)]
        piece: usize,
        /// Where the data starts; 128 is where a `.npy` header usually ends.
        #[arg(long, default_value_t = 128)]
        offset: u64,
        #[arg(long, default_value_t = 1)]
        reps: usize,
        /// Evict the file from the page cache before each run (Linux).
        #[arg(long)]
        cold: bool,
    },
}

/// One connection's share: `[start, end)` of the file.
#[derive(Clone, Copy)]
struct Range {
    start: u64,
    end: u64,
}

fn main() -> std::io::Result<()> {
    match Cli::parse().command {
        Command::Sink { listen } => sink(&listen),
        Command::Send {
            path,
            to,
            bytes,
            streams,
            mode,
            readers,
            buffers,
            piece,
            offset,
            reps,
            cold,
        } => {
            let file = Arc::new(File::open(&path)?);
            let share = bytes / streams as u64;
            let ranges: Vec<Range> = (0..streams as u64)
                .map(|i| Range {
                    start: offset + i * share,
                    end: offset + (i + 1) * share,
                })
                .collect();

            let mut runs = Vec::new();
            for _ in 0..reps {
                if cold {
                    evict(&file)?;
                }
                let conns: Vec<TcpStream> = (0..streams)
                    .map(|_| {
                        let s = TcpStream::connect(&to)?;
                        s.set_nodelay(true)?;
                        Ok(s)
                    })
                    .collect::<std::io::Result<_>>()?;

                let cpu = cpu_seconds();
                let started = Instant::now();
                match mode {
                    Mode::Serial => serial(&file, &ranges, conns, piece),
                    Mode::Pair => pair(&file, &ranges, conns, piece, buffers),
                    Mode::Pool => pool(&file, &ranges, conns, piece, buffers, readers),
                }?;
                runs.push(Run {
                    bytes: share * streams as u64,
                    elapsed: started.elapsed(),
                    cpu: cpu_seconds() - cpu,
                });
            }
            let label = match mode {
                Mode::Serial => "serial".to_string(),
                Mode::Pair => format!("pair buffers={buffers}"),
                Mode::Pool => format!("pool readers={readers} buffers={buffers}"),
            };
            report(
                &format!(
                    "{label} streams={streams}{}",
                    if cold { " cold" } else { "" }
                ),
                &runs,
            );
            Ok(())
        }
    }
}

fn sink(listen: &str) -> std::io::Result<()> {
    for conn in TcpListener::bind(listen)?.incoming() {
        let mut conn = conn?;
        std::thread::spawn(move || {
            let mut buf = vec![0u8; 1 << 20];
            while matches!(conn.read(&mut buf), Ok(n) if n > 0) {}
        });
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn evict(file: &File) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if rc != 0 {
        return Err(std::io::Error::from_raw_os_error(rc));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn evict(_file: &File) -> std::io::Result<()> {
    Err(std::io::ErrorKind::Unsupported.into())
}

/// Send one piece with its header, then wait for nothing: the sink only reads.
fn send_piece(conn: &mut TcpStream, piece: &[u8]) -> std::io::Result<()> {
    let header = [0u8; HEADER_LEN];
    let mut slices = [IoSlice::new(&header), IoSlice::new(piece)];
    let mut rest = &mut slices[..];
    while !rest.is_empty() {
        let n = conn.write_vectored(rest)?;
        IoSlice::advance_slices(&mut rest, n);
    }
    Ok(())
}

/// Close the sending side and wait for the sink to have read it all.
fn finish(mut conn: TcpStream) -> std::io::Result<()> {
    conn.shutdown(Shutdown::Write)?;
    let mut rest = [0u8; 1];
    conn.read(&mut rest).map(|_| ())
}

fn read_piece(file: &File, at: u64, buf: &mut [u8]) -> std::io::Result<()> {
    file.read_exact_at(buf, at)
}

/// Spawn one thread per connection and wait for them all.
fn per_conn(
    ranges: &[Range],
    conns: Vec<TcpStream>,
    work: impl Fn(usize, Range, TcpStream) -> std::io::Result<()> + Sync,
) -> std::io::Result<()> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = conns
            .into_iter()
            .enumerate()
            .map(|(i, conn)| {
                let work = &work;
                let range = ranges[i];
                scope.spawn(move || work(i, range, conn))
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().expect("sender panicked"))
    })
}

fn serial(
    file: &File,
    ranges: &[Range],
    conns: Vec<TcpStream>,
    piece: usize,
) -> std::io::Result<()> {
    per_conn(ranges, conns, |_, range, mut conn| {
        let mut buf = vec![0u8; piece];
        let mut at = range.start;
        while at < range.end {
            let len = piece.min((range.end - at) as usize);
            read_piece(file, at, &mut buf[..len])?;
            send_piece(&mut conn, &buf[..len])?;
            at += len as u64;
        }
        finish(conn)
    })
}

fn pair(
    file: &File,
    ranges: &[Range],
    conns: Vec<TcpStream>,
    piece: usize,
    buffers: usize,
) -> std::io::Result<()> {
    per_conn(ranges, conns, |_, range, mut conn| {
        let (free_tx, free_rx) = channel::<Vec<u8>>();
        let (full_tx, full_rx) = sync_channel::<(Vec<u8>, usize)>(buffers);
        for _ in 0..buffers {
            free_tx.send(vec![0u8; piece]).unwrap();
        }
        std::thread::scope(|scope| {
            let reader = scope.spawn(move || -> std::io::Result<()> {
                let mut at = range.start;
                while at < range.end {
                    let mut buf = free_rx.recv().expect("writer alive");
                    let len = piece.min((range.end - at) as usize);
                    read_piece(file, at, &mut buf[..len])?;
                    full_tx.send((buf, len)).expect("writer alive");
                    at += len as u64;
                }
                Ok(())
            });
            for (buf, len) in full_rx {
                send_piece(&mut conn, &buf[..len])?;
                let _ = free_tx.send(buf);
            }
            reader.join().expect("reader panicked")
        })?;
        finish(conn)
    })
}

/// A read for the shared pool: where, and whose buffer to fill.
struct Job {
    at: u64,
    len: usize,
    buf: Vec<u8>,
    done: Sender<std::io::Result<(Vec<u8>, usize)>>,
}

fn pool(
    file: &File,
    ranges: &[Range],
    conns: Vec<TcpStream>,
    piece: usize,
    buffers: usize,
    readers: usize,
) -> std::io::Result<()> {
    let (job_tx, job_rx) = channel::<Job>();
    let job_rx: Mutex<Receiver<Job>> = Mutex::new(job_rx);
    // Taken inside the scope so that dropping it there lets the readers stop.
    let mut job_tx = Some(job_tx);

    std::thread::scope(|scope| {
        for _ in 0..readers {
            scope.spawn(|| loop {
                let job = match job_rx.lock().unwrap().recv() {
                    Ok(job) => job,
                    Err(_) => return,
                };
                let Job {
                    at,
                    len,
                    mut buf,
                    done,
                } = job;
                let result = read_piece(file, at, &mut buf[..len]).map(|()| (buf, len));
                let _ = done.send(result);
            });
        }

        let owned = job_tx.take().expect("taken once");
        let job_tx = &owned;
        let result = per_conn(ranges, conns, |_, range, mut conn| {
            let (done_tx, done_rx) = channel();
            let mut next = range.start;
            let submit = |buf: Vec<u8>, next: &mut u64| {
                let len = piece.min((range.end - *next) as usize);
                job_tx
                    .send(Job {
                        at: *next,
                        len,
                        buf,
                        done: done_tx.clone(),
                    })
                    .expect("readers alive");
                *next += len as u64;
            };
            // Each buffer is out at most once, so the readers can run ahead of
            // this connection by `buffers` pieces and no further.
            let mut out = 0;
            for _ in 0..buffers {
                if next < range.end {
                    submit(vec![0u8; piece], &mut next);
                    out += 1;
                }
            }
            // Pieces may come back out of order; the sink does not care, and a
            // real frame names its own offset.
            while out > 0 {
                let (buf, len) = done_rx.recv().expect("reader alive")?;
                send_piece(&mut conn, &buf[..len])?;
                out -= 1;
                if next < range.end {
                    submit(buf, &mut next);
                    out += 1;
                }
            }
            finish(conn)
        });
        drop(owned);
        result
    })
}
