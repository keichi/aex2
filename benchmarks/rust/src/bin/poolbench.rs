//! How should the server's reading be threaded across connections?
//!
//! The server's send path, cut out of the server: read pieces of a file, send
//! each with a 32-byte header, over N connections to a sink that discards them.
//! Three ways to arrange the reading:
//!
//! - `serial`: one thread per connection reads and writes in turn
//! - `pair`: each connection has a reader thread and a writer thread
//! - `pool`: P reader threads shared by every connection, one writer each
//! - `fadvise`: `serial`, with the next `--depth` pieces hinted to the kernel
//! - `uring`: one thread per connection, reads on its own io_uring, blocking send
//! - `uring-copy` / `uring-zc`: reads and sends both on the ring
//!
//! `pair` doubles the threads with the connections; `pool` keeps the readers
//! at a count chosen for the machine or the storage. The `uring` modes keep one
//! thread per connection and overlap the reads with the send from that same
//! thread, so nothing is handed between threads.

use std::fs::File;
use std::io::{IoSlice, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
    Fadvise,
    Uring,
    UringCopy,
    UringZc,
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
        /// Pieces read ahead per connection, for the `uring` modes.
        #[arg(long, default_value_t = 4)]
        depth: usize,
        /// Check every piece against a pread of the same range. Costs time.
        #[arg(long)]
        verify: bool,
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
            depth,
            verify,
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

            let stats = uring::Stats::default();
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
                    Mode::Fadvise => fadvise(&file, &ranges, conns, piece, depth),
                    _ => uring::run(&file, &ranges, conns, piece, depth, mode, verify, &stats),
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
                Mode::Fadvise => format!("fadvise depth={depth}"),
                Mode::Uring => format!("uring depth={depth}"),
                Mode::UringCopy => format!("uring-copy depth={depth}"),
                Mode::UringZc => format!("uring-zc depth={depth}"),
            };
            report(
                &format!(
                    "{label} streams={streams}{}",
                    if cold { " cold" } else { "" }
                ),
                &runs,
            );
            stats.report();
            Ok(())
        }
    }
}

/// Discard what arrives, and report what receiving it cost once the last
/// connection of a run closes: the sender is only worth speeding up while the
/// receiver still has headroom.
fn sink(listen: &str) -> std::io::Result<()> {
    let active = Arc::new(AtomicUsize::new(0));
    let bytes = Arc::new(AtomicU64::new(0));
    let start: Arc<Mutex<(Instant, f64)>> = Arc::new(Mutex::new((Instant::now(), 0.0)));

    for conn in TcpListener::bind(listen)?.incoming() {
        let mut conn = conn?;
        let (active, bytes, start) = (active.clone(), bytes.clone(), start.clone());
        std::thread::spawn(move || {
            if active.fetch_add(1, Ordering::SeqCst) == 0 {
                bytes.store(0, Ordering::SeqCst);
                *start.lock().unwrap() = (Instant::now(), cpu_seconds());
            }
            let mut buf = vec![0u8; 1 << 20];
            while let Ok(n) = conn.read(&mut buf) {
                if n == 0 {
                    break;
                }
                bytes.fetch_add(n as u64, Ordering::Relaxed);
            }
            // Let the sender's `finish` return: it waits for a read to end.
            let _ = conn.shutdown(Shutdown::Both);
            if active.fetch_sub(1, Ordering::SeqCst) == 1 {
                let (at, cpu) = *start.lock().unwrap();
                report(
                    "sink",
                    &[Run {
                        bytes: bytes.load(Ordering::SeqCst),
                        elapsed: at.elapsed(),
                        cpu: cpu_seconds() - cpu,
                    }],
                );
            }
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

/// Ask the kernel to start reading `[at, at + len)` now.
#[cfg(target_os = "linux")]
fn will_need(file: &File, at: u64, len: usize) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let rc = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            at as i64,
            len as i64,
            libc::POSIX_FADV_WILLNEED,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::from_raw_os_error(rc));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn will_need(_file: &File, _at: u64, _len: usize) -> std::io::Result<()> {
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

/// `serial`, with the read of the next `depth` pieces already under way.
///
/// The cheapest way to have more than one read in flight per connection: the
/// kernel starts them, and nothing is handed between threads.
fn fadvise(
    file: &File,
    ranges: &[Range],
    conns: Vec<TcpStream>,
    piece: usize,
    depth: usize,
) -> std::io::Result<()> {
    per_conn(ranges, conns, |_, range, mut conn| {
        let mut buf = vec![0u8; piece];
        let window = (depth * piece) as u64;
        let advise = |at: u64| {
            if at < range.end {
                will_need(file, at, piece.min((range.end - at) as usize))
            } else {
                Ok(())
            }
        };
        for i in 0..depth as u64 {
            advise(range.start + i * piece as u64)?;
        }
        let mut at = range.start;
        while at < range.end {
            let len = piece.min((range.end - at) as usize);
            // Keep the hint `depth` pieces ahead of where the reading is.
            advise(at + window)?;
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

/// Reads (and optionally sends) on an io_uring owned by the connection's own
/// thread: the locality of `serial` with the overlap of `pair`, and no handoff.
#[cfg(target_os = "linux")]
mod uring {
    use super::{finish, per_conn, send_piece, Mode, Range, HEADER_LEN};
    use std::collections::VecDeque;
    use std::fs::File;
    use std::io;
    use std::net::TcpStream;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    use io_uring::{cqueue, opcode, squeue, types, IoUring};

    /// Ask the kernel to say, in the notification, whether a zero-copy send had
    /// to copy after all. Not in the crate's exported bindings.
    const SEND_ZC_REPORT_USAGE: u16 = 8;
    const NOTIF_ZC_COPIED: u32 = 1 << 31;

    /// Completion kind, in the top bits of `user_data`; the rest is the slot.
    const SEND_TAG: u64 = 1 << 32;

    #[derive(Clone, Copy, PartialEq)]
    enum How {
        /// Reads on the ring, send with a blocking `writev`.
        Blocking,
        Copy,
        Zc,
    }

    #[derive(Default)]
    pub struct Stats {
        sends: AtomicU64,
        copied: AtomicU64,
        enobufs: AtomicU64,
    }

    impl Stats {
        pub fn report(&self) {
            let sends = self.sends.load(Ordering::Relaxed);
            if sends == 0 {
                return;
            }
            let copied = self.copied.load(Ordering::Relaxed);
            println!(
                "  sends {sends}  copied {copied} ({:.0} %)  enobufs {}",
                100.0 * copied as f64 / sends as f64,
                self.enobufs.load(Ordering::Relaxed),
            );
        }
    }

    /// One piece in flight: its buffer holds the header and the payload next to
    /// each other so a ring send is one SQE.
    struct Slot {
        buf: Vec<u8>,
        at: u64,
        len: usize,
        got: usize,
        sent: usize,
        ready: bool,
        done: bool,
        /// Zero-copy notifications still owed for this buffer.
        notifs: u32,
    }

    impl Slot {
        fn frame(&self) -> usize {
            HEADER_LEN + self.len
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        file: &File,
        ranges: &[Range],
        conns: Vec<TcpStream>,
        piece: usize,
        depth: usize,
        mode: Mode,
        verify: bool,
        stats: &Stats,
    ) -> io::Result<()> {
        let how = match mode {
            Mode::Uring => How::Blocking,
            Mode::UringCopy => How::Copy,
            Mode::UringZc => How::Zc,
            _ => unreachable!("only the uring modes come here"),
        };
        per_conn(ranges, conns, |_, range, conn| {
            connection(file, range, conn, piece, depth, how, verify, stats)
        })
    }

    /// Push one SQE, making room by submitting if the queue is full.
    ///
    /// # Safety
    ///
    /// The buffer the entry points at must stay put until its completion (and,
    /// for a zero-copy send, its notification) has been reaped.
    unsafe fn push(ring: &mut IoUring, entry: &squeue::Entry) -> io::Result<()> {
        while ring.submission().push(entry).is_err() {
            ring.submit()?;
        }
        Ok(())
    }

    fn read_sqe(fd: types::Fd, slot: &mut Slot, i: usize) -> squeue::Entry {
        let at = slot.got;
        opcode::Read::new(
            fd,
            slot.buf[HEADER_LEN + at..].as_mut_ptr(),
            (slot.len - at) as u32,
        )
        .offset(slot.at + at as u64)
        .build()
        .user_data(i as u64)
    }

    fn send_sqe(fd: types::Fd, slot: &Slot, i: usize, how: How) -> squeue::Entry {
        let (ptr, len) = (
            slot.buf[slot.sent..].as_ptr(),
            (slot.frame() - slot.sent) as u32,
        );
        // MSG_WAITALL: let the kernel finish the piece rather than come back
        // with a partial send and leave the socket idle until it is resubmitted.
        let entry = if how == How::Zc {
            opcode::SendZc::new(fd, ptr, len)
                .flags(libc::MSG_WAITALL)
                .zc_flags(SEND_ZC_REPORT_USAGE)
                .build()
        } else {
            opcode::Send::new(fd, ptr, len)
                .flags(libc::MSG_WAITALL)
                .build()
        };
        entry.user_data(SEND_TAG | i as u64)
    }

    fn check(file: &File, slot: &Slot) -> io::Result<()> {
        let mut want = vec![0u8; slot.len];
        file.read_exact_at(&mut want, slot.at)?;
        if want != slot.buf[HEADER_LEN..slot.frame()] {
            return Err(io::Error::other("piece read through the ring differs"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn connection(
        file: &File,
        range: Range,
        mut conn: TcpStream,
        piece: usize,
        depth: usize,
        how: How,
        verify: bool,
        stats: &Stats,
    ) -> io::Result<()> {
        // Enough room for every read plus the send and a retry.
        let mut ring = IoUring::new(((depth + 4) as u32).next_power_of_two())?;
        let file_fd = types::Fd(file.as_raw_fd());
        let sock_fd = types::Fd(conn.as_raw_fd());
        // A blocking socket makes io_uring punt the send to a kernel worker
        // thread, which is the handoff this is trying to avoid.
        if how != How::Blocking {
            conn.set_nonblocking(true)?;
        }

        let mut slots: Vec<Slot> = (0..depth)
            .map(|_| Slot {
                buf: vec![0u8; HEADER_LEN + piece],
                at: 0,
                len: 0,
                got: 0,
                sent: 0,
                ready: false,
                done: false,
                notifs: 0,
            })
            .collect();
        let mut free: Vec<usize> = (0..depth).rev().collect();
        // Submission order, which is also the order the pieces must be sent in.
        let mut order: VecDeque<usize> = VecDeque::new();
        let mut next = range.start;
        let mut sending = false;
        let mut notifs = 0u32;

        loop {
            let mut progress = false;

            while next < range.end {
                let Some(i) = free.pop() else { break };
                let len = piece.min((range.end - next) as usize);
                let slot = &mut slots[i];
                (slot.at, slot.len, slot.got, slot.sent) = (next, len, 0, 0);
                (slot.ready, slot.done) = (false, false);
                let sqe = read_sqe(file_fd, slot, i);
                // SAFETY: the slot is not touched again until its completion.
                unsafe { push(&mut ring, &sqe)? };
                order.push_back(i);
                next += len as u64;
                progress = true;
            }

            if !sending {
                if let Some(&head) = order.front() {
                    if slots[head].ready {
                        if verify {
                            check(file, &slots[head])?;
                        }
                        if how == How::Blocking {
                            let slot = &slots[head];
                            send_piece(&mut conn, &slot.buf[HEADER_LEN..slot.frame()])?;
                            order.pop_front();
                            free.push(head);
                        } else {
                            let sqe = send_sqe(sock_fd, &slots[head], head, how);
                            // SAFETY: the buffer is held until the send, and any
                            // notification, has completed.
                            unsafe { push(&mut ring, &sqe)? };
                            sending = true;
                        }
                        progress = true;
                    }
                }
            }

            if order.is_empty() && next >= range.end && notifs == 0 {
                break;
            }

            ring.submit_and_wait(usize::from(!progress))?;
            for cqe in ring.completion().collect::<Vec<cqueue::Entry>>() {
                let i = (cqe.user_data() & 0xffff_ffff) as usize;
                if cqe.user_data() & SEND_TAG == 0 {
                    let n = cqe.result();
                    if n < 0 {
                        return Err(io::Error::from_raw_os_error(-n));
                    }
                    if n == 0 {
                        return Err(io::ErrorKind::UnexpectedEof.into());
                    }
                    let slot = &mut slots[i];
                    slot.got += n as usize;
                    if slot.got == slot.len {
                        slot.ready = true;
                    } else {
                        let sqe = read_sqe(file_fd, slot, i);
                        // SAFETY: as above; the slot stays put.
                        unsafe { push(&mut ring, &sqe)? };
                    }
                    continue;
                }

                if cqueue::notif(cqe.flags()) {
                    if cqe.result() as u32 & NOTIF_ZC_COPIED != 0 {
                        stats.copied.fetch_add(1, Ordering::Relaxed);
                    }
                    notifs -= 1;
                    slots[i].notifs -= 1;
                    if slots[i].done && slots[i].notifs == 0 {
                        free.push(i);
                    }
                    continue;
                }

                let n = cqe.result();
                if n == -libc::ENOBUFS {
                    // Out of pinned memory for zero copy; this piece goes the
                    // ordinary way. No notification follows a failed send.
                    stats.enobufs.fetch_add(1, Ordering::Relaxed);
                    let sqe = send_sqe(sock_fd, &slots[i], i, How::Copy);
                    // SAFETY: as above; the slot stays put.
                    unsafe { push(&mut ring, &sqe)? };
                    continue;
                }
                if n < 0 {
                    return Err(io::Error::from_raw_os_error(-n));
                }
                if cqueue::more(cqe.flags()) {
                    slots[i].notifs += 1;
                    notifs += 1;
                }
                let slot = &mut slots[i];
                slot.sent += n as usize;
                if slot.sent < slot.frame() {
                    let sqe = send_sqe(sock_fd, slot, i, how);
                    // SAFETY: as above; the slot stays put.
                    unsafe { push(&mut ring, &sqe)? };
                    continue;
                }
                stats.sends.fetch_add(1, Ordering::Relaxed);
                sending = false;
                debug_assert_eq!(order.front(), Some(&i), "sends complete in order");
                order.pop_front();
                slot.done = true;
                if slot.notifs == 0 {
                    free.push(i);
                }
            }
        }

        conn.set_nonblocking(false)?;
        finish(conn)
    }
}

#[cfg(not(target_os = "linux"))]
mod uring {
    use super::{Mode, Range};
    use std::fs::File;
    use std::net::TcpStream;

    #[derive(Default)]
    pub struct Stats(());

    impl Stats {
        pub fn report(&self) {}
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        _file: &File,
        _ranges: &[Range],
        _conns: Vec<TcpStream>,
        _piece: usize,
        _depth: usize,
        _mode: Mode,
        _verify: bool,
        _stats: &Stats,
    ) -> std::io::Result<()> {
        Err(std::io::ErrorKind::Unsupported.into())
    }
}
