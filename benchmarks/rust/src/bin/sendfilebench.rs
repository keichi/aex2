//! Does `sendfile` beat reading into a buffer and writing that?
//!
//! The server reads a piece of the file into a buffer of its own and hands the
//! buffer and a frame header to `writev`. `sendfile` would send straight from
//! the page cache to the socket, with the header carried alongside, and never
//! bring the payload into the process at all.
//!
//! Both modes here send the same bytes over a loopback socket in the same
//! pieces, each preceded by a 32-byte header, and differ only in how the
//! payload gets there. Over loopback there is no network card to hand the
//! pages to, so this measures the saved copy and nothing else.
//!
//! The call differs by system. macOS takes the header in the same call and
//! counts its length toward the length to send; Linux has no room for a header
//! at all, so it goes first under `MSG_MORE` to keep it in the same segment.
//! That is two syscalls a piece either way — `writev` needs only one — which is
//! part of what is being compared.

use std::fs::File;
use std::io::{IoSlice, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

use aex_bench::{cpu_seconds, report, Run};
use clap::{Parser, ValueEnum};

/// The frame header the data plane puts in front of every piece.
const HEADER_LEN: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// pread into a buffer, then writev the header and the buffer, one after
    /// the other on one thread.
    Pread,
    /// The same, with the reading on a thread of its own so that it overlaps
    /// the writing. What the server does.
    PreadThreaded,
    /// sendfile straight from the page cache, header carried alongside.
    Sendfile,
}

#[derive(Parser)]
#[command(about = "Compare sendfile with pread + writev over a loopback socket")]
struct Cli {
    path: PathBuf,
    #[arg(long)]
    bytes: u64,
    /// Bytes of payload per piece, as the server's read buffers are sized.
    #[arg(long, default_value_t = 524288)]
    piece: usize,
    #[arg(long, default_value_t = 5)]
    reps: usize,
    /// Where the data starts; 128 is where a `.npy` header usually ends.
    #[arg(long, default_value_t = 128)]
    offset: u64,
    #[arg(long, value_enum)]
    mode: Mode,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let file = File::open(&cli.path)?;

    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    // Headers are on the wire too, so the receiver has to expect them.
    let pieces = cli.bytes.div_ceil(cli.piece as u64);
    let on_the_wire = cli.bytes + pieces * HEADER_LEN as u64;
    let reps = cli.reps;

    // Reads everything and acknowledges each round, so that the timing covers
    // the bytes arriving rather than the socket accepting them.
    let receiver = std::thread::spawn(move || -> std::io::Result<()> {
        let (mut stream, _) = listener.accept()?;
        stream.set_nodelay(true)?;
        let mut sink = vec![0u8; 4 << 20];
        for _ in 0..reps {
            let mut read = 0u64;
            while read < on_the_wire {
                let want = sink.len().min((on_the_wire - read) as usize);
                let n = stream.read(&mut sink[..want])?;
                if n == 0 {
                    return Err(std::io::ErrorKind::UnexpectedEof.into());
                }
                read += n as u64;
            }
            stream.write_all(b"k")?;
        }
        Ok(())
    });

    let mut stream = TcpStream::connect(addr)?;
    stream.set_nodelay(true)?;

    let mut buffer = vec![0u8; cli.piece];
    let header = [0u8; HEADER_LEN];
    let mut ack = [0u8; 1];

    // Only the threaded mode uses these; the reader fills buffers while the
    // sender writes the last one out, as the server's connection does.
    let (filled_tx, filled_rx) = std::sync::mpsc::channel::<(Vec<u8>, usize)>();
    let (free_tx, free_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let (want_tx, want_rx) = std::sync::mpsc::channel::<(u64, usize)>();
    let reader = if cli.mode == Mode::PreadThreaded {
        for _ in 0..3 {
            free_tx
                .send(vec![0u8; cli.piece])
                .expect("the pool is open");
        }
        let file = File::open(&cli.path)?;
        Some(std::thread::spawn(move || {
            while let Ok((at, piece)) = want_rx.recv() {
                let Ok(mut bytes) = free_rx.recv() else {
                    return;
                };
                file.read_exact_at(&mut bytes[..piece], at).expect("pread");
                if filled_tx.send((bytes, piece)).is_err() {
                    return;
                }
            }
        }))
    } else {
        None
    };

    let mut runs = Vec::with_capacity(cli.reps);
    for _ in 0..cli.reps {
        let cpu = cpu_seconds();
        let started = Instant::now();

        if cli.mode == Mode::PreadThreaded {
            // Queue the whole round before writing any of it, so the reader is
            // never waiting to be told what to do next.
            let mut at = 0u64;
            while at < cli.bytes {
                let piece = cli.piece.min((cli.bytes - at) as usize);
                want_tx
                    .send((cli.offset + at, piece))
                    .expect("the reader is alive");
                at += piece as u64;
            }
        }

        let mut sent = 0u64;
        while sent < cli.bytes {
            let piece = cli.piece.min((cli.bytes - sent) as usize);
            match cli.mode {
                Mode::Pread => {
                    file.read_exact_at(&mut buffer[..piece], cli.offset + sent)?;
                    write_all_vectored(&mut stream, &header, &buffer[..piece])?;
                }
                Mode::PreadThreaded => {
                    let (bytes, len) = filled_rx.recv().expect("the reader is alive");
                    write_all_vectored(&mut stream, &header, &bytes[..len])?;
                    free_tx.send(bytes).expect("the reader is alive");
                }
                Mode::Sendfile => {
                    send_file(&file, &stream, (cli.offset + sent) as i64, piece, &header)?;
                }
            }
            sent += piece as u64;
        }
        stream.read_exact(&mut ack)?;
        runs.push(Run {
            bytes: cli.bytes,
            elapsed: started.elapsed(),
            cpu: cpu_seconds() - cpu,
        });
    }

    drop(stream);
    drop(want_tx);
    receiver
        .join()
        .expect("the receiver thread panicked")
        .expect("the receiver failed");
    if let Some(reader) = reader {
        let _ = reader.join();
    }

    let name = match cli.mode {
        Mode::Pread => "pread + writev, one thread ",
        Mode::PreadThreaded => "pread + writev, overlapped",
        Mode::Sendfile => "sendfile                  ",
    };
    report(&format!("{name} piece={}", cli.piece), &runs);
    Ok(())
}

/// Write a header and a payload, resuming where a short write stopped.
fn write_all_vectored(
    stream: &mut TcpStream,
    header: &[u8],
    payload: &[u8],
) -> std::io::Result<()> {
    let mut slices = [IoSlice::new(header), IoSlice::new(payload)];
    let mut remaining = &mut slices[..];
    while !remaining.is_empty() {
        match stream.write_vectored(remaining) {
            Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
            Ok(n) => IoSlice::advance_slices(&mut remaining, n),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Send `len` bytes of `file` from `offset`, with `header` in front of them.
///
/// macOS counts the header toward the length to send, unlike FreeBSD where the
/// two are separate, so the call asks for both together. The socket is
/// blocking, so a successful call has sent all of it.
#[cfg(target_os = "macos")]
fn send_file(
    file: &File,
    stream: &TcpStream,
    offset: i64,
    len: usize,
    header: &[u8],
) -> std::io::Result<()> {
    let mut iov = libc::iovec {
        iov_base: header.as_ptr() as *mut libc::c_void,
        iov_len: header.len(),
    };
    let mut hdtr = libc::sf_hdtr {
        headers: &mut iov,
        hdr_cnt: 1,
        trailers: std::ptr::null_mut(),
        trl_cnt: 0,
    };
    let mut sent = (len + header.len()) as libc::off_t;

    // SAFETY: the descriptors are open for the call, and the iovec and the
    // length outlive it.
    let rc = unsafe {
        libc::sendfile(
            file.as_raw_fd(),
            stream.as_raw_fd(),
            offset,
            &mut sent,
            &mut hdtr,
            0,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Send `len` bytes of `file` from `offset`, with `header` in front of them.
///
/// Linux's `sendfile` carries only the file, so the header goes first under
/// `MSG_MORE`, which holds it back until the payload joins it rather than
/// pushing a 32-byte segment on its own.
#[cfg(target_os = "linux")]
fn send_file(
    file: &File,
    stream: &TcpStream,
    offset: i64,
    len: usize,
    header: &[u8],
) -> std::io::Result<()> {
    // SAFETY: the socket is open for the call and the header outlives it.
    let sent = unsafe {
        libc::send(
            stream.as_raw_fd(),
            header.as_ptr() as *const libc::c_void,
            header.len(),
            libc::MSG_MORE,
        )
    };
    if sent < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if sent as usize != header.len() {
        return Err(std::io::ErrorKind::WriteZero.into());
    }

    let mut at = offset as libc::off_t;
    let mut left = len;
    while left > 0 {
        // SAFETY: both descriptors are open, and `at` outlives the call.
        let n = unsafe { libc::sendfile(stream.as_raw_fd(), file.as_raw_fd(), &mut at, left) };
        if n < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if n == 0 {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        left -= n as usize;
    }
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn send_file(
    _file: &File,
    _stream: &TcpStream,
    _offset: i64,
    _len: usize,
    _header: &[u8],
) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "sendfile is only wired up for macOS and Linux",
    ))
}
