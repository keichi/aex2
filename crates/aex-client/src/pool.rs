//! The data connections of a session.
//!
//! The connections are opened with the session and kept. A transfer puts its
//! chunks on one queue and starts a thread per connection it needs; each thread
//! takes the next chunk whenever it has room, so a slow connection simply ends
//! up carrying less (work stealing). Room is the credit: how many fetches a
//! connection may have outstanding at once, which hides the round trip on a
//! long link.
//!
//! The connection carries no state past the handshake, which is what makes
//! recovery simple: a chunk is asked for by its position in the logical byte
//! stream, so a chunk lost with a connection goes back on the queue and any
//! connection can ask for it again.
//!
//! The threads are scoped to the transfer. That costs a thread start per
//! transfer, which is nothing next to a transfer too large to come back inline,
//! and it lets the compiler prove the output buffer outlives every writer.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

use aex_core::{Codec, ErrorClass};
use aex_wire::{
    read_frame_header, write_frame, ErrorPayload, FrameHeader, FrameType, Hello, Ready,
    ScatterBuffer, ScatterSlice, Ticket, READY_LEN,
};
use socket2::{Domain, Protocol, Socket, Type};

use crate::error::{ClientError, Result};

/// How long one read or write on a data connection may take.
///
/// This is per syscall, not per transfer, so it only ever fires on a server
/// that has stopped answering. It matches the server's default idle timeout,
/// past which the server would have hung up anyway.
const IO_TIMEOUT: Duration = Duration::from_secs(300);

/// What it takes to open a data connection.
#[derive(Debug, Clone)]
pub struct ConnSettings {
    pub host: String,
    pub port: u16,
    pub session_id: [u8; 16],
    /// Never logged: it is what a connection proves itself with.
    pub session_token: [u8; 16],
    pub nodelay: bool,
    pub rcvbuf: Option<usize>,
    pub connect_timeout: Duration,
}

/// One data connection, past its handshake.
struct DataConn {
    stream: TcpStream,
}

impl DataConn {
    fn connect(settings: &ConnSettings) -> Result<Self> {
        let addrs: Vec<_> = (settings.host.as_str(), settings.port)
            .to_socket_addrs()
            .map_err(|e| {
                ClientError::BadRequest(format!(
                    "cannot resolve the data endpoint {}:{}: {e}",
                    settings.host, settings.port
                ))
            })?
            .collect();

        let mut last = None;
        let stream = addrs
            .iter()
            .find_map(|addr| match dial(addr, settings) {
                Ok(stream) => Some(stream),
                Err(e) => {
                    last = Some(e);
                    None
                }
            })
            .ok_or_else(|| {
                let reason = last.map(|e| e.to_string()).unwrap_or_else(|| {
                    format!("{}:{} resolved to no address", settings.host, settings.port)
                });
                ClientError::Data {
                    class: ErrorClass::Transient,
                    message: format!("cannot reach the data plane: {reason}"),
                }
            })?;

        // Without this, Nagle holds a fetch back waiting for more to send, and
        // a fetch is 48 bytes that the whole chunk is waiting on.
        stream.set_nodelay(settings.nodelay)?;
        stream.set_read_timeout(Some(IO_TIMEOUT))?;
        stream.set_write_timeout(Some(IO_TIMEOUT))?;

        let mut conn = DataConn { stream };
        conn.handshake(settings)?;
        Ok(conn)
    }

    fn handshake(&mut self, settings: &ConnSettings) -> Result<()> {
        let hello = Hello::new(settings.session_id, settings.session_token);
        self.stream.write_all(&hello.encode())?;

        let mut bytes = [0u8; READY_LEN];
        self.stream.read_exact(&mut bytes)?;
        let ready = Ready::decode(&bytes)?;

        if !ready.is_accepted() {
            // READY carries no message; the class is what the client acts on,
            // and the reason in full is in the server's log.
            return Err(ClientError::Data {
                class: ready.status,
                message: format!("the server refused the data connection: {:?}", ready.status),
            });
        }
        Ok(())
    }

    fn send_fetch(
        &mut self,
        request_id: u32,
        ticket: &Ticket,
        offset: u64,
        len: u64,
    ) -> Result<()> {
        let header = FrameHeader::fetch(request_id, offset, len);
        write_frame(&mut self.stream, &header, ticket)?;
        Ok(())
    }

    fn read_header(&mut self) -> Result<FrameHeader> {
        Ok(read_frame_header(&mut self.stream)?)
    }

    fn pong(&mut self) -> Result<()> {
        write_frame(&mut self.stream, &FrameHeader::bare(FrameType::Pong), &[])?;
        Ok(())
    }

    /// Read one data frame into its place in the chunk starting at `offset`.
    ///
    /// The bytes go from the kernel into the caller's buffer with nothing in
    /// between, which is the whole reason the payload is raw and the header is
    /// fixed width. A server may answer one fetch with several frames, so the
    /// position comes from the frame rather than from how much has arrived.
    fn receive_data(
        &mut self,
        header: &FrameHeader,
        request_id: u32,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<()> {
        if header.request_id != request_id {
            return Err(ClientError::Protocol(format!(
                "a data frame for transfer {} arrived on a fetch of transfer {request_id}",
                header.request_id
            )));
        }
        if header.codec != Codec::Raw || header.wire_len != header.logical_len {
            // Nothing negotiates a codec yet, so this means the server sent
            // something this client never asked for and cannot expand.
            return Err(ClientError::Protocol(format!(
                "a data frame arrived {:?} encoded; this client asked for raw bytes",
                header.codec
            )));
        }

        let start = header.offset.checked_sub(offset).filter(|start| {
            start
                .checked_add(header.logical_len)
                .is_some_and(|end| end <= dst.len() as u64)
        });
        let Some(start) = start else {
            return Err(ClientError::Protocol(format!(
                "a data frame covering [{}, {}) does not fall inside the fetch of [{offset}, {})",
                header.offset,
                header.offset.saturating_add(header.logical_len),
                offset + dst.len() as u64
            )));
        };

        let start = start as usize;
        let end = start + header.logical_len as usize;
        self.stream.read_exact(&mut dst[start..end])?;
        Ok(())
    }

    /// Turn an error frame into the error it stands for.
    fn receive_error(&mut self, header: &FrameHeader) -> Result<ClientError> {
        let mut payload = vec![0u8; header.wire_len as usize];
        self.stream.read_exact(&mut payload)?;
        let error = ErrorPayload::decode(&payload)?;
        Ok(ClientError::Data {
            class: error.class,
            message: error.message,
        })
    }
}

/// Connect one socket, sizing its receive buffer first.
///
/// Before the connect, because the window scale is agreed in the SYN and a
/// buffer enlarged afterwards cannot be advertised in full.
fn dial(addr: &SocketAddr, settings: &ConnSettings) -> std::io::Result<TcpStream> {
    let socket = Socket::new(
        Domain::for_address(*addr),
        Type::STREAM,
        Some(Protocol::TCP),
    )?;
    if let Some(bytes) = settings.rcvbuf {
        socket.set_recv_buffer_size(bytes)?;
    }
    socket.connect_timeout(&(*addr).into(), settings.connect_timeout)?;
    Ok(socket.into())
}

/// How the pool runs one batch of transfers.
#[derive(Debug, Clone, Copy)]
pub struct FetchSpec {
    /// Fetches one connection may have outstanding.
    pub credit: u32,
    /// Times one chunk may be fetched again before the transfer fails.
    pub max_retries: u32,
}

/// One plan's share of a batch: its chunks, and where they go.
pub struct FetchPart<'a> {
    pub request_id: u32,
    pub ticket: &'a Ticket,
    pub chunks: Vec<(u64, u64)>,
    pub dst: &'a mut [u8],
}

/// A part once its buffer is split up for the connections.
struct Part<'a> {
    request_id: u32,
    ticket: &'a Ticket,
    scatter: ScatterBuffer<'a>,
}

/// How a transfer went on the data plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fetched {
    pub streams: u32,
    pub retries: u32,
}

/// The session's data connections.
pub struct DataPool {
    settings: ConnSettings,
    /// `None` once a connection has broken, until a transfer rebuilds it.
    conns: Vec<Mutex<Option<DataConn>>>,
}

impl std::fmt::Debug for DataPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataPool")
            .field("endpoint", &(&self.settings.host, self.settings.port))
            .field("streams", &self.conns.len())
            .finish()
    }
}

impl DataPool {
    /// Open the session's connections.
    ///
    /// Eagerly, while the session is being set up, so that the first large
    /// transfer does not pay for handshakes and so that a data plane which
    /// cannot be reached is reported as such rather than much later. All at
    /// once, so that the handshakes cost one round trip rather than one each.
    pub fn connect(settings: ConnSettings, streams: u32) -> Result<Self> {
        let conns = std::thread::scope(|scope| {
            let handshakes: Vec<_> = (0..streams.max(1))
                .map(|_| scope.spawn(|| DataConn::connect(&settings)))
                .collect();
            handshakes
                .into_iter()
                .map(|h| {
                    let conn = h.join().expect("a handshake thread panicked")?;
                    Ok(Mutex::new(Some(conn)))
                })
                .collect::<Result<_>>()
        })?;
        Ok(DataPool { settings, conns })
    }

    /// Connections held.
    pub fn streams(&self) -> u32 {
        self.conns.len() as u32
    }

    /// Fetch the chunks of every part into its buffer, over as many
    /// connections as there are chunks to keep busy.
    ///
    /// The parts share one queue, so a batch of plans fills the connections as
    /// one transfer would, rather than one plan at a time.
    pub fn fetch(&self, spec: FetchSpec, parts: Vec<FetchPart<'_>>) -> Result<Fetched> {
        let mut queue = VecDeque::new();
        let parts: Vec<Part<'_>> = parts
            .into_iter()
            .enumerate()
            .map(|(part, p)| {
                queue.extend(p.chunks.iter().map(|&(offset, len)| Chunk {
                    part,
                    offset,
                    len,
                    attempts: 0,
                }));
                Part {
                    request_id: p.request_id,
                    ticket: p.ticket,
                    scatter: ScatterBuffer::new(p.dst),
                }
            })
            .collect();
        let used = queue.len().min(self.conns.len());
        if used == 0 {
            return Ok(Fetched {
                streams: 0,
                retries: 0,
            });
        }

        let transfer = Transfer {
            pool: self,
            spec,
            parts: &parts,
            state: Mutex::new(State {
                remaining: queue.len(),
                queue,
                live: used,
                retries: 0,
                error: None,
                last_conn_error: None,
            }),
            wake: Condvar::new(),
        };

        std::thread::scope(|scope| {
            for slot in &self.conns[..used] {
                let spawned = std::thread::Builder::new()
                    .name("aex-data".to_string())
                    .spawn_scoped(scope, || transfer.work(slot));
                if let Err(e) = spawned {
                    // The other connections pick up its share.
                    transfer.leave(Some(e.into()));
                }
            }
        });

        let state = transfer
            .state
            .into_inner()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(error) = state.error {
            return Err(error);
        }
        if state.remaining > 0 {
            return Err(state.last_conn_error.unwrap_or_else(|| ClientError::Data {
                class: ErrorClass::Transient,
                message: "every data connection was lost".to_string(),
            }));
        }
        Ok(Fetched {
            streams: used as u32,
            retries: state.retries,
        })
    }

    /// Open a connection to replace one that broke.
    ///
    /// The server counts a connection until its thread notices the hang-up, so
    /// a refusal right after a break is expected and worth a short wait.
    fn reconnect(&self, attempts: u32) -> Result<DataConn> {
        let mut attempt = 0;
        loop {
            match DataConn::connect(&self.settings) {
                Err(e) if e.is_retryable() && attempt < attempts => {
                    attempt += 1;
                    std::thread::sleep(Duration::from_millis(50) * attempt);
                }
                result => return result,
            }
        }
    }
}

/// A range of the logical byte stream still to be fetched.
#[derive(Debug, Clone, Copy)]
struct Chunk {
    /// Which part of the batch it belongs to.
    part: usize,
    offset: u64,
    len: u64,
    /// Fetches of it that failed.
    attempts: u32,
}

/// A chunk that has been asked for on a connection.
struct InFlight<'s> {
    chunk: Chunk,
    slice: ScatterSlice<'s>,
    received: u64,
}

/// What the threads of one transfer share.
struct State {
    queue: VecDeque<Chunk>,
    /// Chunks not yet complete, whether queued or in flight.
    remaining: usize,
    /// Threads still working, this one included.
    live: usize,
    retries: u32,
    /// Ends the transfer: no more fetches go out, and what is in flight drains.
    error: Option<ClientError>,
    /// Why a connection could not be rebuilt; reported if none are left.
    last_conn_error: Option<ClientError>,
}

struct Transfer<'a> {
    pool: &'a DataPool,
    spec: FetchSpec,
    parts: &'a [Part<'a>],
    state: Mutex<State>,
    wake: Condvar,
}

impl Transfer<'_> {
    fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// One connection's share of the transfer.
    fn work(&self, slot: &Mutex<Option<DataConn>>) {
        // Leaves the transfer even on a panic, so the others are not left
        // waiting for chunks this thread will never finish.
        struct Leave<'t, 'a>(&'t Transfer<'a>, Option<ClientError>);
        impl Drop for Leave<'_, '_> {
            fn drop(&mut self) {
                if std::thread::panicking() {
                    self.0.fail(ClientError::Protocol(
                        "a data connection thread panicked".to_string(),
                    ));
                }
                self.0.leave(self.1.take());
            }
        }
        let mut leave = Leave(self, None);

        let mut held = lock(slot);
        let mut flight = VecDeque::new();
        leave.1 = self.drive(&mut held, &mut flight).err();
    }

    /// Take a thread out of the transfer.
    fn leave(&self, conn_error: Option<ClientError>) {
        let mut state = self.lock();
        state.live -= 1;
        if conn_error.is_some() {
            state.last_conn_error = conn_error;
        }
        drop(state);
        self.wake.notify_all();
    }

    /// Serve the transfer on one connection until nothing is left for it.
    ///
    /// Fails only when the connection cannot be rebuilt, with its chunks
    /// already back on the queue for the others.
    fn drive<'s>(
        &'s self,
        held: &mut Option<DataConn>,
        flight: &mut VecDeque<InFlight<'s>>,
    ) -> Result<()> {
        loop {
            let conn = match held {
                Some(conn) => conn,
                None if self.lock().error.is_some() => return Ok(()),
                None => held.insert(self.pool.reconnect(self.spec.max_retries)?),
            };
            let error = match self.step(conn, flight) {
                Ok(true) => continue,
                Ok(false) => return Ok(()),
                Err(e) => e,
            };

            // Either way the connection is in an unknown state, mid-frame
            // perhaps, so it is not used again.
            *held = None;
            if let ClientError::Io(_) = error {
                let mut exhausted = false;
                let oldest = flight.len().saturating_sub(1);
                // Newest first, so that each lands in front of the last and
                // they go out again in their original order.
                for (i, InFlight { chunk, slice, .. }) in flight.drain(..).rev().enumerate() {
                    // Released before it is queued, or another connection
                    // could find the range still claimed.
                    drop(slice);
                    // Only the oldest was being answered; the rest were just
                    // queued behind it, and deeper credit must not make a
                    // break more likely to fail the transfer.
                    exhausted |= !self.retry(chunk, i == oldest);
                }
                if exhausted {
                    self.fail(error);
                }
            } else {
                flight.clear();
                self.fail(error);
            }
        }
    }

    /// Top up this connection's fetches, then handle one frame.
    ///
    /// Returns false once there is nothing left for this connection to do.
    fn step<'s>(
        &'s self,
        conn: &mut DataConn,
        flight: &mut VecDeque<InFlight<'s>>,
    ) -> Result<bool> {
        let credit = self.spec.credit.max(1) as usize;
        let mut fresh: VecDeque<Chunk> = {
            let mut state = self.lock();
            loop {
                let mut fresh = VecDeque::new();
                if state.error.is_none() {
                    while flight.len() + fresh.len() < credit {
                        match state.queue.pop_front() {
                            Some(chunk) => fresh.push_back(chunk),
                            None => break,
                        }
                    }
                }
                if !flight.is_empty() || !fresh.is_empty() {
                    break fresh;
                }
                if state.error.is_some() || state.remaining == 0 {
                    return Ok(false);
                }
                if state.live == 1 {
                    // Last one here, with nothing queued and nothing of its
                    // own in flight: no chunk can come back, so waiting would
                    // hang the transfer instead of failing it.
                    return Err(ClientError::Protocol(format!(
                        "{} chunks of the transfer are unaccounted for",
                        state.remaining
                    )));
                }
                // Everything left is in flight elsewhere; wait in case some of
                // it comes back.
                state = self.wake.wait(state).unwrap_or_else(|p| p.into_inner());
            }
        };

        while let Some(chunk) = fresh.pop_front() {
            let part = &self.parts[chunk.part];
            let asked = match part.scatter.claim(chunk.offset, chunk.len) {
                Ok(slice) => {
                    flight.push_back(InFlight {
                        chunk,
                        slice,
                        received: 0,
                    });
                    conn.send_fetch(part.request_id, part.ticket, chunk.offset, chunk.len)
                }
                Err(e) => Err(e.into()),
            };
            if let Err(e) = asked {
                // The connection broke partway through topping it up. What is
                // in flight goes back with it, but these were taken off the
                // queue and never asked for, so nothing else would fetch them
                // and the transfer would wait for them forever.
                self.give_back(fresh);
                return Err(e);
            }
        }

        let header = conn.read_header()?;
        match header.frame_type {
            FrameType::Data => {
                // The server answers fetches in order, so data is always for
                // the oldest one outstanding.
                let front = flight.front_mut().expect("a fetch is outstanding");
                let request_id = self.parts[front.chunk.part].request_id;
                conn.receive_data(&header, request_id, front.chunk.offset, &mut front.slice)?;
                front.received += header.logical_len;
                if front.received >= front.chunk.len {
                    flight.pop_front();
                    self.complete();
                }
            }
            FrameType::Error => {
                let error = conn.receive_error(&header)?;
                if header.request_id == 0 {
                    // About the connection, not about a fetch.
                    return Err(error);
                }
                let InFlight { chunk, slice, .. } =
                    flight.pop_front().expect("a fetch is outstanding");
                drop(slice);
                let request_id = self.parts[chunk.part].request_id;
                if header.request_id != request_id || header.offset != chunk.offset {
                    return Err(ClientError::Protocol(format!(
                        "an error for [{}, ..) of transfer {} arrived while [{}, ..) of {request_id} was outstanding",
                        header.offset, header.request_id, chunk.offset
                    )));
                }
                if !(error.is_retryable() && self.retry(chunk, true)) {
                    self.fail(error);
                }
            }
            FrameType::Ping => conn.pong()?,
            FrameType::Pong => {}
            other => {
                return Err(ClientError::Protocol(format!(
                    "{other:?} is not a frame a server may send"
                )))
            }
        }
        Ok(true)
    }

    fn complete(&self) {
        let mut state = self.lock();
        state.remaining -= 1;
        if state.remaining == 0 {
            drop(state);
            self.wake.notify_all();
        }
    }

    /// Put chunks that were taken off the queue but never asked for back on it.
    ///
    /// Not counted as retries: nothing was sent, so they have not been tried.
    fn give_back(&self, chunks: VecDeque<Chunk>) {
        if chunks.is_empty() {
            return;
        }
        let mut state = self.lock();
        for chunk in chunks.into_iter().rev() {
            state.queue.push_front(chunk);
        }
        drop(state);
        self.wake.notify_all();
    }

    /// Put a chunk back at the front of the queue, counting it against its
    /// retries if `charge`. False once it has used them up.
    fn retry(&self, mut chunk: Chunk, charge: bool) -> bool {
        chunk.attempts += u32::from(charge);
        if chunk.attempts > self.spec.max_retries {
            return false;
        }
        let mut state = self.lock();
        state.retries += 1;
        state.queue.push_front(chunk);
        drop(state);
        self.wake.notify_all();
        true
    }

    /// End the transfer. The first error is the one reported.
    fn fail(&self, error: ClientError) {
        self.lock().error.get_or_insert(error);
        self.wake.notify_all();
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
