//! The data plane: raw TCP, two dedicated threads per connection.
//!
//! Reading a regular file has no true asynchronous form on Linux or macOS —
//! `O_NONBLOCK` does not apply to it, and epoll and kqueue always report it
//! ready — which is why `tokio::fs` hands the work to a blocking pool anyway.
//! An async runtime here would therefore not save a thread, so the connections
//! get real ones and the control plane keeps tokio to itself.
//!
//! Each connection has two: one reading ([`crate::reader`]) and one writing.
//! Doing both in turn on one thread leaves the disk idle while the socket works
//! and the socket idle while the disk works, and for data that does not fit in
//! memory that is most of the throughput.
//!
//! A connection carries no state past the handshake. Any connection may serve
//! any fetch of its session, which is what later makes work stealing, parallel
//! streams and re-fetching a lost chunk all fall out for free.
//!
//! Nothing is encrypted and no user is identified here. The session token
//! proves which session a connection belongs to, and the ticket in each `FETCH`
//! proves which transfers it may read; who may open which file was settled by
//! the control plane before any of this existed.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use aex_core::{codec, ErrorClass, SelectionLayout};
use aex_wire::{
    write_frame, ErrorPayload, FrameHeader, FrameType, Hello, Ready, Ticket, HEADER_LEN, HELLO_LEN,
    PROTOCOL_VERSION, TICKET_LEN,
};
use socket2::{Domain, Protocol, Socket, Type};

use crate::config::ServerConfig;
use crate::error::{Result, ServerError};
use crate::reader::{Piece, ReadPipeline};
use crate::session::{Session, SessionId, SessionRegistry};
use crate::transfer::TransferRegistry;

/// How often a blocked thread looks up to see whether the server is stopping.
///
/// Short enough that shutting down is not noticeable, long enough that an idle
/// connection costs a handful of wakeups a second.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// How long a connection has to finish its handshake.
///
/// Separate from the idle timeout, which is measured between frames: a peer
/// that connects and then says nothing at all should not hold a thread for the
/// several minutes a working connection is allowed to be quiet for.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// What to do after handling a frame.
enum Disposition {
    /// Keep serving this connection.
    Continue,
    /// Hang up. Used where continuing would mean talking to a peer that has
    /// already proved it is not following the protocol.
    Close,
}

/// A data plane that has its socket and is ready to serve.
pub struct DataPlane {
    listener: TcpListener,
    context: Arc<Context>,
}

/// What every connection thread needs.
struct Context {
    sessions: Arc<SessionRegistry>,
    transfers: Arc<TransferRegistry>,
    config: Arc<ServerConfig>,
    stop: Arc<AtomicBool>,
}

impl DataPlane {
    /// Take the socket.
    ///
    /// Separate from serving so that a caller can bind port 0 and still learn
    /// where to connect, which is also what the control plane advertises.
    pub fn bind(
        config: Arc<ServerConfig>,
        sessions: Arc<SessionRegistry>,
        transfers: Arc<TransferRegistry>,
    ) -> Result<Self> {
        let listener = listen(&config)?;
        // The accept loop polls rather than blocking, so that it can be told to
        // stop without something else having to connect to wake it.
        listener.set_nonblocking(true)?;

        Ok(DataPlane {
            listener,
            context: Arc::new(Context {
                sessions,
                transfers,
                config,
                stop: Arc::new(AtomicBool::new(false)),
            }),
        })
    }

    /// The address actually bound, which differs from the configured one when
    /// the port was 0.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Start accepting. Serving stops when the returned handle is dropped.
    pub fn serve(self) -> DataPlaneHandle {
        let DataPlane { listener, context } = self;
        let stop = context.stop.clone();
        let connections: Arc<Mutex<Vec<JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));

        let accept = {
            let connections = connections.clone();
            std::thread::Builder::new()
                .name("aex-data-accept".to_string())
                .spawn(move || accept_loop(listener, context, connections))
                .expect("the accept thread must start")
        };

        DataPlaneHandle {
            stop,
            accept: Some(accept),
            connections,
        }
    }
}

/// Keeps the data plane running, and stops it when dropped.
pub struct DataPlaneHandle {
    stop: Arc<AtomicBool>,
    accept: Option<JoinHandle<()>>,
    connections: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl DataPlaneHandle {
    /// Stop accepting and wait for the connections to notice.
    ///
    /// Each thread is between polls at worst, so this takes about as long as
    /// one poll interval rather than as long as a transfer.
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
        }
        let threads = std::mem::take(&mut *lock(&self.connections));
        for thread in threads {
            let _ = thread.join();
        }
    }
}

impl Drop for DataPlaneHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn accept_loop(
    listener: TcpListener,
    context: Arc<Context>,
    connections: Arc<Mutex<Vec<JoinHandle<()>>>>,
) {
    while !context.stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, peer)) => {
                let context = context.clone();
                let thread = std::thread::Builder::new()
                    .name("aex-data-conn".to_string())
                    .spawn(move || serve_connection(stream, peer, context));
                match thread {
                    Ok(thread) => {
                        let mut live = lock(&connections);
                        // Threads that have already finished would otherwise
                        // accumulate for the life of the server.
                        live.retain(|thread| !thread.is_finished());
                        live.push(thread);
                    }
                    Err(e) => tracing::error!("cannot start a connection thread: {e}"),
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => std::thread::sleep(POLL_INTERVAL),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => {
                tracing::error!("data plane accept failed: {e}");
                std::thread::sleep(POLL_INTERVAL);
            }
        }
    }
}

fn serve_connection(mut stream: TcpStream, peer: SocketAddr, context: Arc<Context>) {
    // Back to blocking, with a short timeout so that a read can be interrupted
    // by the server stopping.
    if let Err(e) = configure(&stream) {
        tracing::warn!(%peer, "cannot configure a data connection: {e}");
        return;
    }

    let session = match handshake(&mut stream, &context) {
        Ok(session) => session,
        Err(e) => {
            tracing::info!(%peer, "data connection refused: {e}");
            return;
        }
    };
    let session_id = *session.id();
    tracing::debug!(%peer, session = %hex(&session_id), "data connection accepted");

    let result = serve_frames(&mut stream, &session_id, &context);
    session.close_data_conn();
    match result {
        Ok(()) => tracing::debug!(%peer, "data connection closed"),
        Err(e) => tracing::info!(%peer, "data connection ended: {e}"),
    }
}

/// Open the listening socket with the configured TCP options.
///
/// They are set on the listener because accepted sockets inherit them, and so
/// that a congestion control algorithm the kernel lacks fails at startup rather
/// than on every connection.
fn listen(config: &ServerConfig) -> Result<TcpListener> {
    let addr = config.data_addr;
    let socket = Socket::new(Domain::for_address(addr), Type::STREAM, Some(Protocol::TCP))?;
    // A restarted server would otherwise wait out TIME_WAIT on its own port.
    socket.set_reuse_address(true)?;
    if config.tcp.sndbuf != 0 {
        socket.set_send_buffer_size(config.tcp.sndbuf)?;
        // Linux silently caps it at net.core.wmem_max.
        let applied = socket.send_buffer_size()?;
        if applied < config.tcp.sndbuf {
            tracing::warn!(
                "tcp.sndbuf {} was capped at {applied} by the OS (net.core.wmem_max on Linux)",
                config.tcp.sndbuf
            );
        }
    }
    if !config.tcp.congestion.is_empty() {
        set_congestion(&socket, &config.tcp.congestion)?;
    }
    socket.bind(&addr.into())?;
    socket.listen(1024)?;
    Ok(socket.into())
}

#[cfg(target_os = "linux")]
fn set_congestion(socket: &Socket, name: &str) -> Result<()> {
    socket
        .set_tcp_congestion(name.as_bytes())
        .map_err(|e| ServerError::Config(format!("tcp.congestion {name:?} cannot be applied: {e}")))
}

#[cfg(not(target_os = "linux"))]
fn set_congestion(_socket: &Socket, name: &str) -> Result<()> {
    // Saying so beats leaving an operator to conclude from a benchmark that the
    // setting made no difference.
    tracing::warn!("tcp.congestion {name:?} is ignored: it is only applied on Linux");
    Ok(())
}

fn configure(stream: &TcpStream) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    // Without this, Nagle holds back a DATA header whose payload has already
    // gone, and every small reply waits for an ack.
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(POLL_INTERVAL))?;
    Ok(())
}

/// Check the client in, or refuse it and say which class of thing was wrong.
fn handshake(stream: &mut TcpStream, context: &Context) -> Result<Arc<Session>> {
    let mut bytes = [0u8; HELLO_LEN];
    if !read_full(stream, &mut bytes, &context.stop, HANDSHAKE_TIMEOUT)? {
        // A connection that opened and closed without saying anything. A port
        // scan or a health check, not something to report as a refusal.
        return Err(ServerError::BadRequest("no handshake was sent".to_string()));
    }

    let result = check_hello(&bytes, context);
    let ready = match &result {
        Ok(_) => Ready::accepted(),
        Err(e) => Ready::refused(e.class()),
    };
    // Answer before hanging up: READY carries no message, but its class is all
    // the client needs in order to know what to do next.
    stream.write_all(&ready.encode())?;
    result
}

fn check_hello(bytes: &[u8; HELLO_LEN], context: &Context) -> Result<Arc<Session>> {
    let hello = Hello::decode(bytes).map_err(|e| ServerError::Protocol(e.to_string()))?;
    if hello.version != PROTOCOL_VERSION {
        return Err(ServerError::Protocol(format!(
            "client speaks data plane version {}, this server speaks {PROTOCOL_VERSION}",
            hello.version
        )));
    }

    let session = context.sessions.get(&hello.session_id)?;
    if !constant_time_eq(session.token(), &hello.session_token) {
        return Err(ServerError::Auth(
            "the token presented is not the one this session was issued".to_string(),
        ));
    }
    // Counted here so that it is released exactly when the connection ends.
    session.open_data_conn()?;
    Ok(session)
}

/// Serve fetches until the peer hangs up, goes quiet or breaks the protocol.
fn serve_frames(stream: &mut TcpStream, session: &SessionId, context: &Context) -> Result<()> {
    let idle = Duration::from_secs(context.config.limits.data_conn_idle_timeout_sec);
    // The buffer is allocated here and reused for the life of the connection,
    // so a transfer in progress allocates nothing.
    let pipeline = ReadPipeline::start(context.config.transfer.read_buffer_bytes as usize);
    // Where a block is compressed before it goes out. Empty for a transfer
    // that is not encoded, and reused for the life of the connection.
    let mut packed = Vec::new();

    loop {
        let mut bytes = [0u8; HEADER_LEN];
        if !read_full(stream, &mut bytes, &context.stop, idle)? {
            return Ok(());
        }
        let header = match FrameHeader::decode(&bytes) {
            Ok(header) => header,
            Err(e) => {
                send_connection_error(stream, e.class(), &e)?;
                return Err(ServerError::Protocol(e.to_string()));
            }
        };

        // Any activity keeps the session alive, so a transfer in progress
        // cannot be cut off by the control plane's idle timeout.
        context.sessions.touch(session);

        match handle_frame(stream, &header, session, context, &pipeline, &mut packed)? {
            Disposition::Continue => {}
            Disposition::Close => return Ok(()),
        }
    }
}

fn handle_frame(
    stream: &mut TcpStream,
    header: &FrameHeader,
    session: &SessionId,
    context: &Context,
    pipeline: &ReadPipeline,
    packed: &mut Vec<u8>,
) -> Result<Disposition> {
    match header.frame_type {
        FrameType::Fetch => handle_fetch(stream, header, session, context, pipeline, packed),
        FrameType::Ping => {
            write_frame(stream, &FrameHeader::bare(FrameType::Pong), &[])?;
            Ok(Disposition::Continue)
        }
        // A reply to a ping this server sent. Nothing to do but note that the
        // peer is alive, which reading the frame already did.
        FrameType::Pong => Ok(Disposition::Continue),
        // Server-to-client frames arriving from a client mean the two
        // implementations disagree about who says what. That is wrong with the
        // connection rather than with any one fetch, so it is reported against
        // no transfer at all.
        other => {
            let e = ServerError::Protocol(format!("{other:?} is not a frame a client may send"));
            send_connection_error(stream, ErrorClass::Protocol, &e)?;
            Ok(Disposition::Close)
        }
    }
}

fn handle_fetch(
    stream: &mut TcpStream,
    header: &FrameHeader,
    session: &SessionId,
    context: &Context,
    pipeline: &ReadPipeline,
    packed: &mut Vec<u8>,
) -> Result<Disposition> {
    if header.wire_len != TICKET_LEN as u64 {
        let e = ServerError::Protocol(format!(
            "a fetch carries a {TICKET_LEN} byte ticket, not {} bytes",
            header.wire_len
        ));
        send_error(stream, header, ErrorClass::Protocol, &e)?;
        return Ok(Disposition::Close);
    }

    let mut ticket: Ticket = [0u8; TICKET_LEN];
    let idle = Duration::from_secs(context.config.limits.data_conn_idle_timeout_sec);
    if !read_full(stream, &mut ticket, &context.stop, idle)? {
        return Err(ServerError::Protocol(
            "a fetch header arrived without its ticket".to_string(),
        ));
    }

    let max_fetch = context.config.transfer.max_fetch_bytes;
    if header.logical_len > max_fetch {
        // Not fatal: the client can ask again for less.
        let e = ServerError::BadRequest(format!(
            "a fetch of {} bytes is over this server's limit of {max_fetch}",
            header.logical_len
        ));
        send_error(stream, header, ErrorClass::Request, &e)?;
        return Ok(Disposition::Continue);
    }

    let entry = match context.transfers.fetch(header.request_id, &ticket, session) {
        Ok(entry) => entry,
        Err(e) => {
            let class = e.class();
            send_error(stream, header, class, &e)?;
            // A wrong ticket is either a bug or someone guessing at another
            // session's transfers; neither is worth carrying on with.
            return Ok(match class {
                ErrorClass::Auth => Disposition::Close,
                _ => Disposition::Continue,
            });
        }
    };

    if let Err(e) = entry
        .layout()
        .check_range(header.offset, header.logical_len)
    {
        let class = e.class();
        send_error(stream, header, class, &e)?;
        return Ok(Disposition::Continue);
    }

    // An encoded transfer is cut into blocks of whole elements, so a fetch that
    // does not name whole elements cannot be answered at all.
    let itemsize = entry.layout().dtype.itemsize();
    if entry.layout().quality.eps().is_some()
        && (header.offset % itemsize != 0 || header.logical_len % itemsize != 0)
    {
        let e = ServerError::BadRequest(format!(
            "an error-bounded transfer is fetched in whole {} elements, and [{}, {}) is not",
            entry.layout().dtype,
            header.offset,
            header.offset.saturating_add(header.logical_len)
        ));
        send_error(stream, header, ErrorClass::Request, &e)?;
        return Ok(Disposition::Continue);
    }

    // Hand the range to the reader and write out each piece as it arrives. The
    // pieces go as separate DATA frames; a fetch and a frame were never
    // required to be the same size, and this is what that is for.
    pipeline.request(entry.clone(), header.offset, header.logical_len);
    loop {
        match pipeline.next_piece() {
            Piece::Data { offset, bytes, len } => {
                let sent = send_piece(
                    stream,
                    header.request_id,
                    entry.layout(),
                    offset,
                    &bytes[..len],
                    packed,
                );
                pipeline.recycle(bytes);
                sent?;
            }
            Piece::Done => break,
            Piece::Failed(e) => {
                // Whatever went out before this stands; the client abandons the
                // fetch on the error and asks for the same range again, which
                // is always safe because a fetch names what it wants.
                let class = e.class();
                send_error(stream, header, class, &e)?;
                return Ok(Disposition::Continue);
            }
        }
        // A large fetch off a cold cache takes a while, and the plan must not
        // expire out from under the reply it is still sending.
        context.transfers.touch(&entry);
    }

    Ok(Disposition::Continue)
}

/// Write one piece out, compressed if the transfer asked for that.
///
/// A block that does not shrink goes raw. The codec is named per frame, so one
/// transfer can carry both kinds and the receiver reads each frame for what it
/// says it is.
fn send_piece(
    stream: &mut TcpStream,
    request_id: u32,
    layout: &SelectionLayout,
    offset: u64,
    bytes: &[u8],
    packed: &mut Vec<u8>,
) -> Result<()> {
    let len = bytes.len() as u64;
    if let Some(spec) = layout.block(offset, len)? {
        let codec = layout.quality.codec();
        if codec::compress(codec, &spec, bytes, packed)? {
            let frame = FrameHeader::data_encoded(
                request_id,
                offset,
                codec,
                layout.quality.encoding,
                packed.len() as u64,
                len,
            );
            write_frame(stream, &frame, packed)?;
            return Ok(());
        }
    }
    write_frame(stream, &FrameHeader::data(request_id, offset, len), bytes)?;
    Ok(())
}

/// Report a failure against the fetch that caused it.
///
/// The reply names that fetch — its transfer, offset and length — so that the
/// client can put exactly that chunk back on its queue rather than starting the
/// transfer again.
fn send_error(
    stream: &mut TcpStream,
    fetch: &FrameHeader,
    class: ErrorClass,
    error: impl std::fmt::Display,
) -> Result<()> {
    send_error_for(
        stream,
        fetch.request_id,
        fetch.offset,
        fetch.logical_len,
        class,
        error,
    )
}

/// Report a failure that no fetch can be blamed for.
///
/// A transfer is numbered from 1, so 0 says the trouble is with the connection
/// itself rather than with anything that was asked for.
fn send_connection_error(
    stream: &mut TcpStream,
    class: ErrorClass,
    error: impl std::fmt::Display,
) -> Result<()> {
    send_error_for(stream, 0, 0, 0, class, error)
}

fn send_error_for(
    stream: &mut TcpStream,
    request_id: u32,
    offset: u64,
    logical_len: u64,
    class: ErrorClass,
    error: impl std::fmt::Display,
) -> Result<()> {
    tracing::debug!(request_id, ?class, "{error}");
    let payload = ErrorPayload::new(class, error.to_string()).encode();
    let header = FrameHeader::error(request_id, offset, logical_len, payload.len() as u64);
    write_frame(stream, &header, &payload)?;
    Ok(())
}

/// Fill `buf`, waking often enough to notice the server stopping.
///
/// Returns `false` for a clean end of stream before any byte of the buffer,
/// which is how a peer says it is done rather than how it fails.
fn read_full(
    stream: &mut TcpStream,
    buf: &mut [u8],
    stop: &AtomicBool,
    idle: Duration,
) -> io::Result<bool> {
    let mut filled = 0;
    let mut since_progress = Instant::now();
    while filled < buf.len() {
        if stop.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "the server is shutting down",
            ));
        }
        match stream.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "the peer hung up {filled} bytes into a {} byte read",
                        buf.len()
                    ),
                ))
            }
            Ok(n) => {
                filled += n;
                since_progress = Instant::now();
            }
            // The read timeout, which is how the loop gets to look at `stop`.
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                if since_progress.elapsed() >= idle {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        format!("nothing arrived for {idle:?}"),
                    ));
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

/// Compare two tokens without leaking where they differ.
fn constant_time_eq(a: &[u8; 16], b: &[u8; 16]) -> bool {
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Hex for logs. Session ids are opaque, so they are shown as bytes.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(adjust: impl FnOnce(&mut ServerConfig)) -> ServerConfig {
        let mut config = ServerConfig {
            data_addr: "127.0.0.1:0".parse().unwrap(),
            ..ServerConfig::default()
        };
        adjust(&mut config);
        config
    }

    #[test]
    fn the_send_buffer_is_applied_to_the_listener() {
        // Below Linux's default net.core.wmem_max, which would cap it.
        let listener = listen(&config(|c| c.tcp.sndbuf = 100_000)).expect("listen");
        let applied = socket2::SockRef::from(&listener)
            .send_buffer_size()
            .unwrap();
        // The kernel may round it up (Linux doubles it).
        assert!(applied >= 100_000, "{applied}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn an_unknown_congestion_algorithm_fails_at_startup() {
        let err = listen(&config(|c| c.tcp.congestion = "no-such-cc".into())).unwrap_err();
        assert!(matches!(err, ServerError::Config(_)), "{err}");
    }

    #[test]
    fn a_restarted_server_can_take_its_port_straight_back() {
        let listener = listen(&config(|_| {})).expect("listen");
        let reuse = socket2::SockRef::from(&listener).reuse_address().unwrap();
        assert!(reuse);
    }
}
