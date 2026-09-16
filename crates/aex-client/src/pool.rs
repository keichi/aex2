//! The data connections of a session.
//!
//! One connection for now, established when the session opens and reused for
//! every transfer. Parallel streams, work stealing and credit-based pipelining
//! are what turn this into a pool; the shape they fill in is already here — a
//! transfer loop that borrows a connection and a pool that re-establishes one
//! that broke.
//!
//! The connection carries no state past the handshake, which is what makes
//! re-establishing it enough to recover: a chunk is asked for by its position
//! in the logical byte stream, so asking again for the one that was lost is
//! always correct, whichever connection it goes out on.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Mutex;
use std::time::Duration;

use aex_core::{Codec, ErrorClass};
use aex_wire::{
    read_frame_header, write_frame, ErrorPayload, FrameHeader, FrameType, Hello, Ready, Ticket,
    READY_LEN,
};

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
            .find_map(
                |addr| match TcpStream::connect_timeout(addr, settings.connect_timeout) {
                    Ok(stream) => Some(stream),
                    Err(e) => {
                        last = Some(e);
                        None
                    }
                },
            )
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

    /// Ask for `[offset, offset + dst.len())` and read it straight into `dst`.
    ///
    /// The bytes go from the kernel into the caller's buffer with nothing in
    /// between, which is the whole reason the payload is raw and the header is
    /// fixed width.
    fn fetch_into(
        &mut self,
        request_id: u32,
        ticket: &Ticket,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<()> {
        let len = dst.len() as u64;
        write_frame(
            &mut self.stream,
            &FrameHeader::fetch(request_id, offset, len),
            ticket,
        )?;

        let mut received = 0u64;
        while received < len {
            let header = read_frame_header(&mut self.stream)?;
            match header.frame_type {
                FrameType::Data => {
                    let at = self.receive_data(&header, request_id, offset, dst)?;
                    received += at;
                }
                FrameType::Error => return Err(self.receive_error(&header)?),
                FrameType::Ping => {
                    write_frame(&mut self.stream, &FrameHeader::bare(FrameType::Pong), &[])?;
                }
                FrameType::Pong => {}
                other => {
                    return Err(ClientError::Protocol(format!(
                        "{other:?} is not a frame a server may send"
                    )))
                }
            }
        }
        Ok(())
    }

    /// Read one data frame into its place in `dst`, returning its length.
    ///
    /// A server may answer one fetch with several frames, so the position comes
    /// from the frame rather than from how much has arrived so far.
    fn receive_data(
        &mut self,
        header: &FrameHeader,
        request_id: u32,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<u64> {
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
        Ok(header.logical_len)
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

/// The session's data connections.
pub struct DataPool {
    settings: ConnSettings,
    /// `None` once a connection has broken, until the next fetch rebuilds it.
    conn: Mutex<Option<DataConn>>,
}

impl std::fmt::Debug for DataPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataPool")
            .field("endpoint", &(&self.settings.host, self.settings.port))
            .field("open", &lock(&self.conn).is_some())
            .finish()
    }
}

impl DataPool {
    /// Open the session's connections.
    ///
    /// Eagerly, while the session is being set up, so that the first large
    /// transfer does not pay for a handshake and so that a data plane which
    /// cannot be reached is reported as such rather than much later.
    pub fn connect(settings: ConnSettings) -> Result<Self> {
        let conn = DataConn::connect(&settings)?;
        Ok(DataPool {
            settings,
            conn: Mutex::new(Some(conn)),
        })
    }

    /// Connections currently held.
    pub fn streams(&self) -> u32 {
        1
    }

    /// Fetch one chunk, retrying what is worth retrying.
    ///
    /// Returns how many attempts were wasted, which the transfer statistics
    /// report. A chunk is asked for by its position, so a retry always asks for
    /// exactly the same bytes and can be served by a fresh connection.
    pub fn fetch_into(
        &self,
        request_id: u32,
        ticket: &Ticket,
        offset: u64,
        dst: &mut [u8],
        max_retries: u32,
    ) -> Result<u32> {
        let mut retries = 0;
        loop {
            let error = match self.try_fetch(request_id, ticket, offset, dst) {
                Ok(()) => return Ok(retries),
                Err(e) => e,
            };

            // Anything else is either the request being wrong or the server
            // being sure; neither improves on a second attempt.
            let worth_retrying =
                matches!(
                    error.class(),
                    Some(ErrorClass::Transient) | Some(ErrorClass::Ok) | None
                ) && !matches!(error, ClientError::Protocol(_) | ClientError::BadRequest(_));
            if !worth_retrying || retries >= max_retries {
                return Err(error);
            }
            retries += 1;
        }
    }

    fn try_fetch(
        &self,
        request_id: u32,
        ticket: &Ticket,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<()> {
        let mut held = lock(&self.conn);
        let conn = match held.as_mut() {
            Some(conn) => conn,
            None => held.insert(DataConn::connect(&self.settings)?),
        };

        let result = conn.fetch_into(request_id, ticket, offset, dst);
        if let Err(ClientError::Io(_)) = &result {
            // The connection is in an unknown state: the next fetch gets a
            // fresh one rather than resuming mid-frame.
            *held = None;
        }
        result
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
