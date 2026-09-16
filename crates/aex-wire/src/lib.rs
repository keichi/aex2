//! The data plane wire format.
//!
//! Server and client encode and decode frames from this one crate. A format
//! mismatch is the worst bug a two-sided protocol can have, and sharing the
//! definition rules it out by construction rather than by discipline.
//!
//! Three things live here: the handshake that opens a connection
//! ([`handshake`]), the frames that travel on it ([`frame`]), and the receive
//! buffer that lets several connections write into one array at once
//! ([`scatter`]).
//!
//! Everything is little-endian, and every offset is a position in the logical
//! byte stream of a transfer — not in the file, and not in the connection.

pub mod error;
pub mod frame;
pub mod handshake;
pub mod scatter;

pub use error::{Result, WireError};
pub use frame::{
    read_frame_header, write_frame, ErrorPayload, FrameHeader, FrameType, Ticket, HEADER_LEN,
    TICKET_LEN,
};
pub use handshake::{Hello, Ready, HELLO_LEN, MAGIC, PROTOCOL_VERSION, READY_LEN};
pub use scatter::{ScatterBuffer, ScatterSlice};
