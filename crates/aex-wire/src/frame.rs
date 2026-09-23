//! The 32-byte frame header, and the frames built on it.
//!
//! The header is fixed width so that a receiver can read it in one go and then
//! read the payload straight into the output array. A variable-length header
//! would need either a second read to learn its own length or a look-ahead
//! buffer, and the look-ahead is what destroys the zero-copy receive.
//!
//! ```text
//! offset size field
//!      0    1 frame_type
//!      1    1 codec
//!      2    1 encoding
//!      3    1 flags
//!      4    4 request_id
//!      8    8 offset       position in the logical byte stream
//!     16    8 wire_len     bytes of payload that follow
//!     24    8 logical_len  what those bytes expand to
//! ```
//!
//! `wire_len == logical_len` exactly when the codec is RAW, and only then can
//! the payload be read straight into the output array.

use std::io::{self, IoSlice, Read, Write};

use aex_core::{Codec, Encoding, ErrorClass};

use crate::error::{Result, WireError};

pub const HEADER_LEN: usize = 32;

/// Size of the ticket a `FETCH` carries as its payload.
pub const TICKET_LEN: usize = 16;

/// The capability that authorises one transfer.
pub type Ticket = [u8; TICKET_LEN];

/// What a frame is for.
///
/// 0x04 is deliberately unused: the value is held open for a write path, which
/// would pair a `PUSH` with `FETCH` without disturbing anything here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameType {
    /// Client to server: send me this range. Payload is the ticket.
    Fetch = 0x01,
    /// Server to client: here is a range. Payload is the data.
    Data = 0x02,
    /// Server to client: this fetch failed. Payload is [`ErrorPayload`].
    Error = 0x03,
    /// Either way: are you there.
    Ping = 0x05,
    /// Either way: yes.
    Pong = 0x06,
}

impl FrameType {
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// From the wire. `None` for a type this build does not know, which the
    /// caller reports as a protocol error rather than guessing at.
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(FrameType::Fetch),
            0x02 => Some(FrameType::Data),
            0x03 => Some(FrameType::Error),
            0x05 => Some(FrameType::Ping),
            0x06 => Some(FrameType::Pong),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub frame_type: FrameType,
    pub codec: Codec,
    pub encoding: Encoding,
    /// Reserved, and 0 in this version. Version is agreed in the handshake, so
    /// a connection never sees a flag whose meaning it does not know.
    pub flags: u8,
    /// Which transfer. 0 means the frame is about the connection itself, which
    /// is why transfers are numbered from 1.
    pub request_id: u32,
    /// Where in the transfer's logical byte stream.
    pub offset: u64,
    /// Bytes of payload that follow this header.
    pub wire_len: u64,
    /// What those bytes mean once decoded. Equal to `wire_len` under RAW.
    pub logical_len: u64,
}

impl FrameHeader {
    /// A request for `logical_len` bytes from `offset`.
    pub fn fetch(request_id: u32, offset: u64, logical_len: u64) -> Self {
        FrameHeader {
            frame_type: FrameType::Fetch,
            codec: Codec::Raw,
            encoding: Encoding::Exact,
            flags: 0,
            request_id,
            offset,
            // The payload of a FETCH is its ticket, never the range it asks for.
            wire_len: TICKET_LEN as u64,
            logical_len,
        }
    }

    /// A reply carrying `len` uncompressed bytes from `offset`.
    pub fn data(request_id: u32, offset: u64, len: u64) -> Self {
        FrameHeader {
            frame_type: FrameType::Data,
            codec: Codec::Raw,
            encoding: Encoding::Exact,
            flags: 0,
            request_id,
            offset,
            wire_len: len,
            logical_len: len,
        }
    }

    /// A reply carrying one encoded block.
    ///
    /// `wire_len` bytes follow; they expand to `logical_len` bytes of the
    /// stream, so the receiver reads them somewhere else and expands into
    /// place rather than reading straight into the output array.
    pub fn data_encoded(
        request_id: u32,
        offset: u64,
        codec: Codec,
        encoding: Encoding,
        wire_len: u64,
        logical_len: u64,
    ) -> Self {
        FrameHeader {
            frame_type: FrameType::Data,
            codec,
            encoding,
            flags: 0,
            request_id,
            offset,
            wire_len,
            logical_len,
        }
    }

    /// A failure, named by the fetch that caused it.
    ///
    /// The range is the one that failed, not the one that succeeded, so that
    /// the client can put exactly that chunk back on its queue.
    pub fn error(request_id: u32, offset: u64, logical_len: u64, payload_len: u64) -> Self {
        FrameHeader {
            frame_type: FrameType::Error,
            codec: Codec::Raw,
            encoding: Encoding::Exact,
            flags: 0,
            request_id,
            offset,
            wire_len: payload_len,
            logical_len,
        }
    }

    /// A frame with no payload.
    pub fn bare(frame_type: FrameType) -> Self {
        FrameHeader {
            frame_type,
            codec: Codec::Raw,
            encoding: Encoding::Exact,
            flags: 0,
            request_id: 0,
            offset: 0,
            wire_len: 0,
            logical_len: 0,
        }
    }

    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let mut out = [0u8; HEADER_LEN];
        out[0] = self.frame_type.as_u8();
        out[1] = self.codec.as_u8();
        out[2] = self.encoding.as_u8();
        out[3] = self.flags;
        out[4..8].copy_from_slice(&self.request_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.offset.to_le_bytes());
        out[16..24].copy_from_slice(&self.wire_len.to_le_bytes());
        out[24..32].copy_from_slice(&self.logical_len.to_le_bytes());
        out
    }

    /// Parse a header.
    ///
    /// Only the three tag bytes can be wrong; the rest is whatever the sender
    /// put there, and what counts as a sensible range depends on the transfer
    /// rather than on the frame.
    pub fn decode(bytes: &[u8; HEADER_LEN]) -> Result<Self> {
        let frame_type = FrameType::from_u8(bytes[0]).ok_or_else(|| {
            WireError::Protocol(format!("frame type {:#04x} is not defined", bytes[0]))
        })?;
        let codec = Codec::from_u8(bytes[1])
            .ok_or_else(|| WireError::Protocol(format!("codec {} is not defined", bytes[1])))?;
        let encoding = Encoding::from_u8(bytes[2])
            .ok_or_else(|| WireError::Protocol(format!("encoding {} is not defined", bytes[2])))?;

        Ok(FrameHeader {
            frame_type,
            codec,
            encoding,
            flags: bytes[3],
            request_id: u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes")),
            offset: u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")),
            wire_len: u64::from_le_bytes(bytes[16..24].try_into().expect("8 bytes")),
            logical_len: u64::from_le_bytes(bytes[24..32].try_into().expect("8 bytes")),
        })
    }
}

/// The payload of an `ERROR` frame: a class byte and a diagnostic.
///
/// The class is all a client branches on. The message is for a human reading a
/// log, and must never be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorPayload {
    pub class: ErrorClass,
    pub message: String,
}

impl ErrorPayload {
    pub fn new(class: ErrorClass, message: impl Into<String>) -> Self {
        ErrorPayload {
            class,
            message: message.into(),
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + self.message.len());
        out.push(self.class.as_u8());
        out.extend_from_slice(self.message.as_bytes());
        out
    }

    /// Parse a payload. A message that is not UTF-8 is replaced rather than
    /// refused: losing the diagnostic must not also lose the class it explains.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let (class, message) = bytes
            .split_first()
            .ok_or_else(|| WireError::Protocol("an error frame carries no class".to_string()))?;
        Ok(ErrorPayload {
            class: ErrorClass::from_u8(*class),
            message: String::from_utf8_lossy(message).into_owned(),
        })
    }
}

/// Write a header and its payload in one syscall.
///
/// The two are handed to `writev` together so that a 4 MiB data frame costs one
/// call rather than two, and so that no header can reach the peer without the
/// bytes it describes following it.
pub fn write_frame(w: &mut impl Write, header: &FrameHeader, payload: &[u8]) -> io::Result<()> {
    debug_assert_eq!(header.wire_len, payload.len() as u64);

    let encoded = header.encode();
    let mut slices = [IoSlice::new(&encoded), IoSlice::new(payload)];
    let mut remaining = &mut slices[..];
    while !remaining.is_empty() {
        match w.write_vectored(remaining) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => IoSlice::advance_slices(&mut remaining, n),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

pub fn read_frame_header(r: &mut impl Read) -> Result<FrameHeader> {
    let mut bytes = [0u8; HEADER_LEN];
    r.read_exact(&mut bytes)?;
    FrameHeader::decode(&bytes)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// Every frame type, for sweeps.
    const ALL_TYPES: [FrameType; 5] = [
        FrameType::Fetch,
        FrameType::Data,
        FrameType::Error,
        FrameType::Ping,
        FrameType::Pong,
    ];

    const ALL_ENCODINGS: [Encoding; 3] =
        [Encoding::Exact, Encoding::DtypeCast, Encoding::ErrorBound];

    const ALL_CODECS: [Codec; 3] = [Codec::Raw, Codec::Sz, Codec::Zfp];

    #[test]
    fn a_header_is_thirty_two_bytes_with_the_documented_layout() {
        let header = FrameHeader {
            // Three different values, so a swapped pair of tag bytes shows up.
            frame_type: FrameType::Error,
            codec: Codec::Zfp,
            encoding: Encoding::DtypeCast,
            flags: 0,
            request_id: 0x0a0b0c0d,
            offset: 0x1122334455667788,
            wire_len: 0x0102030405060708,
            logical_len: 0xf0e0d0c0b0a09080,
        };
        let bytes = header.encode();
        assert_eq!(bytes.len(), HEADER_LEN);

        assert_eq!(bytes[0], 0x03);
        assert_eq!(bytes[1], 0x02);
        assert_eq!(bytes[2], 0x01);
        assert_eq!(bytes[3], 0x00);
        // Little-endian, so the low byte comes first.
        assert_eq!(&bytes[4..8], &[0x0d, 0x0c, 0x0b, 0x0a]);
        assert_eq!(&bytes[8..16], &0x1122334455667788u64.to_le_bytes());
        assert_eq!(&bytes[16..24], &0x0102030405060708u64.to_le_bytes());
        assert_eq!(&bytes[24..32], &0xf0e0d0c0b0a09080u64.to_le_bytes());
    }

    #[test]
    fn frame_types_keep_their_wire_values() {
        assert_eq!(FrameType::Fetch.as_u8(), 0x01);
        assert_eq!(FrameType::Data.as_u8(), 0x02);
        assert_eq!(FrameType::Error.as_u8(), 0x03);
        assert_eq!(FrameType::Ping.as_u8(), 0x05);
        assert_eq!(FrameType::Pong.as_u8(), 0x06);
        for frame_type in ALL_TYPES {
            assert_eq!(FrameType::from_u8(frame_type.as_u8()), Some(frame_type));
        }
        // 0x04 is held open for a write path.
        assert_eq!(FrameType::from_u8(0x04), None);
        assert_eq!(FrameType::from_u8(0x00), None);
        assert_eq!(FrameType::from_u8(0xff), None);
    }

    #[test]
    fn the_constructors_fill_in_what_each_frame_means() {
        let fetch = FrameHeader::fetch(7, 4096, 1 << 20);
        assert_eq!(fetch.frame_type, FrameType::Fetch);
        // A fetch carries its ticket, not the bytes it is asking for.
        assert_eq!(fetch.wire_len, TICKET_LEN as u64);
        assert_eq!(fetch.logical_len, 1 << 20);

        let data = FrameHeader::data(7, 4096, 1 << 20);
        // Uncompressed, so the receiver can read it straight into the array.
        assert_eq!(data.wire_len, data.logical_len);
        assert_eq!(data.codec, Codec::Raw);

        // An error names the fetch that failed, so that just that chunk can be
        // put back on the queue.
        let error = FrameHeader::error(7, 4096, 1 << 20, 13);
        assert_eq!(error.offset, 4096);
        assert_eq!(error.logical_len, 1 << 20);
        assert_eq!(error.wire_len, 13);

        let ping = FrameHeader::bare(FrameType::Ping);
        assert_eq!(ping.wire_len, 0);
        assert_eq!(ping.request_id, 0);
    }

    #[test]
    fn a_header_this_build_cannot_read_is_a_protocol_error() {
        let mut bytes = FrameHeader::data(1, 0, 8).encode();

        bytes[0] = 0x04;
        assert!(matches!(
            FrameHeader::decode(&bytes),
            Err(WireError::Protocol(_))
        ));
        bytes[0] = FrameType::Data.as_u8();

        bytes[1] = 9;
        let err = FrameHeader::decode(&bytes).unwrap_err();
        assert!(err.to_string().contains("codec 9"), "{err}");
        bytes[1] = Codec::Raw.as_u8();

        bytes[2] = 42;
        assert!(FrameHeader::decode(&bytes).is_err());
    }

    #[test]
    fn an_error_payload_survives_a_roundtrip() {
        let payload = ErrorPayload::new(ErrorClass::Transient, "read failed: input/output error");
        let bytes = payload.encode();
        assert_eq!(bytes[0], ErrorClass::Transient.as_u8());
        assert_eq!(ErrorPayload::decode(&bytes).unwrap(), payload);

        // A class with no message is still a usable error.
        let bare = ErrorPayload::new(ErrorClass::Plan, "");
        assert_eq!(ErrorPayload::decode(&bare.encode()).unwrap(), bare);
        // An empty payload is not.
        assert!(ErrorPayload::decode(&[]).is_err());

        // A class this build does not know fails safe, and the message it came
        // with is still readable.
        let unknown = ErrorPayload::decode(&[200, b'h', b'i']).unwrap();
        assert_eq!(unknown.class, ErrorClass::Permanent);
        assert_eq!(unknown.message, "hi");
    }

    #[test]
    fn a_frame_is_written_as_its_header_followed_by_its_payload() {
        let header = FrameHeader::data(3, 64, 4);
        let mut out = Vec::new();
        write_frame(&mut out, &header, b"abcd").expect("write");

        assert_eq!(out.len(), HEADER_LEN + 4);
        assert_eq!(&out[..HEADER_LEN], &header.encode());
        assert_eq!(&out[HEADER_LEN..], b"abcd");

        let mut reader = out.as_slice();
        assert_eq!(read_frame_header(&mut reader).unwrap(), header);
        assert_eq!(reader, b"abcd");
    }

    /// A writer that accepts a few bytes at a time, as a socket under pressure
    /// does. Vec never writes short, so a partial write has to be staged.
    struct TrickleWriter {
        written: Vec<u8>,
        per_call: usize,
    }

    impl Write for TrickleWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let n = buf.len().min(self.per_call);
            self.written.extend_from_slice(&buf[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_partial_write_is_resumed_where_it_stopped() {
        // The default write_vectored forwards to write, so this also covers a
        // header that is only half sent.
        for per_call in [1, 7, 32, 33] {
            let header = FrameHeader::data(1, 0, 40);
            let payload: Vec<u8> = (0..40u8).collect();
            let mut writer = TrickleWriter {
                written: Vec::new(),
                per_call,
            };
            write_frame(&mut writer, &header, &payload).expect("write");

            assert_eq!(
                writer.written.len(),
                HEADER_LEN + 40,
                "{per_call} at a time"
            );
            assert_eq!(&writer.written[..HEADER_LEN], &header.encode());
            assert_eq!(&writer.written[HEADER_LEN..], &payload[..]);
        }
    }

    #[test]
    fn a_truncated_header_is_reported_rather_than_read_as_zeros() {
        let bytes = FrameHeader::data(1, 0, 8).encode();
        let mut short = &bytes[..HEADER_LEN - 1];
        assert!(matches!(
            read_frame_header(&mut short),
            Err(WireError::Io(_))
        ));
    }

    proptest! {
        /// The property that matters most here: whatever one side encodes, the
        /// other reads back unchanged. Format bugs hide in the fields nobody
        /// thought to write a case for.
        #[test]
        fn any_header_survives_a_roundtrip(
            type_index in 0usize..ALL_TYPES.len(),
            codec_index in 0usize..ALL_CODECS.len(),
            encoding_index in 0usize..ALL_ENCODINGS.len(),
            flags in any::<u8>(),
            request_id in any::<u32>(),
            offset in any::<u64>(),
            wire_len in any::<u64>(),
            logical_len in any::<u64>(),
        ) {
            let header = FrameHeader {
                frame_type: ALL_TYPES[type_index],
                codec: ALL_CODECS[codec_index],
                encoding: ALL_ENCODINGS[encoding_index],
                flags,
                request_id,
                offset,
                wire_len,
                logical_len,
            };
            prop_assert_eq!(FrameHeader::decode(&header.encode()).unwrap(), header);
        }

        #[test]
        fn any_error_payload_survives_a_roundtrip(class in 0u8..7, message in ".*") {
            let payload = ErrorPayload::new(ErrorClass::from_u8(class), message);
            prop_assert_eq!(ErrorPayload::decode(&payload.encode()).unwrap(), payload);
        }
    }
}
