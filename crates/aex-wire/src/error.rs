//! Errors from the wire format.
//!
//! Each one carries the [`ErrorClass`] it would travel as in an `ERROR` frame,
//! because that class is the whole of what a peer needs in order to decide what
//! to do next.

use std::io;

use aex_core::ErrorClass;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, WireError>;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WireError {
    /// Something on the wire was not what the protocol says. Means a bug in an
    /// implementation, or a peer that is not speaking AEX at all.
    #[error("protocol error: {0}")]
    Protocol(String),

    /// A frame named a range outside the buffer it would have been written to.
    #[error("range [{offset}, {}) does not fit a {capacity} byte buffer",
            .offset.saturating_add(*.len))]
    OutOfRange {
        offset: u64,
        len: u64,
        capacity: u64,
    },

    /// Two receivers claimed overlapping parts of one buffer. Means the chunk
    /// allocation is wrong, which would otherwise corrupt the result silently.
    #[error("range [{offset}, {}) overlaps one already being written",
            .offset.saturating_add(*.len))]
    Overlap { offset: u64, len: u64 },

    #[error("i/o error: {0}")]
    Io(#[from] io::Error),
}

impl WireError {
    /// The class this error travels as.
    pub fn class(&self) -> ErrorClass {
        match self {
            WireError::Protocol(_) => ErrorClass::Protocol,
            // Both mean the peer asked for something that does not fit what it
            // was told, which is a bad request rather than a broken frame.
            WireError::OutOfRange { .. } | WireError::Overlap { .. } => ErrorClass::Request,
            // A connection that broke or stalled: the transfer is split into
            // chunks precisely so that only the lost one has to be redone.
            WireError::Io(_) => ErrorClass::Transient,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_carry_the_class_they_travel_as() {
        assert_eq!(
            WireError::Protocol("x".into()).class(),
            ErrorClass::Protocol
        );
        assert_eq!(
            WireError::OutOfRange {
                offset: 0,
                len: 1,
                capacity: 0
            }
            .class(),
            ErrorClass::Request
        );
        assert_eq!(
            WireError::Io(io::Error::from(io::ErrorKind::ConnectionReset)).class(),
            ErrorClass::Transient
        );
    }

    #[test]
    fn the_message_says_which_range_was_wrong() {
        let err = WireError::OutOfRange {
            offset: 96,
            len: 32,
            capacity: 100,
        };
        assert_eq!(
            err.to_string(),
            "range [96, 128) does not fit a 100 byte buffer"
        );
        // An overflowing range still renders rather than panicking.
        let err = WireError::Overlap {
            offset: u64::MAX,
            len: 8,
        };
        assert!(err.to_string().contains(&u64::MAX.to_string()));
    }
}
