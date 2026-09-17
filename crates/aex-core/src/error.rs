//! Errors and their classification.
//!
//! The gRPC control plane, the data plane's `ERROR` frame and the Python
//! exception hierarchy all share [`ErrorClass`]. Client recovery is decided by
//! the class alone, so there are no finer-grained codes: the specific cause
//! lives in the diagnostic message and the server log.

use std::io;

use thiserror::Error;

/// Error classification. One byte on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ErrorClass {
    /// Success. Only used where one value carries both outcomes, e.g. `READY`.
    Ok = 0,
    /// Malformed frame or handshake. Means a bug in an implementation.
    Protocol = 1,
    /// Bad or expired session, token or ticket.
    Auth = 2,
    /// Transfer plan is unknown or expired.
    Plan = 3,
    /// Bad request: out of range, over a limit, unsupported format.
    Request = 4,
    /// Temporary server-side condition. Worth retrying.
    Transient = 5,
    /// Permanent server-side condition: corrupt file, internal error.
    Permanent = 6,
}

impl ErrorClass {
    /// The wire representation.
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Classify a value read off the wire.
    ///
    /// Unknown values become `Permanent` so that an old client never retries
    /// something a newer server meant as fatal.
    pub const fn from_u8(v: u8) -> Self {
        match v {
            0 => ErrorClass::Ok,
            1 => ErrorClass::Protocol,
            2 => ErrorClass::Auth,
            3 => ErrorClass::Plan,
            4 => ErrorClass::Request,
            5 => ErrorClass::Transient,
            _ => ErrorClass::Permanent,
        }
    }
}

/// Result type for this crate.
pub type Result<T> = std::result::Result<T, AexError>;

/// Errors raised by the core layer.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AexError {
    /// A numpy dtype AEX cannot transfer: structured, datetime, object, big-endian.
    #[error("unsupported dtype: {0}")]
    UnsupportedDType(String),

    /// A well-formed `.npy` that AEX cannot serve, such as fortran order.
    #[error("unsupported npy file: {0}")]
    UnsupportedNpy(String),

    /// The `.npy` is corrupt, or its header disagrees with the file itself.
    #[error("malformed npy file: {0}")]
    MalformedNpy(String),

    /// A readable HDF5 file holding something AEX cannot serve, such as a
    /// compressed or virtual dataset.
    #[error("unsupported HDF5 file: {0}")]
    UnsupportedHdf5(String),

    /// libhdf5 failed, or the file disagrees with its own metadata.
    #[error("malformed HDF5 file: {0}")]
    MalformedHdf5(String),

    /// A selection numpy would reject too: an index off the end, more indices
    /// than the array has axes, a zero step.
    #[error("invalid selection: {0}")]
    BadSelection(String),

    /// A selection this release cannot resolve, such as one whose bytes are not
    /// one contiguous run of the source.
    #[error("unsupported selection: {0}")]
    UnsupportedSelection(String),

    /// No item at this path in the file.
    #[error("no such item: {0}")]
    NotFound(String),

    /// A group operation was given the path of a dataset.
    #[error("{0:?} is a dataset, not a group")]
    NotAGroup(String),

    /// Access outside the logical byte stream.
    #[error("range [{offset}, {}) is out of bounds for a {total} byte logical stream",
            .offset.saturating_add(*.len))]
    OutOfRange { offset: u64, len: u64, total: u64 },

    #[error("i/o error: {0}")]
    Io(#[from] io::Error),
}

#[cfg(feature = "hdf5")]
impl From<hdf5::Error> for AexError {
    fn from(e: hdf5::Error) -> Self {
        AexError::MalformedHdf5(e.to_string())
    }
}

impl AexError {
    /// Map this error onto its wire class.
    pub fn class(&self) -> ErrorClass {
        match self {
            AexError::UnsupportedDType(_)
            | AexError::UnsupportedNpy(_)
            | AexError::UnsupportedHdf5(_)
            | AexError::BadSelection(_)
            | AexError::UnsupportedSelection(_)
            | AexError::NotFound(_)
            | AexError::NotAGroup(_)
            | AexError::OutOfRange { .. } => ErrorClass::Request,
            AexError::MalformedNpy(_) | AexError::MalformedHdf5(_) => ErrorClass::Permanent,
            AexError::Io(e) => match e.kind() {
                // A missing or unreadable file is the requester's problem.
                io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied => ErrorClass::Request,
                // Unparseable npy lands here. Retrying will not help.
                io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof => ErrorClass::Permanent,
                // A real I/O failure or resource pressure. Worth retrying.
                _ => ErrorClass::Transient,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_class_survives_a_wire_roundtrip() {
        for class in [
            ErrorClass::Ok,
            ErrorClass::Protocol,
            ErrorClass::Auth,
            ErrorClass::Plan,
            ErrorClass::Request,
            ErrorClass::Transient,
            ErrorClass::Permanent,
        ] {
            assert_eq!(ErrorClass::from_u8(class.as_u8()), class);
        }
        // Pin the wire values.
        assert_eq!(ErrorClass::Ok.as_u8(), 0);
        assert_eq!(ErrorClass::Permanent.as_u8(), 6);
        // Unknown values fail safe.
        assert_eq!(ErrorClass::from_u8(7), ErrorClass::Permanent);
        assert_eq!(ErrorClass::from_u8(255), ErrorClass::Permanent);
    }

    #[test]
    fn io_errors_are_classified_by_kind() {
        let classify = |kind| AexError::Io(io::Error::from(kind)).class();
        assert_eq!(classify(io::ErrorKind::NotFound), ErrorClass::Request);
        assert_eq!(
            classify(io::ErrorKind::PermissionDenied),
            ErrorClass::Request
        );
        // Corrupt or non-npy files do not get better on retry.
        assert_eq!(classify(io::ErrorKind::InvalidData), ErrorClass::Permanent);
        assert_eq!(
            classify(io::ErrorKind::UnexpectedEof),
            ErrorClass::Permanent
        );
        // A real I/O failure is worth retrying.
        assert_eq!(classify(io::ErrorKind::Interrupted), ErrorClass::Transient);
    }

    #[test]
    fn domain_errors_are_classified() {
        assert_eq!(
            AexError::UnsupportedDType("x".into()).class(),
            ErrorClass::Request
        );
        assert_eq!(
            AexError::UnsupportedNpy("x".into()).class(),
            ErrorClass::Request
        );
        // A selection a client could fix by asking for something else.
        assert_eq!(
            AexError::BadSelection("x".into()).class(),
            ErrorClass::Request
        );
        assert_eq!(
            AexError::UnsupportedSelection("x".into()).class(),
            ErrorClass::Request
        );
        assert_eq!(
            AexError::OutOfRange {
                offset: 1,
                len: 2,
                total: 2
            }
            .class(),
            ErrorClass::Request
        );
        assert_eq!(AexError::NotFound("x".into()).class(), ErrorClass::Request);
        assert_eq!(AexError::NotAGroup("x".into()).class(), ErrorClass::Request);
        assert_eq!(
            AexError::MalformedNpy("x".into()).class(),
            ErrorClass::Permanent
        );
        assert_eq!(
            AexError::UnsupportedHdf5("x".into()).class(),
            ErrorClass::Request
        );
        assert_eq!(
            AexError::MalformedHdf5("x".into()).class(),
            ErrorClass::Permanent
        );
    }
}
