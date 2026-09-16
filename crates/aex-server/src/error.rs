//! Server errors and their gRPC status codes.
//!
//! Every error a client can see maps onto one of the six error classes, and
//! each class onto exactly one gRPC code. The client turns the code back into a
//! class to decide whether to retry, so a code outside that set would tell it
//! nothing: `RESOURCE_EXHAUSTED`, for instance, is reported as `UNAVAILABLE`,
//! the code for a condition that may clear on its own.

use aex_core::{AexError, ErrorClass};
use tonic::{Code, Status};

pub type Result<T> = std::result::Result<T, ServerError>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ServerError {
    /// Bad configuration. Only ever surfaces at startup.
    #[error("configuration error: {0}")]
    Config(String),

    /// Unknown or expired session.
    #[error("{0}")]
    Auth(String),

    /// A request the server understood but will not serve.
    #[error("{0}")]
    BadRequest(String),

    /// An unknown or expired transfer plan. The client re-prepares on this,
    /// and on nothing else, so it must not be conflated with a bad request.
    #[error("{0}")]
    NoSuchPlan(String),

    /// A path outside every configured root.
    #[error("{0}")]
    PathNotAllowed(String),

    /// A limit that will not be exceeded forever: sessions, streams, transfers.
    #[error("{0}")]
    Exhausted(String),

    /// The OS entropy source failed, so no session id could be minted.
    #[error("cannot generate random bytes: {0}")]
    Random(#[from] getrandom::Error),

    #[error(transparent)]
    Core(#[from] AexError),

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),
}

impl ServerError {
    /// Map this error onto its wire class.
    pub fn class(&self) -> ErrorClass {
        match self {
            ServerError::Auth(_) => ErrorClass::Auth,
            ServerError::NoSuchPlan(_) => ErrorClass::Plan,
            ServerError::BadRequest(_) | ServerError::PathNotAllowed(_) => ErrorClass::Request,
            ServerError::Exhausted(_) => ErrorClass::Transient,
            ServerError::Core(e) => e.class(),
            ServerError::Io(e) => AexError::Io(std::io::Error::from(e.kind())).class(),
            // Nothing a client did caused these, and no retry helps.
            ServerError::Config(_) | ServerError::Random(_) | ServerError::Transport(_) => {
                ErrorClass::Permanent
            }
        }
    }
}

impl From<ServerError> for Status {
    fn from(err: ServerError) -> Status {
        let message = err.to_string();
        let code = match &err {
            // Within REQUEST, the finer code says which kind of bad request it
            // was. All three map back to REQUEST on the client.
            ServerError::PathNotAllowed(_) => Code::PermissionDenied,
            ServerError::Core(AexError::NotFound(_)) => Code::NotFound,
            ServerError::Core(AexError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                Code::NotFound
            }
            ServerError::Core(AexError::Io(e))
                if e.kind() == std::io::ErrorKind::PermissionDenied =>
            {
                Code::PermissionDenied
            }
            other => match other.class() {
                ErrorClass::Ok | ErrorClass::Protocol => Code::Internal,
                ErrorClass::Auth => Code::Unauthenticated,
                ErrorClass::Plan => Code::FailedPrecondition,
                ErrorClass::Request => Code::InvalidArgument,
                ErrorClass::Transient => Code::Unavailable,
                ErrorClass::Permanent => Code::DataLoss,
            },
        };
        Status::new(code, message)
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    #[test]
    fn each_class_maps_to_the_code_the_client_expects() {
        let code = |err: ServerError| Status::from(err).code();
        assert_eq!(code(ServerError::Auth("x".into())), Code::Unauthenticated);
        // A plan the client can reissue, and nothing else, gets this code.
        assert_eq!(
            code(ServerError::NoSuchPlan("x".into())),
            Code::FailedPrecondition
        );
        assert_eq!(
            code(ServerError::BadRequest("x".into())),
            Code::InvalidArgument
        );
        assert_eq!(
            code(ServerError::PathNotAllowed("x".into())),
            Code::PermissionDenied
        );
        // Too many sessions is a condition that clears, not a client mistake.
        assert_eq!(code(ServerError::Exhausted("x".into())), Code::Unavailable);
        assert_eq!(code(AexError::NotFound("x".into()).into()), Code::NotFound);
        assert_eq!(
            code(AexError::UnsupportedNpy("x".into()).into()),
            Code::InvalidArgument
        );
        assert_eq!(
            code(AexError::MalformedNpy("x".into()).into()),
            Code::DataLoss
        );
        assert_eq!(
            code(AexError::Io(io::Error::from(io::ErrorKind::NotFound)).into()),
            Code::NotFound
        );
        assert_eq!(
            code(AexError::Io(io::Error::from(io::ErrorKind::PermissionDenied)).into()),
            Code::PermissionDenied
        );
        // A real I/O failure is worth retrying.
        assert_eq!(
            code(AexError::Io(io::Error::from(io::ErrorKind::Interrupted)).into()),
            Code::Unavailable
        );
    }

    #[test]
    fn the_status_message_survives() {
        let status = Status::from(ServerError::BadRequest("no such format 'h5'".into()));
        assert_eq!(status.message(), "no such format 'h5'");
    }
}
