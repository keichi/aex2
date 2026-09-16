//! Client errors.
//!
//! A server error arrives as a gRPC status, whose code the client turns back
//! into the error class the server assigned it. Recovery is decided by the
//! class: only `Transient` is worth retrying.

use aex_core::ErrorClass;
use tonic::{Code, Status};

pub type Result<T> = std::result::Result<T, ClientError>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ClientError {
    /// The server refused the request.
    #[error("{message}")]
    Server {
        class: ErrorClass,
        code: Code,
        message: String,
    },

    /// The connection could not be made or did not survive.
    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    /// A reply that does not make sense. Means a bug on one side.
    #[error("malformed reply: {0}")]
    Protocol(String),

    /// The caller asked for something the client can tell is wrong.
    #[error("{0}")]
    BadRequest(String),

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

impl ClientError {
    /// The class of a server error; `None` for a local failure.
    pub fn class(&self) -> Option<ErrorClass> {
        match self {
            ClientError::Server { class, .. } => Some(*class),
            _ => None,
        }
    }

    /// Whether trying again could succeed.
    pub fn is_retryable(&self) -> bool {
        self.class() == Some(ErrorClass::Transient)
    }
}

impl From<Status> for ClientError {
    fn from(status: Status) -> Self {
        ClientError::Server {
            class: class_of(status.code()),
            code: status.code(),
            message: status.message().to_string(),
        }
    }
}

/// The class a status code came from.
///
/// The inverse of the server's mapping. Anything outside it is treated as
/// permanent: a server that answers with a code AEX never sends is not one a
/// retry is going to reach.
fn class_of(code: Code) -> ErrorClass {
    match code {
        Code::Ok => ErrorClass::Ok,
        Code::Internal => ErrorClass::Protocol,
        Code::Unauthenticated => ErrorClass::Auth,
        Code::FailedPrecondition => ErrorClass::Plan,
        Code::InvalidArgument | Code::NotFound | Code::PermissionDenied => ErrorClass::Request,
        Code::Unavailable => ErrorClass::Transient,
        _ => ErrorClass::Permanent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_map_back_to_their_class() {
        let class = |code| ClientError::from(Status::new(code, "x")).class().unwrap();
        assert_eq!(class(Code::Unauthenticated), ErrorClass::Auth);
        assert_eq!(class(Code::FailedPrecondition), ErrorClass::Plan);
        assert_eq!(class(Code::InvalidArgument), ErrorClass::Request);
        assert_eq!(class(Code::NotFound), ErrorClass::Request);
        assert_eq!(class(Code::PermissionDenied), ErrorClass::Request);
        assert_eq!(class(Code::Unavailable), ErrorClass::Transient);
        assert_eq!(class(Code::DataLoss), ErrorClass::Permanent);
        assert_eq!(class(Code::Internal), ErrorClass::Protocol);
        // An RPC this server does not serve yet will not start working on a
        // retry either.
        assert_eq!(class(Code::Unimplemented), ErrorClass::Permanent);
    }

    #[test]
    fn only_transient_errors_are_retried() {
        let err = ClientError::from(Status::unavailable("busy"));
        assert!(err.is_retryable());
        assert!(!ClientError::from(Status::not_found("gone")).is_retryable());
        // A local failure has no class and is not retried on its own.
        assert!(!ClientError::Protocol("x".into()).is_retryable());
        assert!(ClientError::Protocol("x".into()).class().is_none());
    }

    #[test]
    fn the_server_message_reaches_the_caller() {
        let err = ClientError::from(Status::invalid_argument("format \"h5\" is not supported"));
        assert_eq!(err.to_string(), "format \"h5\" is not supported");
    }
}
