//! Opening a data connection.
//!
//! One exchange per connection, before any frame: the client says which session
//! it belongs to and proves it with the token the control plane gave it, and
//! the server either accepts or says which class of thing was wrong and hangs
//! up.
//!
//! The token is all the data plane checks. Who the user is and what they may
//! open is the control plane's business; by the time a connection exists, that
//! has already been decided. Per-transfer authorisation is the ticket in each
//! `FETCH`, not anything here.
//!
//! Nothing is encrypted. That is the trusted-environment assumption of this
//! release, and `flags` holds a bit open for negotiating TLS later.

use aex_core::ErrorClass;

use crate::error::{Result, WireError};

/// First eight bytes of both messages. The trailing byte is the wire format
/// generation, which changes only if the framing itself is redesigned.
pub const MAGIC: [u8; 8] = *b"AEXDATA\x01";

/// The data plane protocol version this build speaks.
pub const PROTOCOL_VERSION: u16 = 2;

pub const HELLO_LEN: usize = 48;

pub const READY_LEN: usize = 16;

/// Client to server, once per connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hello {
    pub version: u16,
    /// Reserved, and 0 in this version.
    pub flags: u32,
    pub session_id: [u8; 16],
    /// Never logged: it is the capability the whole session rests on.
    pub session_token: [u8; 16],
}

impl Hello {
    pub fn new(session_id: [u8; 16], session_token: [u8; 16]) -> Self {
        Hello {
            version: PROTOCOL_VERSION,
            flags: 0,
            session_id,
            session_token,
        }
    }

    pub fn encode(&self) -> [u8; HELLO_LEN] {
        let mut out = [0u8; HELLO_LEN];
        out[0..8].copy_from_slice(&MAGIC);
        out[8..10].copy_from_slice(&self.version.to_le_bytes());
        // 10..12 is reserved and stays zero.
        out[12..16].copy_from_slice(&self.flags.to_le_bytes());
        out[16..32].copy_from_slice(&self.session_id);
        out[32..48].copy_from_slice(&self.session_token);
        out
    }

    /// Parse a `HELLO`. The version is carried through rather than checked: the
    /// server has to answer a version it refuses, which it cannot do from here.
    pub fn decode(bytes: &[u8; HELLO_LEN]) -> Result<Self> {
        check_magic(&bytes[0..8])?;
        Ok(Hello {
            version: u16::from_le_bytes(bytes[8..10].try_into().expect("2 bytes")),
            flags: u32::from_le_bytes(bytes[12..16].try_into().expect("4 bytes")),
            session_id: bytes[16..32].try_into().expect("16 bytes"),
            session_token: bytes[32..48].try_into().expect("16 bytes"),
        })
    }
}

/// Server to client, in reply to a `HELLO`.
///
/// Sixteen bytes carry no message, which is enough because the client's next
/// move is decided by the class alone. The reason in full goes to the server
/// log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ready {
    /// [`ErrorClass::Ok`] to accept. Anything else and the server hangs up.
    pub status: ErrorClass,
    /// The version the server settled on.
    pub version: u16,
    /// Reserved, and 0 in this version.
    pub flags: u32,
}

impl Ready {
    pub fn accepted() -> Self {
        Ready {
            status: ErrorClass::Ok,
            version: PROTOCOL_VERSION,
            flags: 0,
        }
    }

    pub fn refused(status: ErrorClass) -> Self {
        Ready {
            status,
            version: PROTOCOL_VERSION,
            flags: 0,
        }
    }

    pub fn is_accepted(&self) -> bool {
        self.status == ErrorClass::Ok
    }

    pub fn encode(&self) -> [u8; READY_LEN] {
        let mut out = [0u8; READY_LEN];
        out[0..8].copy_from_slice(&MAGIC);
        out[8..10].copy_from_slice(&(self.status.as_u8() as u16).to_le_bytes());
        out[10..12].copy_from_slice(&self.version.to_le_bytes());
        out[12..16].copy_from_slice(&self.flags.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8; READY_LEN]) -> Result<Self> {
        check_magic(&bytes[0..8])?;
        let status = u16::from_le_bytes(bytes[8..10].try_into().expect("2 bytes"));
        let status = u8::try_from(status)
            .map_err(|_| WireError::Protocol(format!("status {status} is not an error class")))?;
        Ok(Ready {
            // An unknown class fails safe: a client must not retry something a
            // newer server meant as final.
            status: ErrorClass::from_u8(status),
            version: u16::from_le_bytes(bytes[10..12].try_into().expect("2 bytes")),
            flags: u32::from_le_bytes(bytes[12..16].try_into().expect("4 bytes")),
        })
    }
}

fn check_magic(bytes: &[u8]) -> Result<()> {
    if bytes != MAGIC {
        return Err(WireError::Protocol(format!(
            "expected the data plane magic {:?}, got {:?}",
            String::from_utf8_lossy(&MAGIC),
            String::from_utf8_lossy(bytes)
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hello_is_forty_eight_bytes_with_the_documented_layout() {
        let hello = Hello::new([0xaa; 16], [0xbb; 16]);
        let bytes = hello.encode();

        assert_eq!(bytes.len(), HELLO_LEN);
        assert_eq!(&bytes[0..8], b"AEXDATA\x01");
        assert_eq!(&bytes[8..10], &PROTOCOL_VERSION.to_le_bytes());
        // The two reserved bytes stay zero so that a later version can use them.
        assert_eq!(&bytes[10..12], &[0, 0]);
        assert_eq!(&bytes[12..16], &[0, 0, 0, 0]);
        assert_eq!(&bytes[16..32], &[0xaa; 16]);
        assert_eq!(&bytes[32..48], &[0xbb; 16]);

        assert_eq!(Hello::decode(&bytes).unwrap(), hello);
    }

    #[test]
    fn a_ready_is_sixteen_bytes_with_the_documented_layout() {
        let ready = Ready::accepted();
        let bytes = ready.encode();

        assert_eq!(bytes.len(), READY_LEN);
        assert_eq!(&bytes[0..8], &MAGIC);
        assert_eq!(&bytes[8..10], &0u16.to_le_bytes());
        assert_eq!(&bytes[10..12], &PROTOCOL_VERSION.to_le_bytes());
        assert_eq!(Ready::decode(&bytes).unwrap(), ready);
        assert!(ready.is_accepted());

        for class in [
            ErrorClass::Protocol,
            ErrorClass::Auth,
            ErrorClass::Transient,
        ] {
            let refused = Ready::refused(class);
            let decoded = Ready::decode(&refused.encode()).unwrap();
            assert_eq!(decoded, refused);
            assert!(!decoded.is_accepted());
            assert_eq!(decoded.status, class);
        }
    }

    #[test]
    fn something_that_is_not_aex_is_refused_at_the_first_eight_bytes() {
        let mut bytes = Hello::new([0; 16], [0; 16]).encode();
        bytes[0..8].copy_from_slice(b"GET / HT");
        let err = Hello::decode(&bytes).unwrap_err();
        assert!(matches!(err, WireError::Protocol(_)), "{err}");
        assert_eq!(err.class(), ErrorClass::Protocol);

        // A generation of the format this build does not speak is refused the
        // same way, since the generation is part of the magic.
        let mut bytes = Ready::accepted().encode();
        bytes[7] = 0x02;
        assert!(Ready::decode(&bytes).is_err());
    }

    #[test]
    fn a_version_the_server_refuses_still_arrives_intact() {
        // The check belongs to the server, which has to answer before hanging
        // up; decode only has to carry the number to it.
        let mut hello = Hello::new([1; 16], [2; 16]);
        hello.version = 99;
        assert_eq!(Hello::decode(&hello.encode()).unwrap().version, 99);
    }

    #[test]
    fn an_unknown_status_fails_safe() {
        let mut bytes = Ready::accepted().encode();
        bytes[8..10].copy_from_slice(&200u16.to_le_bytes());
        assert_eq!(
            Ready::decode(&bytes).unwrap().status,
            ErrorClass::Permanent,
            "a class this build does not know must not be retried"
        );

        // A status that is not even a byte means the two sides disagree about
        // the layout itself.
        bytes[8..10].copy_from_slice(&1000u16.to_le_bytes());
        assert!(matches!(Ready::decode(&bytes), Err(WireError::Protocol(_))));
    }
}
