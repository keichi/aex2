//! Sessions and the files they hold open.
//!
//! A session is what a client authenticates with, what its data connections
//! will present a token for, and what bounds the server's memory: every file
//! handle and, from M2, every transfer plan hangs off one.
//!
//! Idle sessions are collected two ways. A request that names an expired
//! session fails on the spot, and a sweep drops the ones nobody asks about, so
//! that a client which simply vanished does not pin its files open forever.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use aex_core::ArrayFile;
use dashmap::DashMap;

use crate::config::{ServerConfig, DEFAULT_STREAMS};
use crate::error::{Result, ServerError};

/// A session identifier: 16 bytes, as on the wire.
pub type SessionId = [u8; 16];

/// Authenticates a data connection as belonging to a session.
pub type SessionToken = [u8; 16];

/// The files one session has open.
///
/// Handles are integers rather than v1's UUID strings: they are hashed and
/// compared on every request, and neither should allocate.
pub struct FileRegistry {
    files: DashMap<u64, Arc<dyn ArrayFile>>,
    next_handle: AtomicU64,
}

impl std::fmt::Debug for FileRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileRegistry")
            .field("open", &self.files.len())
            .finish()
    }
}

impl FileRegistry {
    fn new() -> Self {
        FileRegistry {
            files: DashMap::new(),
            // Start at 1: a zero handle is then always a client-side mistake
            // rather than a valid reference to whatever opened first.
            next_handle: AtomicU64::new(1),
        }
    }

    /// Register an open file and return its handle.
    pub fn insert(&self, file: Arc<dyn ArrayFile>) -> u64 {
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.files.insert(handle, file);
        handle
    }

    /// The file behind a handle.
    pub fn get(&self, handle: u64) -> Result<Arc<dyn ArrayFile>> {
        self.files
            .get(&handle)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| ServerError::BadRequest(format!("unknown file handle {handle}")))
    }

    /// Forget a handle. Closing one twice is an error: it usually means the
    /// client is using a handle it already dropped.
    pub fn remove(&self, handle: u64) -> Result<()> {
        self.files
            .remove(&handle)
            .map(|_| ())
            .ok_or_else(|| ServerError::BadRequest(format!("unknown file handle {handle}")))
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

/// One client's session.
#[derive(Debug)]
pub struct Session {
    id: SessionId,
    token: SessionToken,
    granted_streams: u32,
    client_name: String,
    files: FileRegistry,
    /// Milliseconds since the registry's epoch, at the last request.
    last_seen_ms: AtomicU64,
}

impl Session {
    pub fn id(&self) -> &SessionId {
        &self.id
    }

    /// The token a data connection has to present. Never logged.
    pub fn token(&self) -> &SessionToken {
        &self.token
    }

    pub fn granted_streams(&self) -> u32 {
        self.granted_streams
    }

    pub fn client_name(&self) -> &str {
        &self.client_name
    }

    pub fn files(&self) -> &FileRegistry {
        &self.files
    }

    fn touch(&self, now_ms: u64) {
        self.last_seen_ms.store(now_ms, Ordering::Relaxed);
    }

    fn is_expired(&self, now_ms: u64, timeout_ms: u64) -> bool {
        now_ms.saturating_sub(self.last_seen_ms.load(Ordering::Relaxed)) > timeout_ms
    }
}

/// Every live session.
#[derive(Debug)]
pub struct SessionRegistry {
    sessions: DashMap<SessionId, Arc<Session>>,
    config: Arc<ServerConfig>,
    /// Sessions are timed against a monotonic clock, so a wall-clock jump
    /// cannot expire them all at once.
    epoch: Instant,
}

impl SessionRegistry {
    pub fn new(config: Arc<ServerConfig>) -> Self {
        SessionRegistry {
            sessions: DashMap::new(),
            config,
            epoch: Instant::now(),
        }
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    fn idle_timeout_ms(&self) -> u64 {
        self.config
            .limits
            .session_idle_timeout_sec
            .saturating_mul(1000)
    }

    /// Open a session, granting at most the configured number of streams.
    ///
    /// `desired_streams` of 0 means the client has no preference.
    pub fn create(&self, client_name: &str, desired_streams: u32) -> Result<Arc<Session>> {
        self.create_at(client_name, desired_streams, self.now_ms())
    }

    fn create_at(
        &self,
        client_name: &str,
        desired_streams: u32,
        now_ms: u64,
    ) -> Result<Arc<Session>> {
        if self.sessions.len() >= self.config.limits.max_sessions as usize {
            return Err(ServerError::Exhausted(format!(
                "server is at its limit of {} sessions",
                self.config.limits.max_sessions
            )));
        }

        let wanted = if desired_streams == 0 {
            DEFAULT_STREAMS
        } else {
            desired_streams
        };
        let granted_streams = wanted.min(self.config.limits.max_streams_per_session);

        let session = Arc::new(Session {
            id: random_bytes()?,
            token: random_bytes()?,
            granted_streams,
            client_name: client_name.to_string(),
            files: FileRegistry::new(),
            last_seen_ms: AtomicU64::new(now_ms),
        });
        self.sessions.insert(session.id, session.clone());
        Ok(session)
    }

    /// Look up a session and mark it as used.
    pub fn get(&self, id: &[u8]) -> Result<Arc<Session>> {
        self.get_at(id, self.now_ms())
    }

    fn get_at(&self, id: &[u8], now_ms: u64) -> Result<Arc<Session>> {
        let id: SessionId = id.try_into().map_err(|_| {
            ServerError::Auth(format!(
                "session id is {} bytes, expected {}",
                id.len(),
                std::mem::size_of::<SessionId>()
            ))
        })?;

        let session = self
            .sessions
            .get(&id)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| ServerError::Auth("no such session".to_string()))?;

        if session.is_expired(now_ms, self.idle_timeout_ms()) {
            self.sessions.remove(&id);
            return Err(ServerError::Auth(format!(
                "session expired after {} seconds of inactivity",
                self.config.limits.session_idle_timeout_sec
            )));
        }

        session.touch(now_ms);
        Ok(session)
    }

    /// Whether a session is still live, without counting as activity.
    ///
    /// Used by the transfer sweep, which must not keep a session alive merely
    /// by asking about it.
    pub fn contains(&self, id: &SessionId) -> bool {
        self.sessions.contains_key(id)
    }

    /// Drop a session and everything it holds. Returns its id, so that what
    /// hangs off a session elsewhere can go with it.
    pub fn remove(&self, id: &[u8]) -> Result<SessionId> {
        let session = self.get(id)?;
        self.sessions.remove(&session.id);
        Ok(session.id)
    }

    /// Drop every session nobody has touched within the idle timeout.
    pub fn sweep_expired(&self) -> usize {
        self.sweep_expired_at(self.now_ms())
    }

    fn sweep_expired_at(&self, now_ms: u64) -> usize {
        let timeout_ms = self.idle_timeout_ms();
        let before = self.sessions.len();
        self.sessions
            .retain(|_, session| !session.is_expired(now_ms, timeout_ms));
        before - self.sessions.len()
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

/// Unguessable bytes for a session id, a token or a transfer ticket.
///
/// Straight from the OS: these are capabilities, and a seeded generator would
/// make them predictable to anyone who can watch a few of them.
pub(crate) fn random_bytes<const N: usize>() -> Result<[u8; N]> {
    let mut bytes = [0u8; N];
    getrandom::fill(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use aex_core::{ArrayFile, Item, Result as CoreResult};

    use super::*;

    fn registry() -> SessionRegistry {
        SessionRegistry::new(Arc::new(ServerConfig::default()))
    }

    /// A file with nothing in it; the registry only ever stores and hands back.
    struct EmptyFile;

    impl ArrayFile for EmptyFile {
        fn contains(&self, _path: &str) -> bool {
            false
        }
        fn get_item(&self, path: &str) -> CoreResult<Item> {
            Err(aex_core::AexError::NotFound(path.to_string()))
        }
        fn list_children(&self, path: &str) -> CoreResult<Vec<(String, Item)>> {
            Err(aex_core::AexError::NotFound(path.to_string()))
        }
    }

    #[test]
    fn a_session_can_be_created_looked_up_and_dropped() {
        let reg = registry();
        let session = reg.create("test", 4).expect("create");
        assert_eq!(reg.len(), 1);

        let found = reg.get(session.id()).expect("get");
        assert_eq!(found.id(), session.id());
        assert_eq!(found.client_name(), "test");

        assert_eq!(reg.remove(session.id()).expect("remove"), *session.id());
        assert!(reg.is_empty());
        assert!(matches!(reg.get(session.id()), Err(ServerError::Auth(_))));
    }

    #[test]
    fn ids_and_tokens_differ_between_sessions() {
        let reg = registry();
        let a = reg.create("a", 0).unwrap();
        let b = reg.create("b", 0).unwrap();
        assert_ne!(a.id(), b.id());
        assert_ne!(a.token(), b.token());
        // An id must not double as its own token.
        assert_ne!(a.id(), a.token());
    }

    #[test]
    fn a_malformed_session_id_is_an_auth_error() {
        let reg = registry();
        for id in [&[][..], &[0u8; 8][..], &[0u8; 32][..]] {
            assert!(matches!(reg.get(id), Err(ServerError::Auth(_))));
        }
        // Well-formed but never issued.
        assert!(matches!(reg.get(&[7u8; 16]), Err(ServerError::Auth(_))));
    }

    #[test]
    fn streams_are_capped_and_defaulted() {
        let mut config = ServerConfig::default();
        config.limits.max_streams_per_session = 8;
        let reg = SessionRegistry::new(Arc::new(config));

        assert_eq!(reg.create("a", 4).unwrap().granted_streams(), 4);
        // Asking for more than the server allows grants the server's limit.
        assert_eq!(reg.create("b", 64).unwrap().granted_streams(), 8);
        // No preference gets the default.
        assert_eq!(
            reg.create("c", 0).unwrap().granted_streams(),
            DEFAULT_STREAMS
        );
    }

    #[test]
    fn the_session_limit_is_enforced() {
        let mut config = ServerConfig::default();
        config.limits.max_sessions = 2;
        let reg = SessionRegistry::new(Arc::new(config));

        let first = reg.create("a", 0).unwrap();
        reg.create("b", 0).unwrap();
        let err = reg.create("c", 0).unwrap_err();
        assert!(matches!(err, ServerError::Exhausted(_)), "{err}");
        // The limit is on live sessions, so disconnecting makes room.
        reg.remove(first.id()).unwrap();
        reg.create("c", 0).expect("room after a disconnect");
    }

    #[test]
    fn an_idle_session_expires_and_a_busy_one_does_not() {
        let reg = registry();
        let timeout_ms = reg.idle_timeout_ms();
        let session = reg.create_at("test", 0, 0).expect("create");

        // Still within the timeout.
        assert!(reg.get_at(session.id(), timeout_ms).is_ok());
        // That request pushed the deadline out.
        assert!(reg.get_at(session.id(), timeout_ms + 1).is_ok());

        let err = reg
            .get_at(session.id(), 2 * timeout_ms + 2)
            .expect_err("expired");
        assert!(matches!(err, ServerError::Auth(_)), "{err}");
        // An expired session is dropped, not just refused.
        assert!(reg.is_empty());
    }

    #[test]
    fn the_sweep_drops_only_expired_sessions() {
        let reg = registry();
        let timeout_ms = reg.idle_timeout_ms();
        let old = reg.create_at("old", 0, 0).unwrap();
        let fresh = reg.create_at("fresh", 0, timeout_ms).unwrap();

        assert_eq!(reg.sweep_expired_at(timeout_ms), 0);
        assert_eq!(reg.sweep_expired_at(timeout_ms + 1), 1);
        assert_eq!(reg.len(), 1);
        assert!(reg.get_at(fresh.id(), timeout_ms + 1).is_ok());
        assert!(reg.get_at(old.id(), timeout_ms + 1).is_err());
        // Asking whether a session is live must not count as using it.
        assert!(reg.contains(fresh.id()));
        assert!(!reg.contains(old.id()));
    }

    #[test]
    fn file_handles_are_unique_and_never_zero() {
        let reg = registry();
        let session = reg.create("test", 0).unwrap();
        let files = session.files();

        let first = files.insert(Arc::new(EmptyFile));
        let second = files.insert(Arc::new(EmptyFile));
        assert_ne!(first, 0);
        assert_ne!(first, second);
        assert_eq!(files.len(), 2);

        files.get(first).expect("open handle");
        files.remove(first).expect("close");
        assert!(matches!(files.get(first), Err(ServerError::BadRequest(_))));
        // A handle is not reused after its file is closed.
        assert_ne!(files.insert(Arc::new(EmptyFile)), first);
        // Closing twice is a client mistake worth reporting.
        assert!(files.remove(first).is_err());
    }

    #[test]
    fn files_belong_to_one_session_only() {
        let reg = registry();
        let a = reg.create("a", 0).unwrap();
        let b = reg.create("b", 0).unwrap();

        let handle = a.files().insert(Arc::new(EmptyFile));
        assert!(a.files().get(handle).is_ok());
        assert!(b.files().get(handle).is_err());
    }
}
