//! Server configuration.
//!
//! Every knob has a default, so a server runs with nothing but `--root`. The
//! TOML file mirrors the struct: `[server]`, `[server.limits]`, and so on.
//!
//! Transfer and TCP settings are read and validated here even though the data
//! plane that uses them arrives in M2. Keeping the file format complete from
//! the start means a deployment's config does not have to change under it.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Result, ServerError};

/// The data plane frame version this server speaks.
pub const PROTOCOL_VERSION: u32 = 1;

/// Streams granted when the client expresses no preference.
///
/// Matches the client default: a client asking for "whatever you think" is
/// almost always one that has not been tuned, and four streams is the point
/// where a single connection stops being the limit on a fast link.
pub const DEFAULT_STREAMS: u32 = 4;

/// Codecs this server can produce. Bit 0 is RAW.
pub const SUPPORTED_CODECS: u32 = 1;

/// Encodings this server can produce. Bit 0 is EXACT.
pub const SUPPORTED_ENCODINGS: u32 = 1;

/// A config file. The single `[server]` table keeps room for other sections.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    #[serde(default)]
    server: ServerConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    pub control_addr: SocketAddr,
    pub data_addr: SocketAddr,
    /// Host to advertise for the data plane. Empty means "the host you reached
    /// the control plane on", which is right whenever the server cannot know
    /// how the client addresses it (NAT, container, several interfaces).
    pub data_advertise_host: String,
    pub limits: Limits,
    pub transfer: Transfer,
    pub tcp: Tcp,
    pub paths: Paths,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Limits {
    pub max_sessions: u32,
    /// Ceiling on the expanded indices of one fancy selection. A larger request
    /// would not fit `grpc_max_message_bytes` anyway.
    pub max_fancy_indices: u64,
    pub grpc_max_message_bytes: usize,
    pub max_streams_per_session: u32,
    pub max_transfers_per_session: u32,
    pub session_idle_timeout_sec: u64,
    pub data_conn_idle_timeout_sec: u64,
    pub transfer_ttl_sec: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Transfer {
    pub default_chunk_bytes: u64,
    /// Read buffers per connection, for overlapping a read with a send.
    pub read_buffers: u32,
    pub max_fetch_bytes: u64,
    /// Selections at most this large come back inside the transfer plan, which
    /// keeps an interactive read at one round trip.
    pub inline_limit_bytes: u64,
    /// LRU for decompressed storage chunks. Unused while only `.npy` is served.
    pub decode_cache_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Tcp {
    pub nodelay: bool,
    /// 0 leaves the OS default.
    pub sndbuf: usize,
    /// Congestion control algorithm. Linux only; empty leaves the OS default.
    pub congestion: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Paths {
    /// Files may only be opened under these directories.
    pub roots: Vec<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            control_addr: "0.0.0.0:50051".parse().expect("valid default address"),
            data_addr: "0.0.0.0:50052".parse().expect("valid default address"),
            data_advertise_host: String::new(),
            limits: Limits::default(),
            transfer: Transfer::default(),
            tcp: Tcp::default(),
            paths: Paths::default(),
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_sessions: 64,
            max_fancy_indices: 262_144,
            grpc_max_message_bytes: 4 * 1024 * 1024,
            max_streams_per_session: 32,
            max_transfers_per_session: 64,
            session_idle_timeout_sec: 300,
            data_conn_idle_timeout_sec: 300,
            transfer_ttl_sec: 60,
        }
    }
}

impl Default for Transfer {
    fn default() -> Self {
        Transfer {
            default_chunk_bytes: 4 * 1024 * 1024,
            read_buffers: 3,
            max_fetch_bytes: 16 * 1024 * 1024,
            inline_limit_bytes: 64 * 1024,
            decode_cache_bytes: 0,
        }
    }
}

impl Default for Tcp {
    fn default() -> Self {
        Tcp {
            nodelay: true,
            sndbuf: 0,
            congestion: String::new(),
        }
    }
}

impl Default for Paths {
    fn default() -> Self {
        Paths {
            roots: vec![PathBuf::from("/data")],
        }
    }
}

impl ServerConfig {
    /// Read a config file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| ServerError::Config(format!("cannot read {}: {e}", path.display())))?;
        let file: ConfigFile = toml::from_str(&text)
            .map_err(|e| ServerError::Config(format!("{}: {e}", path.display())))?;
        Ok(file.server)
    }

    /// Reject settings that would misbehave rather than fail loudly later.
    pub fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(ServerError::Config(msg));

        if self.limits.max_sessions == 0 {
            return bad("limits.max_sessions must be at least 1".to_string());
        }
        if self.limits.max_streams_per_session == 0 {
            return bad("limits.max_streams_per_session must be at least 1".to_string());
        }
        if self.limits.max_transfers_per_session == 0 {
            return bad("limits.max_transfers_per_session must be at least 1".to_string());
        }
        if self.limits.session_idle_timeout_sec == 0 {
            return bad("limits.session_idle_timeout_sec must be at least 1".to_string());
        }
        if self.limits.transfer_ttl_sec == 0 {
            return bad("limits.transfer_ttl_sec must be at least 1".to_string());
        }
        if self.transfer.default_chunk_bytes == 0 {
            return bad("transfer.default_chunk_bytes must be at least 1".to_string());
        }
        if self.transfer.read_buffers == 0 {
            return bad("transfer.read_buffers must be at least 1".to_string());
        }
        if self.transfer.max_fetch_bytes < self.transfer.default_chunk_bytes {
            // A client that follows the recommended chunk size must not have
            // every one of its fetches rejected.
            return bad(format!(
                "transfer.max_fetch_bytes ({}) is below transfer.default_chunk_bytes ({})",
                self.transfer.max_fetch_bytes, self.transfer.default_chunk_bytes
            ));
        }
        if self.transfer.inline_limit_bytes > self.limits.grpc_max_message_bytes as u64 {
            // Inline data travels inside a TransferPlan, so it has to fit one
            // gRPC message with room to spare for the rest of the plan.
            return bad(format!(
                "transfer.inline_limit_bytes ({}) exceeds limits.grpc_max_message_bytes ({})",
                self.transfer.inline_limit_bytes, self.limits.grpc_max_message_bytes
            ));
        }
        if self.paths.roots.is_empty() {
            return bad("paths.roots is empty: no file could be opened".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_documented_ones() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.control_addr.port(), 50051);
        assert_eq!(cfg.data_addr.port(), 50052);
        assert_eq!(cfg.limits.max_sessions, 64);
        assert_eq!(cfg.transfer.default_chunk_bytes, 4 * 1024 * 1024);
        assert_eq!(cfg.transfer.inline_limit_bytes, 64 * 1024);
        assert!(cfg.tcp.nodelay);
        cfg.validate().expect("the defaults must be valid");
    }

    #[test]
    fn a_file_overrides_only_what_it_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("aex.toml");
        std::fs::write(
            &path,
            r#"
            [server]
            control_addr = "127.0.0.1:1234"

            [server.limits]
            max_sessions = 8

            [server.paths]
            roots = ["/srv/data", "/srv/more"]
            "#,
        )
        .unwrap();

        let cfg = ServerConfig::load(&path).expect("load");
        assert_eq!(cfg.control_addr.port(), 1234);
        assert_eq!(cfg.limits.max_sessions, 8);
        assert_eq!(cfg.paths.roots.len(), 2);
        // Untouched settings keep their defaults.
        assert_eq!(cfg.data_addr.port(), 50052);
        assert_eq!(cfg.limits.transfer_ttl_sec, 60);
        assert_eq!(cfg.transfer.max_fetch_bytes, 16 * 1024 * 1024);
    }

    #[test]
    fn an_empty_file_is_the_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.toml");
        std::fs::write(&path, "").unwrap();
        let cfg = ServerConfig::load(&path).expect("load");
        assert_eq!(cfg.control_addr, ServerConfig::default().control_addr);
    }

    #[test]
    fn a_misspelled_key_is_an_error() {
        // Silently ignoring it would leave the operator with a server that is
        // not configured the way they think it is.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("typo.toml");
        std::fs::write(&path, "[server.limits]\nmax_session = 8\n").unwrap();
        let err = ServerConfig::load(&path).unwrap_err();
        assert!(matches!(err, ServerError::Config(_)), "{err}");
        assert!(err.to_string().contains("max_session"), "{err}");
    }

    #[test]
    fn a_missing_file_is_an_error() {
        let err = ServerConfig::load("/nonexistent/aex2/server.toml").unwrap_err();
        assert!(matches!(err, ServerError::Config(_)), "{err}");
    }

    #[test]
    fn validate_rejects_settings_that_contradict_each_other() {
        let mut cfg = ServerConfig::default();
        cfg.paths.roots.clear();
        assert!(cfg.validate().is_err());

        let mut cfg = ServerConfig::default();
        cfg.transfer.max_fetch_bytes = cfg.transfer.default_chunk_bytes - 1;
        assert!(cfg.validate().is_err());

        let mut cfg = ServerConfig::default();
        cfg.transfer.inline_limit_bytes = cfg.limits.grpc_max_message_bytes as u64 + 1;
        assert!(cfg.validate().is_err());

        let mut cfg = ServerConfig::default();
        cfg.limits.max_sessions = 0;
        assert!(cfg.validate().is_err());
    }
}
