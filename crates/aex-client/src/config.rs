//! Client configuration.
//!
//! Everything here can also be set from the environment, so that a benchmark
//! sweep does not need a recompile between points.
//!
//! Only the settings M1 can act on are here. The transfer knobs — chunk size,
//! credit, codec, receive buffer, retry count — join them with the data plane
//! they belong to, rather than sitting unused and looking as if they work.

use std::time::Duration;

/// Data connections the client asks for when it has no reason to ask for more.
///
/// One connection cannot fill a high bandwidth-delay link, and past a handful
/// the gain flattens while the server-side cost does not.
pub const DEFAULT_STREAMS: u32 = 4;

#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Data connections to ask for. 0 accepts whatever the server suggests.
    pub streams: u32,
    /// Reported to the server for its logs.
    pub client_name: String,
    pub connect_timeout: Duration,
    /// Ceiling on one gRPC message. A fancy selection is the message that grows.
    pub max_message_bytes: usize,
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig {
            streams: DEFAULT_STREAMS,
            client_name: default_client_name(),
            connect_timeout: Duration::from_secs(10),
            max_message_bytes: 4 * 1024 * 1024,
        }
    }
}

impl ClientConfig {
    /// The defaults, with `AEX_*` environment variables applied over them.
    pub fn from_env() -> Self {
        Self::default().with_env(|name| std::env::var(name).ok())
    }

    /// Apply environment overrides from an arbitrary source.
    ///
    /// A value that does not parse is ignored rather than fatal: an unusable
    /// `AEX_STREAMS` should not stop a benchmark run, and the setting it fails
    /// to change is visible in the transfer statistics.
    pub fn with_env(mut self, lookup: impl Fn(&str) -> Option<String>) -> Self {
        if let Some(streams) = lookup("AEX_STREAMS").and_then(|v| v.parse().ok()) {
            self.streams = streams;
        }
        if let Some(name) = lookup("AEX_CLIENT_NAME") {
            self.client_name = name;
        }
        if let Some(ms) = lookup("AEX_CONNECT_TIMEOUT_MS").and_then(|v| v.parse().ok()) {
            self.connect_timeout = Duration::from_millis(ms);
        }
        if let Some(bytes) = lookup("AEX_MAX_MESSAGE_BYTES").and_then(|v| v.parse().ok()) {
            self.max_message_bytes = bytes;
        }
        self
    }
}

/// Program name and process id, which is enough to tell two runs apart in a
/// server log.
fn default_client_name() -> String {
    let program = std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "aex-client".to_string());
    format!("{program}[{}]", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_the_documented_ones() {
        let config = ClientConfig::default();
        assert_eq!(config.streams, DEFAULT_STREAMS);
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert!(!config.client_name.is_empty());
    }

    #[test]
    fn the_environment_overrides_what_it_names() {
        let config = ClientConfig::default().with_env(|name| match name {
            "AEX_STREAMS" => Some("16".to_string()),
            "AEX_CONNECT_TIMEOUT_MS" => Some("250".to_string()),
            _ => None,
        });
        assert_eq!(config.streams, 16);
        assert_eq!(config.connect_timeout, Duration::from_millis(250));
        // Untouched settings keep their defaults.
        assert_eq!(config.max_message_bytes, 4 * 1024 * 1024);
    }

    #[test]
    fn an_unparseable_value_leaves_the_default() {
        let config = ClientConfig::default().with_env(|name| match name {
            "AEX_STREAMS" => Some("many".to_string()),
            _ => None,
        });
        assert_eq!(config.streams, DEFAULT_STREAMS);
    }
}
