//! Client configuration.
//!
//! Everything here can also be set from the environment, so that a benchmark
//! sweep does not need a recompile between points.

use std::time::Duration;

/// Data connections the client asks for when it has no reason to ask for more.
///
/// Connections are the one knob that helps at every round trip: on a link with
/// no delay they are the only thing that helps at all, since there is nothing
/// to pipeline. Eight rather than more because a single client should not take
/// half of a server's connection budget to be well served.
pub const DEFAULT_STREAMS: u32 = 8;

/// Fetches outstanding per connection.
///
/// Throughput on a delayed link is set by the bytes in flight, which is this
/// times the chunk size times the connections. Sixteen puts 512 MiB in flight
/// at the default chunk size, enough for 20 Gbit/s at 200 ms, and costs nothing
/// on a fast link: data goes straight into the caller's buffer, so a deeper
/// pipeline holds no more memory, only more sockets with something in them.
pub const DEFAULT_CREDIT: u32 = 16;

#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Data connections to ask for. 0 accepts whatever the server suggests.
    pub streams: u32,
    /// Reported to the server for its logs.
    pub client_name: String,
    pub connect_timeout: Duration,
    /// Ceiling on one gRPC message. A fancy selection is the message that grows.
    pub max_message_bytes: usize,
    /// How much of the logical byte stream one fetch asks for. 0 follows the
    /// server's recommendation, which is the right answer unless a benchmark is
    /// sweeping this.
    pub chunk_bytes: u64,
    /// Fetches one connection may have outstanding at once. Deeper hides more
    /// of the round trip on a long link, at the cost of more data to drain if
    /// the transfer is abandoned.
    pub credit: u32,
    /// How many times a chunk may be fetched again before the transfer fails.
    pub max_retries: u32,
    /// Whether to disable Nagle on the data connections. A fetch is 48 bytes
    /// that a whole chunk is waiting on, so holding it back costs a round trip.
    pub tcp_nodelay: bool,
    /// `SO_RCVBUF` for the data connections. `None` leaves the OS to tune it,
    /// which caps the window below what a high bandwidth-delay link needs.
    pub rcvbuf: Option<usize>,
    /// `host:port` to reach the data plane at instead of what the server
    /// advertises, for when the path goes through a tunnel or a proxy.
    pub data_endpoint: Option<String>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig {
            streams: DEFAULT_STREAMS,
            client_name: default_client_name(),
            connect_timeout: Duration::from_secs(10),
            max_message_bytes: 4 * 1024 * 1024,
            chunk_bytes: 0,
            credit: DEFAULT_CREDIT,
            max_retries: 3,
            tcp_nodelay: true,
            rcvbuf: None,
            data_endpoint: None,
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
        if let Some(bytes) = lookup("AEX_CHUNK_BYTES").and_then(|v| v.parse().ok()) {
            self.chunk_bytes = bytes;
        }
        if let Some(credit) = lookup("AEX_CREDIT").and_then(|v| v.parse().ok()) {
            self.credit = credit;
        }
        if let Some(retries) = lookup("AEX_MAX_RETRIES").and_then(|v| v.parse().ok()) {
            self.max_retries = retries;
        }
        if let Some(nodelay) = lookup("AEX_TCP_NODELAY").and_then(|v| parse_bool(&v)) {
            self.tcp_nodelay = nodelay;
        }
        if let Some(bytes) = lookup("AEX_RCVBUF").and_then(|v| v.parse().ok()) {
            self.rcvbuf = Some(bytes);
        }
        if let Some(endpoint) = lookup("AEX_DATA_ENDPOINT") {
            self.data_endpoint = Some(endpoint);
        }
        self
    }
}

/// Read a flag as a benchmark script is likely to write one.
fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
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
        // 0 means "whatever the server recommends", which is what an untuned
        // client should be doing.
        assert_eq!(config.chunk_bytes, 0);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.credit, DEFAULT_CREDIT);
        assert!(config.tcp_nodelay);
        assert_eq!(config.rcvbuf, None);
    }

    #[test]
    fn the_environment_overrides_what_it_names() {
        let config = ClientConfig::default().with_env(|name| match name {
            "AEX_STREAMS" => Some("16".to_string()),
            "AEX_CONNECT_TIMEOUT_MS" => Some("250".to_string()),
            "AEX_CHUNK_BYTES" => Some("262144".to_string()),
            "AEX_TCP_NODELAY" => Some("off".to_string()),
            "AEX_RCVBUF" => Some("8388608".to_string()),
            "AEX_CREDIT" => Some("1".to_string()),
            _ => None,
        });
        assert_eq!(config.streams, 16);
        assert_eq!(config.rcvbuf, Some(8 << 20));
        assert_eq!(config.credit, 1);
        assert_eq!(config.connect_timeout, Duration::from_millis(250));
        assert_eq!(config.chunk_bytes, 256 * 1024);
        assert!(!config.tcp_nodelay);
        // Untouched settings keep their defaults.
        assert_eq!(config.max_message_bytes, 4 * 1024 * 1024);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn an_unparseable_value_leaves_the_default() {
        let config = ClientConfig::default().with_env(|name| match name {
            "AEX_STREAMS" => Some("many".to_string()),
            "AEX_TCP_NODELAY" => Some("perhaps".to_string()),
            _ => None,
        });
        assert_eq!(config.streams, DEFAULT_STREAMS);
        assert!(config.tcp_nodelay);
    }
}
