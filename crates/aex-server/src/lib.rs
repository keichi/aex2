//! The AEX2 server.
//!
//! As of M1 that is the gRPC control plane: sessions, files and metadata. The
//! data plane listener joins it in M2, sharing the session registry.
//!
//! [`ControlServer::bind`] takes the socket before serving so that a caller —
//! an integration test, say — can ask for port 0 and still learn where to
//! connect.

pub mod config;
pub mod control;
pub mod error;
pub mod paths;
pub mod session;

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use aex_proto::aex_control_server::AexControlServer;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;

pub use crate::config::ServerConfig;
use crate::control::ControlService;
pub use crate::error::{Result, ServerError};
use crate::paths::PathPolicy;
use crate::session::SessionRegistry;

/// How often idle sessions are collected, as a fraction of the idle timeout.
///
/// Sweeping is only about releasing what a vanished client left behind: a
/// request that names an expired session is refused whether or not a sweep has
/// run yet, so sweeping often would buy nothing.
const SWEEPS_PER_TIMEOUT: u32 = 4;

/// A control plane that has its socket and is ready to serve.
pub struct ControlServer {
    listener: TcpListener,
    sessions: Arc<SessionRegistry>,
    service: ControlService,
    config: Arc<ServerConfig>,
}

impl ControlServer {
    /// Validate the configuration, resolve the data roots and take the socket.
    pub async fn bind(config: ServerConfig) -> Result<Self> {
        config.validate()?;
        let paths = Arc::new(PathPolicy::new(&config.paths.roots)?);
        let config = Arc::new(config);
        let sessions = Arc::new(SessionRegistry::new(config.clone()));
        let listener = TcpListener::bind(config.control_addr).await?;

        Ok(ControlServer {
            listener,
            service: ControlService::new(sessions.clone(), paths, config.clone()),
            sessions,
            config,
        })
    }

    /// The address actually bound, which differs from the configured one when
    /// the port was 0.
    pub fn control_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    pub fn config(&self) -> &Arc<ServerConfig> {
        &self.config
    }

    /// Serve until the process is killed.
    pub async fn serve(self) -> Result<()> {
        self.serve_with_shutdown(std::future::pending()).await
    }

    /// Serve until `shutdown` resolves.
    pub async fn serve_with_shutdown(
        self,
        shutdown: impl Future<Output = ()> + Send,
    ) -> Result<()> {
        let ControlServer {
            listener,
            sessions,
            service,
            config,
        } = self;

        let sweeper = tokio::spawn(sweep_sessions(
            sessions,
            Duration::from_secs(
                (config.limits.session_idle_timeout_sec / SWEEPS_PER_TIMEOUT as u64).max(1),
            ),
        ));

        let service = AexControlServer::new(service)
            .max_decoding_message_size(config.limits.grpc_max_message_bytes)
            .max_encoding_message_size(config.limits.grpc_max_message_bytes);

        let result = tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
            .await;

        sweeper.abort();
        result.map_err(ServerError::from)
    }
}

/// Collect sessions no request has touched within the idle timeout.
async fn sweep_sessions(sessions: Arc<SessionRegistry>, period: Duration) {
    let mut ticker = tokio::time::interval(period);
    // The first tick fires immediately; nothing can have expired by then.
    ticker.tick().await;
    loop {
        ticker.tick().await;
        let dropped = sessions.sweep_expired();
        if dropped > 0 {
            tracing::info!(dropped, "collected idle sessions");
        }
    }
}
