//! The AEX2 server.
//!
//! The gRPC control plane decides *what* to send; the data plane sends it. They
//! meet at the transfer registry, which one writes and the other reads, and
//! nowhere else. That separation is what keeps the bulk data out of protobuf
//! encoding entirely.

pub mod config;
pub mod control;
pub mod dataplane;
pub mod error;
pub mod paths;
pub mod reader;
pub mod session;
pub mod transfer;

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use aex_proto::aex_control_server::AexControlServer;
use tokio::net::TcpListener;
use tonic::transport::server::TcpIncoming;

pub use crate::config::ServerConfig;
use crate::control::ControlService;
use crate::dataplane::DataPlane;
pub use crate::error::{Result, ServerError};
use crate::paths::PathPolicy;
use crate::session::SessionRegistry;
use crate::transfer::TransferRegistry;

/// How often idle sessions and stale plans are collected, as a fraction of the
/// idle timeout.
///
/// Sweeping is only about releasing what a vanished client left behind: a
/// request that names an expired session is refused whether or not a sweep has
/// run yet, so sweeping often would buy nothing.
const SWEEPS_PER_TIMEOUT: u32 = 4;

/// A server that has its sockets and is ready to serve.
pub struct Server {
    control: TcpListener,
    data: DataPlane,
    sessions: Arc<SessionRegistry>,
    transfers: Arc<TransferRegistry>,
    service: ControlService,
    config: Arc<ServerConfig>,
}

impl Server {
    /// Validate the configuration, resolve the data roots and take the sockets.
    pub async fn bind(config: ServerConfig) -> Result<Self> {
        config.validate()?;
        let paths = Arc::new(PathPolicy::new(&config.paths.roots)?);
        let config = Arc::new(config);
        let sessions = Arc::new(SessionRegistry::new(config.clone()));
        let transfers = Arc::new(TransferRegistry::new(config.clone()));

        // Bound first so that the port it really got is the port the control
        // plane hands out.
        let data = DataPlane::bind(config.clone(), sessions.clone(), transfers.clone())?;
        let data_port = data.local_addr()?.port();
        let control = TcpListener::bind(config.control_addr).await?;

        Ok(Server {
            control,
            data,
            service: ControlService::new(
                sessions.clone(),
                transfers.clone(),
                paths,
                config.clone(),
                data_port,
            ),
            sessions,
            transfers,
            config,
        })
    }

    /// The control plane address actually bound.
    pub fn control_addr(&self) -> Result<SocketAddr> {
        Ok(self.control.local_addr()?)
    }

    /// The data plane address actually bound.
    pub fn data_addr(&self) -> Result<SocketAddr> {
        self.data.local_addr()
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
        let Server {
            control,
            data,
            sessions,
            transfers,
            service,
            config,
        } = self;

        // Dropping the handle stops the data plane, which happens on every way
        // out of this function.
        let _data = data.serve();

        let sweeper = tokio::spawn(sweep(
            sessions,
            transfers,
            Duration::from_secs(
                (config.limits.session_idle_timeout_sec / SWEEPS_PER_TIMEOUT as u64).max(1),
            ),
        ));

        let service = AexControlServer::new(service)
            .max_decoding_message_size(config.limits.grpc_max_message_bytes)
            .max_encoding_message_size(config.limits.grpc_max_message_bytes);

        // Our own listener bypasses the builder's tcp_nodelay; without it
        // Nagle adds a round trip to every inline reply.
        let incoming = TcpIncoming::from(control).with_nodelay(Some(true));

        let result = tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming_shutdown(incoming, shutdown)
            .await;

        sweeper.abort();
        result.map_err(ServerError::from)
    }
}

/// Collect sessions no request has touched within the idle timeout, and the
/// plans that have gone stale or whose session went with them.
async fn sweep(sessions: Arc<SessionRegistry>, transfers: Arc<TransferRegistry>, period: Duration) {
    let mut ticker = tokio::time::interval(period);
    // The first tick fires immediately; nothing can have expired by then.
    ticker.tick().await;
    loop {
        ticker.tick().await;
        let dropped = sessions.sweep_expired();
        // After the sessions, so that a session dropped in this same pass takes
        // its plans with it rather than leaving them for the next one.
        let stale = transfers.sweep(|session| sessions.contains(session));
        if dropped > 0 || stale > 0 {
            tracing::info!(
                sessions = dropped,
                transfers = stale,
                "collected what went idle"
            );
        }
    }
}
