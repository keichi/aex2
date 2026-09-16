//! The `aex-server` binary.

use std::net::SocketAddr;
use std::path::PathBuf;

use aex_server::{Result, Server, ServerConfig};
use clap::Parser;
use tracing_subscriber::EnvFilter;

/// Serve array data over AEX2.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Configuration file. Anything it leaves out keeps its default.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Address for the gRPC control plane.
    #[arg(long, value_name = "ADDR")]
    control_addr: Option<SocketAddr>,

    /// Address for the data plane.
    #[arg(long, value_name = "ADDR")]
    data_addr: Option<SocketAddr>,

    /// Directory files may be opened under. Repeatable; replaces the
    /// configured roots rather than adding to them.
    #[arg(long = "root", value_name = "DIR")]
    roots: Vec<PathBuf>,
}

impl Cli {
    /// The configuration file, with the command line applied on top.
    fn config(&self) -> Result<ServerConfig> {
        let mut config = match &self.config {
            Some(path) => ServerConfig::load(path)?,
            None => ServerConfig::default(),
        };
        if let Some(addr) = self.control_addr {
            config.control_addr = addr;
        }
        if let Some(addr) = self.data_addr {
            config.data_addr = addr;
        }
        if !self.roots.is_empty() {
            config.paths.roots = self.roots.clone();
        }
        Ok(config)
    }
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("AEX_LOG").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    match run(Cli::parse()).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    let config = cli.config()?;
    let server = Server::bind(config).await?;
    tracing::info!(
        control = %server.control_addr()?,
        data = %server.data_addr()?,
        roots = ?server.config().paths.roots,
        "aex-server listening"
    );

    server
        .serve_with_shutdown(async {
            // Ctrl-C is the shutdown signal; without a handler tokio would let
            // the default one kill the process mid-request.
            if let Err(e) = tokio::signal::ctrl_c().await {
                tracing::error!("cannot listen for ctrl-c: {e}");
                return;
            }
            tracing::info!("shutting down");
        })
        .await
}
