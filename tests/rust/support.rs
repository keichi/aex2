//! What the integration tests share: a server in this process, and fixtures.
//!
//! The server runs on its own thread and loopback ports, so a test exercises
//! the real paths — tonic codecs, the TCP data plane — without anything to set
//! up first.
//!
//! Fixture files are written with `npyz`, which is also what the server parses
//! headers with. Agreement with what numpy itself writes is the job of the
//! differential tests against numpy; what these pin down is the protocol.

// Each test binary uses part of this.
#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use aex_client::{Client, ClientConfig, ClientError};
use aex_core::ErrorClass;
use aex_server::{Server, ServerConfig};
use tempfile::TempDir;

/// A server running on loopback ports, shut down when dropped.
pub struct TestServer {
    pub url: String,
    pub data_addr: SocketAddr,
    root: TempDir,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl TestServer {
    pub fn start() -> TestServer {
        Self::start_with(|_| {})
    }

    /// Start a server, letting the caller adjust the configuration first.
    pub fn start_with(adjust: impl FnOnce(&mut ServerConfig)) -> TestServer {
        let root = tempfile::tempdir().expect("tempdir");

        let mut config = ServerConfig {
            // Port 0: the OS picks one, so tests can run at the same time.
            control_addr: "127.0.0.1:0".parse().unwrap(),
            data_addr: "127.0.0.1:0".parse().unwrap(),
            paths: aex_server::config::Paths {
                roots: vec![root.path().to_path_buf()],
            },
            ..ServerConfig::default()
        };
        adjust(&mut config);

        let (addr_tx, addr_rx) = std::sync::mpsc::channel();
        let (stop, stop_rx) = tokio::sync::oneshot::channel();

        let thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                let server = Server::bind(config).await.expect("bind");
                addr_tx
                    .send((
                        server.control_addr().expect("control addr"),
                        server.data_addr().expect("data addr"),
                    ))
                    .expect("the test is waiting for the addresses");
                server
                    .serve_with_shutdown(async {
                        let _ = stop_rx.await;
                    })
                    .await
                    .expect("serve");
            });
        });

        let (control_addr, data_addr) = addr_rx.recv().expect("server failed to start");
        TestServer {
            url: format!("http://{control_addr}"),
            data_addr,
            root,
            stop: Some(stop),
            thread: Some(thread),
        }
    }

    pub fn connect(&self) -> Client {
        self.connect_with(ClientConfig::default())
    }

    pub fn connect_with(&self, config: ClientConfig) -> Client {
        Client::connect(&self.url, config).expect("connect")
    }

    pub fn root(&self) -> &Path {
        self.root.path()
    }

    /// Write a `.npy` of `f32` counting up from zero, and return what is in it.
    ///
    /// Counting up makes every element say where it came from, so a selection
    /// that lands one row off is a visible mismatch rather than a plausible
    /// number.
    pub fn write_counting_npy(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        self.write_npy(name, shape);
        (0..shape.iter().product::<usize>())
            .map(|i| i as f32)
            .collect()
    }

    /// Write a `.npy` of `f32` counting up from zero.
    pub fn write_npy(&self, name: &str, shape: &[usize]) -> PathBuf {
        let path = self.root().join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        let count: usize = shape.iter().product();
        let shape: Vec<u64> = shape.iter().map(|&n| n as u64).collect();

        let file = std::io::BufWriter::new(std::fs::File::create(&path).expect("create"));
        let mut writer = {
            use npyz::WriterBuilder;
            npyz::WriteOptions::new()
                .default_dtype()
                .shape(&shape)
                .writer(file)
                .begin_nd()
                .expect("npy header")
        };
        for i in 0..count {
            writer.push(&(i as f32)).expect("npy element");
        }
        writer.finish().expect("npy finish");
        path
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A session opened over gRPC alone, holding no data connections, for tests
/// that speak the data plane by hand.
pub struct RawSession {
    pub hello: aex_wire::Hello,
    id: Vec<u8>,
    control: aex_proto::aex_control_client::AexControlClient<tonic::transport::Channel>,
    runtime: tokio::runtime::Runtime,
}

impl RawSession {
    pub fn open(&mut self, path: &str) -> u64 {
        let request = aex_proto::OpenFileRequest {
            session_id: self.id.clone(),
            path: path.to_string(),
            format: String::new(),
        };
        self.runtime
            .block_on(self.control.open_file(request))
            .expect("open")
            .into_inner()
            .handle
    }

    /// Prepare a whole dataset, for tests that then speak the data plane by
    /// hand and so need the request id and the ticket.
    pub fn prepare(
        &mut self,
        handle: u64,
        name: &str,
        quality: Option<aex_proto::QualitySpec>,
    ) -> aex_proto::TransferPlan {
        let request = aex_proto::PrepareSelectionRequest {
            session_id: self.id.clone(),
            handle,
            name: name.to_string(),
            indices: Vec::new(),
            requested_quality: quality,
            requested_codec: 0,
        };
        self.runtime
            .block_on(self.control.prepare_selection(request))
            .expect("prepare")
            .into_inner()
    }

    pub fn disconnect(&mut self) {
        let request = aex_proto::DisconnectRequest {
            session_id: self.id.clone(),
        };
        self.runtime
            .block_on(self.control.disconnect(request))
            .expect("disconnect");
    }
}

impl TestServer {
    pub fn raw_session(&self, streams: u32) -> RawSession {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let channel = runtime
            .block_on(
                tonic::transport::Endpoint::from(
                    self.url.parse::<tonic::codegen::http::Uri>().unwrap(),
                )
                .connect(),
            )
            .expect("connect");
        let mut control = aex_proto::aex_control_client::AexControlClient::new(channel);
        let reply = runtime
            .block_on(control.connect(aex_proto::ConnectRequest {
                protocol_version: aex_client::PROTOCOL_VERSION,
                desired_streams: streams,
                client_name: "raw".to_string(),
            }))
            .expect("session")
            .into_inner();
        RawSession {
            hello: aex_wire::Hello::new(
                reply.session_id.clone().try_into().expect("16 bytes"),
                reply.session_token.try_into().expect("16 bytes"),
            ),
            id: reply.session_id,
            control,
            runtime,
        }
    }
}

/// What a [`Proxy`] does to the connections it carries.
#[derive(Debug, Clone, Copy)]
pub enum Fault {
    None,
    /// Tear each connection down once it has carried this many bytes towards
    /// the client, mid-frame if need be.
    CutAfter(u64),
    /// Answer the first fetch on each connection with a TRANSIENT error
    /// instead of passing it on. Being first, it cannot land inside a frame.
    FailFirstFetch,
    /// Hold every read this long before passing it on, in either direction:
    /// a long link, one round trip costing twice this.
    Delay(Duration),
}

/// A TCP proxy in front of the data plane that can break connections.
///
/// Threads are left to die with the sockets; a test process is short-lived.
pub struct Proxy {
    pub addr: SocketAddr,
    accepted: Arc<AtomicUsize>,
    cuts: Arc<AtomicUsize>,
    failed: Arc<AtomicUsize>,
}

impl Proxy {
    pub fn start(target: SocketAddr, fault: Fault) -> Proxy {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind proxy");
        let addr = listener.local_addr().unwrap();
        let accepted = Arc::new(AtomicUsize::new(0));
        let cuts = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));
        let (a, c, f) = (accepted.clone(), cuts.clone(), failed.clone());
        std::thread::spawn(move || {
            for client in listener.incoming() {
                let Ok(client) = client else { return };
                let Ok(server) = TcpStream::connect(target) else {
                    return;
                };
                a.fetch_add(1, Ordering::Relaxed);
                // Both directions write to the client under this lock, so an
                // injected frame never lands inside a forwarded one.
                let down = Arc::new(Mutex::new(client.try_clone().unwrap()));
                let (up_from, up_to) = (client, server.try_clone().unwrap());
                let (d, f) = (down.clone(), f.clone());
                std::thread::spawn(move || pump_up(up_from, up_to, &d, fault, &f));
                let cuts = c.clone();
                std::thread::spawn(move || pump_down(server, &down, fault, &cuts));
            }
        });
        Proxy {
            addr,
            accepted,
            cuts,
            failed,
        }
    }

    /// `host:port`, as `ClientConfig::data_endpoint` takes it.
    pub fn endpoint(&self) -> String {
        self.addr.to_string()
    }

    /// Connections accepted so far.
    pub fn accepted(&self) -> usize {
        self.accepted.load(Ordering::Relaxed)
    }

    /// Connections torn down so far.
    pub fn cuts(&self) -> usize {
        self.cuts.load(Ordering::Relaxed)
    }

    /// Fetches answered with an injected error so far.
    pub fn failed_fetches(&self) -> usize {
        self.failed.load(Ordering::Relaxed)
    }
}

fn pump_up(
    mut client: TcpStream,
    mut server: TcpStream,
    down: &Mutex<TcpStream>,
    fault: Fault,
    failed: &AtomicUsize,
) {
    if let Fault::FailFirstFetch = fault {
        let mut hello = [0u8; aex_wire::HELLO_LEN];
        let mut header = [0u8; aex_wire::HEADER_LEN];
        let mut ticket = [0u8; aex_wire::TICKET_LEN];
        if client.read_exact(&mut hello).is_err()
            || server.write_all(&hello).is_err()
            || client.read_exact(&mut header).is_err()
            || client.read_exact(&mut ticket).is_err()
        {
            return;
        }
        let fetch = aex_wire::FrameHeader::decode(&header).expect("a fetch");
        let payload =
            aex_wire::ErrorPayload::new(ErrorClass::Transient, "injected by the test proxy")
                .encode();
        let error = aex_wire::FrameHeader::error(
            fetch.request_id,
            fetch.offset,
            fetch.logical_len,
            payload.len() as u64,
        );
        let mut down = down.lock().unwrap();
        if aex_wire::write_frame(&mut *down, &error, &payload).is_err() {
            return;
        }
        failed.fetch_add(1, Ordering::Relaxed);
    }
    if let Fault::Delay(delay) = fault {
        let mut buf = vec![0u8; 64 * 1024];
        while let Ok(n @ 1..) = client.read(&mut buf) {
            std::thread::sleep(delay);
            if server.write_all(&buf[..n]).is_err() {
                break;
            }
        }
    } else {
        let _ = std::io::copy(&mut client, &mut server);
    }
    let _ = server.shutdown(Shutdown::Write);
}

fn pump_down(mut server: TcpStream, down: &Mutex<TcpStream>, fault: Fault, cuts: &AtomicUsize) {
    let mut left = match fault {
        Fault::CutAfter(bytes) => bytes,
        _ => u64::MAX,
    };
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = match server.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        if let Fault::Delay(delay) = fault {
            std::thread::sleep(delay);
        }
        let pass = (n as u64).min(left) as usize;
        let mut client = down.lock().unwrap();
        if client.write_all(&buf[..pass]).is_err() {
            break;
        }
        left -= pass as u64;
        if left == 0 {
            cuts.fetch_add(1, Ordering::Relaxed);
            let _ = client.shutdown(Shutdown::Both);
            let _ = server.shutdown(Shutdown::Both);
            return;
        }
    }
    let _ = down.lock().unwrap().shutdown(Shutdown::Write);
}

/// The class of a server error, or a failure if the call did not fail.
pub fn error_class(result: Result<impl std::fmt::Debug, ClientError>) -> ErrorClass {
    match result {
        Ok(value) => panic!("expected an error, got {value:?}"),
        Err(err) => err
            .class()
            .unwrap_or_else(|| panic!("expected a server error, got {err}")),
    }
}

/// Write a `.npy` with an arbitrary `descr` and raw payload.
///
/// npyz's writer only writes the dtypes it can serialise; this covers the rest.
pub fn write_raw_npy(path: &Path, descr: &str, shape: &[u64], payload: &[u8]) {
    let shape_text = match shape {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        dims => format!(
            "({})",
            dims.iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}");
    // A v1.0 header: magic, version, a 2-byte length, then text padded so the
    // data starts on a 64-byte boundary.
    let padding = (64 - (10 + dict.len() + 1) % 64) % 64;
    let header_len = dict.len() + padding + 1;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(header_len as u16).to_le_bytes());
    bytes.extend_from_slice(dict.as_bytes());
    bytes.extend(std::iter::repeat_n(b' ', padding));
    bytes.push(b'\n');
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes).expect("write npy");
}
