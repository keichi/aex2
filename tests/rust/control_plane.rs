//! Control plane round trips, with a real server and a real client.
//!
//! The server runs in this process on its own thread and a loopback port, so a
//! test exercises the RPC path end to end — tonic codecs included — without
//! anything to set up first.
//!
//! Fixture files are written with `npyz`, which is also what the server parses
//! headers with. Agreement with what numpy itself writes is the job of the
//! differential tests in M3; what these tests pin down is the RPC layer.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use aex_client::{Client, ClientConfig, ClientError, Item};
use aex_core::{DType, ErrorClass};
use aex_server::{ControlServer, ServerConfig};
use tempfile::TempDir;

/// Port advertised for the data plane. Nothing listens on it: M1 only has to
/// carry the number to the client.
const DATA_PORT: u16 = 59999;

/// A server running on a loopback port, shut down when dropped.
struct TestServer {
    url: String,
    root: TempDir,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl TestServer {
    fn start() -> TestServer {
        Self::start_with(|_| {})
    }

    /// Start a server, letting the caller adjust the configuration first.
    fn start_with(adjust: impl FnOnce(&mut ServerConfig)) -> TestServer {
        let root = tempfile::tempdir().expect("tempdir");

        let mut config = ServerConfig {
            // Port 0: the OS picks one, so tests can run at the same time.
            control_addr: "127.0.0.1:0".parse().unwrap(),
            data_addr: SocketAddr::from(([127, 0, 0, 1], DATA_PORT)),
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
                let server = ControlServer::bind(config).await.expect("bind");
                addr_tx
                    .send(server.control_addr().expect("local addr"))
                    .expect("the test is waiting for the address");
                server
                    .serve_with_shutdown(async {
                        let _ = stop_rx.await;
                    })
                    .await
                    .expect("serve");
            });
        });

        let addr = addr_rx.recv().expect("server failed to start");
        TestServer {
            url: format!("http://{addr}"),
            root,
            stop: Some(stop),
            thread: Some(thread),
        }
    }

    fn connect(&self) -> Client {
        Client::connect(&self.url, ClientConfig::default()).expect("connect")
    }

    fn root(&self) -> &Path {
        self.root.path()
    }

    /// Write a `.npy` of `f32` counting up from zero.
    fn write_npy(&self, name: &str, shape: &[usize]) -> PathBuf {
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

/// The class of a server error, or a failure if the call did not fail.
fn error_class(result: Result<impl std::fmt::Debug, ClientError>) -> ErrorClass {
    match result {
        Ok(value) => panic!("expected an error, got {value:?}"),
        Err(err) => err
            .class()
            .unwrap_or_else(|| panic!("expected a server error, got {err}")),
    }
}

#[test]
fn metadata_makes_the_round_trip() {
    let server = TestServer::start();
    server.write_npy("ocean.npy", &[1000, 200]);

    let client = server.connect();
    let handle = client.open("ocean.npy").expect("open");

    // The hierarchy v1 exposed: a root group holding one dataset.
    assert_eq!(client.get_item(handle, "/").expect("root"), Item::Group);
    let Item::Dataset(info) = client.get_item(handle, "array").expect("array") else {
        panic!("array must be a dataset");
    };
    assert_eq!(info.dtype, DType::Float32);
    assert_eq!(info.shape, vec![1000, 200]);
    assert_eq!(info.num_elements(), 200_000);

    let children = client.list_children(handle, "/").expect("children");
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].0, "array");
    assert_eq!(children[0].1, Item::Dataset(info));

    client.close(handle).expect("close");
    client.disconnect().expect("disconnect");
}

#[test]
fn a_session_reports_what_the_server_granted() {
    let server = TestServer::start();
    let client = server.connect();
    let session = client.session();

    assert_eq!(session.id.len(), 16);
    assert_eq!(session.token.len(), 16);
    assert_eq!(session.granted_streams, client.config().streams);
    assert_eq!(session.protocol_version, aex_client::PROTOCOL_VERSION);
    assert_eq!(session.default_chunk_bytes, 4 * 1024 * 1024);
    assert_eq!(session.max_fetch_bytes, 16 * 1024 * 1024);
    // RAW and EXACT, and nothing else, in this release.
    assert_eq!(session.supported_codecs, 1);
    assert_eq!(session.supported_encodings, 1);

    // The server advertises no host, so the client keeps the one it dialled.
    assert_eq!(session.data_endpoint, ("127.0.0.1".to_string(), DATA_PORT));
}

#[test]
fn the_server_caps_the_streams_it_grants() {
    let server = TestServer::start_with(|config| config.limits.max_streams_per_session = 2);
    let client = Client::connect(
        &server.url,
        ClientConfig {
            streams: 16,
            ..ClientConfig::default()
        },
    )
    .expect("connect");
    assert_eq!(client.session().granted_streams, 2);
}

#[test]
fn every_supported_dtype_survives_the_round_trip() {
    let server = TestServer::start();
    let client = server.connect();

    // Written by hand rather than through npyz's writer, which only serialises
    // the dtypes it has impls for.
    let path = server.root().join("dtypes.npy");
    for (descr, expected) in [
        ("|b1", DType::Bool),
        ("|i1", DType::Int8),
        ("<i2", DType::Int16),
        ("<i4", DType::Int32),
        ("<i8", DType::Int64),
        ("|u1", DType::Uint8),
        ("<u2", DType::Uint16),
        ("<u4", DType::Uint32),
        ("<u8", DType::Uint64),
        ("<f4", DType::Float32),
        ("<f8", DType::Float64),
        ("<f2", DType::Float16),
        ("<c8", DType::Complex64),
        ("<c16", DType::Complex128),
    ] {
        let itemsize = expected.itemsize() as usize;
        write_raw_npy(&path, descr, &[3], &vec![0u8; 3 * itemsize]);

        let handle = client.open("dtypes.npy").expect("open");
        let Item::Dataset(info) = client.get_item(handle, "array").expect("array") else {
            panic!("{descr} must be a dataset");
        };
        assert_eq!(info.dtype, expected, "{descr} did not survive");
        assert_eq!(info.shape, vec![3]);
        client.close(handle).expect("close");
    }
}

#[test]
fn a_file_can_be_opened_by_an_explicit_format() {
    let server = TestServer::start();
    server.write_npy("data.bin", &[4]);
    let client = server.connect();

    // Without an extension to go by, the client has to say which format it is.
    assert_eq!(error_class(client.open("data.bin")), ErrorClass::Request);
    let handle = client.open_as("data.bin", "npy").expect("open as npy");
    assert!(matches!(
        client.get_item(handle, "array"),
        Ok(Item::Dataset(_))
    ));
    // Nothing else is served yet.
    assert_eq!(
        error_class(client.open_as("data.bin", "hdf5")),
        ErrorClass::Request
    );
}

#[test]
fn files_are_independent_of_each_other() {
    let server = TestServer::start();
    server.write_npy("a.npy", &[10]);
    server.write_npy("sub/b.npy", &[2, 3]);

    let client = server.connect();
    let a = client.open("a.npy").expect("open a");
    let b = client.open("sub/b.npy").expect("open b");
    assert_ne!(a, b);

    let shape = |handle| match client.get_item(handle, "array").unwrap() {
        Item::Dataset(info) => info.shape,
        Item::Group => panic!("expected a dataset"),
    };
    assert_eq!(shape(a), vec![10]);
    assert_eq!(shape(b), vec![2, 3]);

    // Closing one leaves the other open.
    client.close(a).expect("close a");
    assert_eq!(shape(b), vec![2, 3]);
    assert_eq!(
        error_class(client.get_item(a, "array")),
        ErrorClass::Request
    );
}

#[test]
fn a_handle_belongs_to_the_session_that_opened_it() {
    let server = TestServer::start();
    server.write_npy("a.npy", &[4]);

    let first = server.connect();
    let second = server.connect();
    assert_ne!(first.session().id, second.session().id);

    let handle = first.open("a.npy").expect("open");
    assert!(first.get_item(handle, "array").is_ok());
    assert_eq!(
        error_class(second.get_item(handle, "array")),
        ErrorClass::Request
    );
}

#[test]
fn bad_requests_are_reported_as_such() {
    let server = TestServer::start();
    server.write_npy("a.npy", &[4]);
    let client = server.connect();

    // A file that is not there.
    assert_eq!(error_class(client.open("absent.npy")), ErrorClass::Request);
    // A file outside the data roots.
    let outside = tempfile::tempdir().unwrap();
    let secret = outside.path().join("secret.npy");
    std::fs::write(&secret, b"x").unwrap();
    assert_eq!(
        error_class(client.open(secret.to_str().unwrap())),
        ErrorClass::Request
    );
    // A file that is not a .npy at all.
    std::fs::write(server.root().join("junk.npy"), b"not an npy").unwrap();
    assert_eq!(
        error_class(client.open("junk.npy")),
        ErrorClass::Permanent,
        "a corrupt file does not get better on a retry"
    );

    let handle = client.open("a.npy").expect("open");
    // A name no item has.
    assert_eq!(
        error_class(client.get_item(handle, "temperature")),
        ErrorClass::Request
    );
    // A dataset has no children.
    assert_eq!(
        error_class(client.list_children(handle, "array")),
        ErrorClass::Request
    );
    // A handle that was never issued.
    let unknown = client.open("a.npy").expect("open");
    client.close(unknown).expect("close");
    assert_eq!(
        error_class(client.get_item(unknown, "array")),
        ErrorClass::Request
    );
    assert_eq!(error_class(client.close(unknown)), ErrorClass::Request);
}

#[test]
fn requests_after_a_disconnect_are_refused() {
    let server = TestServer::start();
    server.write_npy("a.npy", &[4]);

    let client = server.connect();
    let handle = client.open("a.npy").expect("open");
    let session_id = client.session().id.clone();
    client.disconnect().expect("disconnect");

    // The session is gone, so its handles are too.
    let client = server.connect();
    assert_ne!(client.session().id, session_id);
    assert_eq!(
        error_class(client.get_item(handle, "array")),
        ErrorClass::Request
    );
}

#[test]
fn the_server_refuses_more_sessions_than_it_allows() {
    let server = TestServer::start_with(|config| config.limits.max_sessions = 1);
    let first = server.connect();

    let err = Client::connect(&server.url, ClientConfig::default()).unwrap_err();
    assert_eq!(
        err.class(),
        Some(ErrorClass::Transient),
        "a full server is worth retrying: {err}"
    );
    assert!(err.is_retryable());

    // Room again once the first client leaves.
    first.disconnect().expect("disconnect");
    server.connect();
}

#[test]
fn connecting_to_nothing_fails_without_hanging() {
    // Port 1 on loopback has nothing behind it; the connect is refused rather
    // than left to the timeout.
    let err = Client::connect("http://127.0.0.1:1", ClientConfig::default()).unwrap_err();
    assert!(err.class().is_none(), "a transport failure has no class");
    assert!(!err.is_retryable());
}

/// Write a `.npy` with an arbitrary `descr` and raw payload.
///
/// npyz's writer only writes the dtypes it can serialise; this covers the rest.
fn write_raw_npy(path: &Path, descr: &str, shape: &[u64], payload: &[u8]) {
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
