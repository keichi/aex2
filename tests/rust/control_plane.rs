//! Control plane round trips, with a real server and a real client.

use aex_client::{Client, ClientConfig, Item};
use aex_core::{DType, ErrorClass};

#[path = "support.rs"]
mod support;

use support::{error_class, write_raw_npy, TestServer};

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

    // The server advertises no host, so the client keeps the one it dialled,
    // and the port is the one the data plane really bound.
    assert_eq!(
        session.data_endpoint,
        ("127.0.0.1".to_string(), server.data_addr.port())
    );
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
    assert_eq!(
        error_class(client.open_as("data.bin", "zarr")),
        ErrorClass::Request
    );
    // A format the server knows, but not what the file holds.
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
