//! Error-bounded transfers, end to end.
//!
//! What these pin down is the bargain the encoding makes: every element comes
//! back within the bound the client asked for, and the plan says truthfully
//! what the server did — including when the answer is that it did nothing.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use aex_client::{ClientConfig, Index, Selection, TransferResult};
use aex_core::{Encoding, ErrorClass, QualitySpec};

#[path = "support.rs"]
mod support;

use support::TestServer;

fn error_bound(abs: f64) -> QualitySpec {
    QualitySpec {
        encoding: Encoding::ErrorBound,
        abs_error_bound: Some(abs),
        ..QualitySpec::default()
    }
}

/// Read a selection at a given quality, and say what the server applied.
fn read(
    server: &TestServer,
    name: &str,
    shape: &[usize],
    indices: &[Index],
    quality: &QualitySpec,
) -> (QualitySpec, Vec<f32>, TransferResult) {
    read_with(
        server,
        name,
        shape,
        indices,
        quality,
        ClientConfig::default(),
    )
}

fn read_with(
    server: &TestServer,
    name: &str,
    shape: &[usize],
    indices: &[Index],
    quality: &QualitySpec,
    config: ClientConfig,
) -> (QualitySpec, Vec<f32>, TransferResult) {
    server.write_npy(name, shape);
    let client = server.connect_with(config);
    let handle = client.open(name).expect("open");
    let selection = Selection {
        quality,
        ..Selection::exact(handle, "array", indices)
    };
    let plan = client.prepare_selection(&selection).expect("prepare");
    let mut bytes = vec![0u8; plan.total_bytes as usize];
    let result = client
        .fill_many(
            std::slice::from_ref(&plan),
            std::slice::from_ref(&selection),
            &mut [&mut bytes],
        )
        .expect("fill");
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    (plan.applied_quality, values, result)
}

/// What `write_npy` puts in a file: f32 counting up from zero.
fn expected(count: usize) -> Vec<f32> {
    (0..count).map(|i| i as f32).collect()
}

fn worst(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0, f64::max)
}

#[test]
fn every_element_arrives_within_the_bound_it_asked_for() {
    let server = TestServer::start();
    // Large enough that the transfer goes over the data plane rather than
    // inline, and several read pieces wide.
    let shape = [2048, 512];
    for eps in [1e-3, 1.0, 64.0] {
        let name = format!("bounded-{eps}.npy");
        let (applied, values, _) = read(&server, &name, &shape, &[], &error_bound(eps));
        assert_eq!(applied.encoding, Encoding::ErrorBound);
        assert_eq!(applied.abs_error_bound, Some(eps));

        let want = expected(shape[0] * shape[1]);
        let seen = worst(&values, &want);
        assert!(seen <= eps, "worst error {seen:e} over a bound of {eps:e}");
    }
}

#[test]
fn the_bound_holds_across_every_connection_and_chunk() {
    let server = TestServer::start();
    let shape = [4096, 256];
    let eps = 1e-2;
    // Several connections, small chunks: every block boundary and every
    // stealing decision gets exercised.
    let config = ClientConfig {
        streams: 4,
        chunk_bytes: 256 * 1024,
        ..ClientConfig::default()
    };
    let (applied, values, _) = read_with(
        &server,
        "parallel.npy",
        &shape,
        &[],
        &error_bound(eps),
        config,
    );
    assert_eq!(applied.encoding, Encoding::ErrorBound);
    let seen = worst(&values, &expected(shape[0] * shape[1]));
    assert!(seen <= eps, "worst error {seen:e} over a bound of {eps:e}");
}

#[test]
fn a_gathered_selection_is_bounded_too() {
    let server = TestServer::start();
    let shape = [2048, 512];
    let eps = 1e-2;
    // Every other column: the logical stream is a resampled grid, and the
    // blocks are cut against that grid rather than against the file.
    let indices = [
        Index::Slice {
            start: None,
            stop: None,
            step: None,
        },
        Index::Slice {
            start: Some(0),
            stop: Some(256),
            step: None,
        },
    ];
    let (applied, values, _) = read(&server, "gathered.npy", &shape, &indices, &error_bound(eps));
    assert_eq!(applied.encoding, Encoding::ErrorBound);

    let want: Vec<f32> = (0..shape[0])
        .flat_map(|r| (0..256).map(move |c| (r * shape[1] + c) as f32))
        .collect();
    let seen = worst(&values, &want);
    assert!(seen <= eps, "worst error {seen:e} over a bound of {eps:e}");
}

#[test]
fn a_quality_the_server_cannot_apply_comes_back_as_exact() {
    let server = TestServer::start();
    let shape = [1024, 512];

    // A bound relative to the value range would have to mean the range of the
    // whole selection, and a block only ever sees its own.
    let relative = QualitySpec {
        encoding: Encoding::ErrorBound,
        rel_error_bound: Some(1e-3),
        ..QualitySpec::default()
    };
    let (applied, values, _) = read(&server, "relative.npy", &shape, &[], &relative);
    assert_eq!(applied.encoding, Encoding::Exact);
    assert_eq!(values, expected(shape[0] * shape[1]), "exact means exact");

    // And an encoding nothing implements.
    let subsampled = QualitySpec {
        encoding: Encoding::Subsample,
        subsample_step: vec![2, 2],
        ..QualitySpec::default()
    };
    let (applied, _, _) = read(&server, "subsampled.npy", &shape, &[], &subsampled);
    assert_eq!(applied.encoding, Encoding::Exact);
}

#[test]
fn an_exact_transfer_is_untouched_by_any_of_this() {
    let server = TestServer::start();
    let shape = [1024, 512];
    let (applied, values, result) = read(&server, "exact.npy", &shape, &[], &QualitySpec::exact());
    assert_eq!(applied.encoding, Encoding::Exact);
    assert_eq!(values, expected(shape[0] * shape[1]));
    // Raw frames carry exactly what they hold, so nothing is saved and nothing
    // is lost.
    assert_eq!(result.wire_bytes, result.bytes);
}

#[test]
fn an_error_bounded_transfer_sends_fewer_bytes_than_it_delivers() {
    // Without this, every test here would still pass if the server quietly
    // sent raw bytes and called them encoded.
    let server = TestServer::start();
    let shape = [2048, 512];
    let (applied, _, result) = read(&server, "smaller.npy", &shape, &[], &error_bound(1.0));
    assert_eq!(applied.encoding, Encoding::ErrorBound);
    assert!(!result.inline, "the transfer has to go over the data plane");
    assert!(
        result.wire_bytes < result.bytes,
        "{} bytes on the wire for {} bytes of stream",
        result.wire_bytes,
        result.bytes
    );
    assert!(result.compression_ratio() > 1.0);
}

#[test]
fn a_fetch_that_splits_an_element_is_refused() {
    let server = TestServer::start();
    server.write_npy("misaligned.npy", &[2048, 512]);
    let mut session = server.raw_session(1);
    let handle = session.open("misaligned.npy");
    let plan = session.prepare(
        handle,
        "array",
        Some(aex_proto::QualitySpec {
            encoding: aex_proto::Encoding::ErrorBound as i32,
            abs_error_bound: Some(1e-2),
            ..Default::default()
        }),
    );
    assert_eq!(
        plan.applied_quality.as_ref().map(|q| q.encoding),
        Some(aex_proto::Encoding::ErrorBound as i32)
    );

    let mut stream = TcpStream::connect(server.data_addr).expect("connect to the data plane");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("read timeout");
    stream
        .write_all(&session.hello.encode())
        .expect("send hello");
    let mut ready = [0u8; aex_wire::READY_LEN];
    stream.read_exact(&mut ready).expect("read ready");
    assert!(aex_wire::Ready::decode(&ready)
        .expect("ready")
        .is_accepted());

    // A block holds whole elements or it holds nothing, so a fetch that names
    // half of one cannot be answered. A correct client never sends this.
    let ticket: [u8; 16] = plan.ticket.clone().try_into().expect("16 bytes");
    let fetch = aex_wire::FrameHeader::fetch(plan.request_id, 2, plan.total_bytes - 2);
    aex_wire::write_frame(&mut stream, &fetch, &ticket).expect("fetch");

    let header = aex_wire::read_frame_header(&mut stream).expect("a reply");
    assert_eq!(header.frame_type, aex_wire::FrameType::Error);
    let mut payload = vec![0u8; header.wire_len as usize];
    stream.read_exact(&mut payload).expect("payload");
    let error = aex_wire::ErrorPayload::decode(&payload).expect("decode");
    assert_eq!(error.class, ErrorClass::Request, "{}", error.message);
}
