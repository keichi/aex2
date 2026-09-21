//! Error-bounded transfers, end to end.
//!
//! What these pin down is the bargain the encoding makes: every element comes
//! back within the bound the client asked for, and the plan says truthfully
//! what the server did — including when the answer is that it did nothing.

// Only the hand-rolled FETCH below speaks to a socket, and it is not built
// without a codec to encode a block with.
#[cfg(any(feature = "sz", feature = "zfp"))]
use std::io::{Read, Write};
#[cfg(any(feature = "sz", feature = "zfp"))]
use std::net::TcpStream;
#[cfg(any(feature = "sz", feature = "zfp"))]
use std::time::Duration;

use aex_client::{ClientConfig, Index, Selection, TransferResult};
#[cfg(any(feature = "sz", feature = "zfp"))]
use aex_core::ErrorClass;
use aex_core::{Codec, Encoding, QualitySpec};

#[path = "support.rs"]
mod support;

use support::TestServer;

fn error_bound(abs: f64, codec: Codec) -> QualitySpec {
    QualitySpec {
        encoding: Encoding::ErrorBound,
        abs_error_bound: Some(abs),
        codec: Some(codec),
        ..QualitySpec::default()
    }
}

/// The error-bounded codecs this build has.
///
/// Every bound these tests check is a promise each codec makes on its own, so
/// each one is put through all of them rather than only the default.
fn codecs() -> Vec<Codec> {
    [Codec::Sz, Codec::Zfp]
        .into_iter()
        .filter(|codec| codec.is_supported())
        .collect()
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
    for codec in codecs() {
        for eps in [1e-3, 1.0, 64.0] {
            let name = format!("bounded-{codec:?}-{eps}.npy");
            let (applied, values, _) = read(&server, &name, &shape, &[], &error_bound(eps, codec));
            assert_eq!(applied.encoding, Encoding::ErrorBound);
            assert_eq!(applied.abs_error_bound, Some(eps));
            assert_eq!(applied.codec, Some(codec), "the codec asked for");

            let want = expected(shape[0] * shape[1]);
            let seen = worst(&values, &want);
            assert!(
                seen <= eps,
                "{codec:?}: worst error {seen:e} over a bound of {eps:e}"
            );
        }
    }
}

#[test]
fn the_bound_holds_across_every_connection_and_chunk() {
    let server = TestServer::start();
    let shape = [4096, 256];
    let eps = 1e-2;
    for codec in codecs() {
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
            &error_bound(eps, codec),
            config,
        );
        assert_eq!(applied.encoding, Encoding::ErrorBound);
        let seen = worst(&values, &expected(shape[0] * shape[1]));
        assert!(
            seen <= eps,
            "{codec:?}: worst error {seen:e} over a bound of {eps:e}"
        );
    }
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
    for codec in codecs() {
        let (applied, values, _) = read(
            &server,
            "gathered.npy",
            &shape,
            &indices,
            &error_bound(eps, codec),
        );
        assert_eq!(applied.encoding, Encoding::ErrorBound);

        let want: Vec<f32> = (0..shape[0])
            .flat_map(|r| (0..256).map(move |c| (r * shape[1] + c) as f32))
            .collect();
        let seen = worst(&values, &want);
        assert!(
            seen <= eps,
            "{codec:?}: worst error {seen:e} over a bound of {eps:e}"
        );
    }
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

    // And a cast that is not one of the two narrowings.
    let widened = QualitySpec {
        encoding: Encoding::DtypeCast,
        cast_dtype: Some(aex_core::DType::Float64),
        ..QualitySpec::default()
    };
    let (applied, _, _) = read(&server, "widened.npy", &shape, &[], &widened);
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
    for codec in codecs() {
        let (applied, _, result) = read(
            &server,
            "smaller.npy",
            &shape,
            &[],
            &error_bound(1.0, codec),
        );
        assert_eq!(applied.encoding, Encoding::ErrorBound);
        assert!(!result.inline, "the transfer has to go over the data plane");
        assert!(
            result.wire_bytes < result.bytes,
            "{codec:?}: {} bytes on the wire for {} bytes of stream",
            result.wire_bytes,
            result.bytes
        );
        assert!(result.compression_ratio() > 1.0);
    }
}

#[test]
fn asking_for_a_codec_is_what_decides_which_one_runs() {
    // Without this, a codec that never reached the request would still pass
    // every test above: whichever one the build defaults to meets all of
    // those bounds on its own.
    let server = TestServer::start();
    let shape = [2048, 512];
    let mut wire: Vec<(Codec, u64)> = Vec::new();
    for codec in codecs() {
        let (applied, _, result) =
            read(&server, "chosen.npy", &shape, &[], &error_bound(1.0, codec));
        assert_eq!(applied.codec, Some(codec));
        wire.push((codec, result.wire_bytes));
    }
    if let [(a, sent_a), (b, sent_b)] = wire[..] {
        assert_ne!(
            sent_a, sent_b,
            "{a:?} and {b:?} put the same {sent_a} bytes on the wire"
        );
    }
}

#[test]
// Needs a server that really applies the bound; with neither codec the plan
// comes back EXACT and there is no encoded block to split.
#[cfg(any(feature = "sz", feature = "zfp"))]
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

#[test]
fn gzip_delivers_the_exact_bytes_over_a_smaller_wire() {
    // The first combination of an exact encoding with a codec, so it is also
    // what pins down that the two are independent of each other.
    let server = TestServer::start();
    let shape = [2048, 512];
    let quality = QualitySpec {
        codec: Some(Codec::Gzip),
        ..QualitySpec::exact()
    };
    let (applied, values, result) = read(&server, "gzipped.npy", &shape, &[], &quality);

    assert_eq!(applied.encoding, Encoding::Exact);
    assert_eq!(applied.codec, Some(Codec::Gzip));
    assert_eq!(values, expected(shape[0] * shape[1]), "exact means exact");
    assert!(!result.inline, "the transfer has to go over the data plane");
    assert!(
        result.wire_bytes < result.bytes,
        "{} wire bytes for {} delivered",
        result.wire_bytes,
        result.bytes
    );
}

#[test]
fn an_error_bounded_codec_asked_for_on_an_exact_transfer_is_not_used() {
    // SZ cannot carry an exact transfer, and swapping in GZIP for it would be
    // answering a question nobody asked. RAW is the honest answer.
    let server = TestServer::start();
    let quality = QualitySpec {
        codec: Some(Codec::Sz),
        ..QualitySpec::exact()
    };
    let (applied, values, result) = read(&server, "exact-sz.npy", &[1024, 512], &[], &quality);
    assert_eq!(applied.encoding, Encoding::Exact);
    assert_eq!(applied.codec, Some(Codec::Raw));
    assert_eq!(values, expected(1024 * 512));
    assert_eq!(result.wire_bytes, result.bytes);
}

#[test]
fn a_cast_halves_the_transfer_and_says_so() {
    // Large enough to go over the data plane in several pieces, so the
    // conversion is exercised at the boundaries between them as well.
    let server = TestServer::start();
    let shape = [2048, 512];
    let quality = QualitySpec {
        encoding: Encoding::DtypeCast,
        cast_dtype: Some(aex_core::DType::Float16),
        ..QualitySpec::exact()
    };
    server.write_npy("cast.npy", &shape);
    let client = server.connect();
    let handle = client.open("cast.npy").expect("open");
    let selection = Selection {
        quality: &quality,
        ..Selection::exact(handle, "array", &[])
    };
    let plan = client.prepare_selection(&selection).expect("prepare");

    assert_eq!(plan.applied_quality.encoding, Encoding::DtypeCast);
    assert_eq!(
        plan.applied_quality.cast_dtype,
        Some(aex_core::DType::Float16)
    );
    assert_eq!(plan.dtype, aex_core::DType::Float16);
    assert_eq!(plan.total_bytes, (shape[0] * shape[1] * 2) as u64);

    let mut bytes = vec![0u8; plan.total_bytes as usize];
    let result = client
        .fill_many(
            std::slice::from_ref(&plan),
            std::slice::from_ref(&selection),
            &mut [&mut bytes],
        )
        .expect("fill");
    // Nothing is compressed, so the wire carries the halved stream and no
    // more: the receiver reads it straight into the array.
    assert_eq!(result.wire_bytes, result.bytes);
    assert_eq!(result.bytes, plan.total_bytes);

    let got: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    let want: Vec<f32> = expected(shape[0] * shape[1])
        .into_iter()
        .map(|v| half::f16::from_f32(v).to_f32())
        .collect();
    assert_eq!(got, want);
}

#[test]
fn a_cast_that_overflows_gives_back_what_the_cast_gives_back() {
    // float16 stops at 65504 and the server does not go looking: reading the
    // whole selection to find out would cost more than the transfer, and the
    // values that arrive say it plainly enough.
    let server = TestServer::start();
    let shape = [256, 512];
    let quality = QualitySpec {
        encoding: Encoding::DtypeCast,
        cast_dtype: Some(aex_core::DType::Float16),
        ..QualitySpec::exact()
    };
    let (applied, _, _) = read(&server, "overflow.npy", &shape, &[], &quality);
    assert_eq!(
        applied.encoding,
        Encoding::DtypeCast,
        "applied all the same"
    );

    server.write_npy("overflow2.npy", &shape);
    let client = server.connect();
    let handle = client.open("overflow2.npy").expect("open");
    let selection = Selection {
        quality: &quality,
        ..Selection::exact(handle, "array", &[])
    };
    let plan = client.prepare_selection(&selection).expect("prepare");
    let mut bytes = vec![0u8; plan.total_bytes as usize];
    client
        .fill_many(
            std::slice::from_ref(&plan),
            std::slice::from_ref(&selection),
            &mut [&mut bytes],
        )
        .expect("fill");
    let got: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    // The fixture counts up past 65504 and everything above it is infinite.
    assert_eq!(got[0], 0.0);
    assert!(got[65504].is_finite());
    assert!(got[70000].is_infinite(), "{}", got[70000]);
}
