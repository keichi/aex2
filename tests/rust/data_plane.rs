//! The data plane handshake, spoken by hand over a socket.
//!
//! Going through `aex-wire` rather than through the client is deliberate: what
//! these pin down is what the server does with a connection that is *not* well
//! behaved, which a correct client cannot produce.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use aex_client::{ClientConfig, Index};
use aex_core::ErrorClass;
use aex_wire::{
    read_frame_header, write_frame, FrameHeader, FrameType, Hello, Ready, HELLO_LEN, READY_LEN,
};

#[path = "support.rs"]
mod support;

use support::{Proxy, TestServer};

/// Connect to the data plane, with a timeout so a hung test fails rather than
/// hangs.
fn dial(server: &TestServer) -> TcpStream {
    let stream = TcpStream::connect(server.data_addr).expect("connect to the data plane");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("read timeout");
    stream
}

/// Send a `HELLO` and read the answer.
fn shake_hands(stream: &mut TcpStream, hello: &Hello) -> Ready {
    stream.write_all(&hello.encode()).expect("send hello");
    let mut bytes = [0u8; READY_LEN];
    stream.read_exact(&mut bytes).expect("read ready");
    Ready::decode(&bytes).expect("decode ready")
}

#[test]
fn a_connection_with_the_session_token_is_accepted() {
    let server = TestServer::start();
    let client = server.connect();
    let session = client.session();

    let mut stream = dial(&server);
    let hello = Hello::new(
        session.id.clone().try_into().expect("16 bytes"),
        session.token.clone().try_into().expect("16 bytes"),
    );
    let ready = shake_hands(&mut stream, &hello);
    assert!(ready.is_accepted(), "{ready:?}");
    assert_eq!(ready.version, aex_wire::PROTOCOL_VERSION);
}

#[test]
fn a_connection_without_the_right_token_is_refused() {
    let server = TestServer::start();
    let client = server.connect();
    let session = client.session();
    let id: [u8; 16] = session.id.clone().try_into().unwrap();
    let token: [u8; 16] = session.token.clone().try_into().unwrap();

    // The right session, the wrong token.
    let mut wrong_token = token;
    wrong_token[0] ^= 1;
    let ready = shake_hands(&mut dial(&server), &Hello::new(id, wrong_token));
    assert_eq!(ready.status, ErrorClass::Auth);

    // A session that was never issued.
    let ready = shake_hands(&mut dial(&server), &Hello::new([9; 16], token));
    assert_eq!(ready.status, ErrorClass::Auth);

    // A session that has ended takes its data connections with it.
    client.disconnect().expect("disconnect");
    let ready = shake_hands(&mut dial(&server), &Hello::new(id, token));
    assert_eq!(ready.status, ErrorClass::Auth);
}

#[test]
fn something_that_is_not_a_handshake_is_refused_as_a_protocol_error() {
    let server = TestServer::start();
    let _client = server.connect();

    let mut stream = dial(&server);
    let mut bytes = [0u8; HELLO_LEN];
    bytes[0..8].copy_from_slice(b"GET / HT");
    stream.write_all(&bytes).expect("send");

    let mut reply = [0u8; READY_LEN];
    stream.read_exact(&mut reply).expect("read ready");
    assert_eq!(
        Ready::decode(&reply).unwrap().status,
        ErrorClass::Protocol,
        "a peer that is not speaking AEX still gets an answer before the hang-up"
    );
}

#[test]
fn a_client_speaking_another_version_is_refused() {
    let server = TestServer::start();
    let client = server.connect();
    let session = client.session();

    let mut hello = Hello::new(
        session.id.clone().try_into().unwrap(),
        session.token.clone().try_into().unwrap(),
    );
    hello.version = aex_wire::PROTOCOL_VERSION + 1;
    let ready = shake_hands(&mut dial(&server), &hello);
    // Refused here rather than showing up later as a frame nobody can parse.
    assert_eq!(ready.status, ErrorClass::Protocol);
}

#[test]
fn a_session_opens_no_more_connections_than_it_was_granted() {
    let server = TestServer::start_with(|config| config.limits.max_streams_per_session = 2);
    // The client opens one of the two as it connects.
    let client = server.connect();
    let session = client.session();
    assert_eq!(session.granted_streams, 2);
    let hello = Hello::new(
        session.id.clone().try_into().unwrap(),
        session.token.clone().try_into().unwrap(),
    );

    let mut second = dial(&server);
    assert!(shake_hands(&mut second, &hello).is_accepted());

    let ready = shake_hands(&mut dial(&server), &hello);
    // Temporary, not a refusal: it clears as the others close.
    assert_eq!(ready.status, ErrorClass::Transient);

    // Closing one makes room again, once the server's thread notices.
    drop(second);
    let freed = (0..100).any(|_| {
        std::thread::sleep(Duration::from_millis(20));
        shake_hands(&mut dial(&server), &hello).is_accepted()
    });
    assert!(freed, "a closed connection must free its slot");
}

#[test]
fn a_ping_is_answered_with_a_pong() {
    let server = TestServer::start();
    let client = server.connect();
    let session = client.session();

    let mut stream = dial(&server);
    let hello = Hello::new(
        session.id.clone().try_into().unwrap(),
        session.token.clone().try_into().unwrap(),
    );
    assert!(shake_hands(&mut stream, &hello).is_accepted());

    write_frame(&mut stream, &FrameHeader::bare(FrameType::Ping), &[]).expect("ping");
    let pong = read_frame_header(&mut stream).expect("pong");
    assert_eq!(pong.frame_type, FrameType::Pong);
    assert_eq!(pong.wire_len, 0);
}

#[test]
fn a_frame_only_a_server_may_send_ends_the_connection() {
    let server = TestServer::start();
    let client = server.connect();
    let session = client.session();

    let mut stream = dial(&server);
    let hello = Hello::new(
        session.id.clone().try_into().unwrap(),
        session.token.clone().try_into().unwrap(),
    );
    assert!(shake_hands(&mut stream, &hello).is_accepted());

    // A DATA frame from a client means the two implementations disagree about
    // who says what, so the server says so and hangs up.
    write_frame(&mut stream, &FrameHeader::data(1, 0, 0), &[]).expect("send");
    let header = read_frame_header(&mut stream).expect("error frame");
    assert_eq!(header.frame_type, FrameType::Error);
    assert_eq!(header.request_id, 0, "no fetch is to blame for this");

    let mut payload = vec![0u8; header.wire_len as usize];
    stream.read_exact(&mut payload).expect("payload");
    let error = aex_wire::ErrorPayload::decode(&payload).expect("decode");
    assert_eq!(error.class, ErrorClass::Protocol);

    // And the connection is gone.
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).expect("read to end");
    assert!(rest.is_empty());
}

#[test]
fn a_fetch_for_a_transfer_that_does_not_exist_is_reported_as_a_plan_error() {
    let server = TestServer::start();
    let client = server.connect();
    let session = client.session();

    let mut stream = dial(&server);
    let hello = Hello::new(
        session.id.clone().try_into().unwrap(),
        session.token.clone().try_into().unwrap(),
    );
    assert!(shake_hands(&mut stream, &hello).is_accepted());

    write_frame(&mut stream, &FrameHeader::fetch(4242, 0, 1024), &[0u8; 16]).expect("fetch");
    let header = read_frame_header(&mut stream).expect("error frame");
    assert_eq!(header.frame_type, FrameType::Error);
    // The failed fetch is named, so that just that chunk can be retried.
    assert_eq!(header.request_id, 4242);
    assert_eq!(header.logical_len, 1024);

    let mut payload = vec![0u8; header.wire_len as usize];
    stream.read_exact(&mut payload).expect("payload");
    let error = aex_wire::ErrorPayload::decode(&payload).expect("decode");
    assert_eq!(
        error.class,
        ErrorClass::Plan,
        "the client re-prepares on this class, and on no other"
    );

    // Not fatal: the connection stays up for the next fetch.
    write_frame(&mut stream, &FrameHeader::bare(FrameType::Ping), &[]).expect("ping");
    assert_eq!(
        read_frame_header(&mut stream).expect("pong").frame_type,
        FrameType::Pong
    );
}

#[test]
fn the_whole_array_comes_back() {
    let server = TestServer::start();
    // 800 KB, well past the inline limit, so this goes over the data plane.
    let expected = server.write_counting_npy("ocean.npy", &[1000, 200]);

    let client = server.connect();
    let handle = client.open("ocean.npy").expect("open");
    let array = client
        .read_selection_as::<f32>(handle, "array", &[])
        .expect("read");

    assert_eq!(array.shape, vec![1000, 200]);
    assert_eq!(array.data, expected);
    assert!(!array.transfer.inline);
    assert_eq!(array.transfer.bytes, 800_000);
    assert_eq!(array.transfer.streams, 1);
    assert_eq!(array.transfer.retries, 0);
}

#[test]
fn a_small_selection_comes_back_with_its_plan() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("small.npy", &[64]);

    let client = server.connect();
    let handle = client.open("small.npy").expect("open");
    let array = client
        .read_selection_as::<f32>(handle, "array", &[])
        .expect("read");

    assert_eq!(array.data, expected);
    // One round trip, and the data plane is not touched at all: without this
    // path a small read would cost two where v1 needed one.
    assert!(array.transfer.inline);
    assert_eq!(array.transfer.chunks, 0);
    assert_eq!(array.transfer.streams, 0);
}

#[test]
fn a_transfer_is_cut_into_chunks_by_the_client() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("ocean.npy", &[1000, 200]);

    // The server states a ceiling and a recommendation; how to cut the stream
    // up is the client's decision, and this is it being made differently.
    let client = server.connect_with(ClientConfig {
        chunk_bytes: 64 * 1024,
        ..ClientConfig::default()
    });
    let handle = client.open("ocean.npy").expect("open");
    let array = client
        .read_selection_as::<f32>(handle, "array", &[])
        .expect("read");

    assert_eq!(array.data, expected);
    assert_eq!(array.transfer.chunks, 800_000u32.div_ceil(64 * 1024));
    assert_eq!(array.transfer.bytes, 800_000);
}

#[test]
fn a_fetch_larger_than_the_server_allows_is_cut_down_to_it() {
    let server = TestServer::start_with(|config| {
        config.transfer.default_chunk_bytes = 128 * 1024;
        config.transfer.max_fetch_bytes = 128 * 1024;
    });
    let expected = server.write_counting_npy("ocean.npy", &[1000, 200]);

    // A client asking for more than the server will serve has to notice; the
    // ceiling comes back in the reply to Connect for exactly this reason.
    let client = server.connect_with(ClientConfig {
        chunk_bytes: 16 * 1024 * 1024,
        ..ClientConfig::default()
    });
    assert_eq!(client.session().max_fetch_bytes, 128 * 1024);

    let handle = client.open("ocean.npy").expect("open");
    let array = client
        .read_selection_as::<f32>(handle, "array", &[])
        .expect("read");
    assert_eq!(array.data, expected);
    assert_eq!(array.transfer.chunks, 800_000u32.div_ceil(128 * 1024));
}

#[test]
fn a_selection_comes_back_as_numpy_would_have_taken_it() {
    let server = TestServer::start();
    // Rows of 2000 f32, so one row is 8 KB and three rows are past the inline
    // limit: the cases below run over both paths.
    let expected = server.write_counting_npy("grid.npy", &[64, 2000]);
    let row = 2000usize;

    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");

    let cases: Vec<(Vec<Index>, Vec<u64>, std::ops::Range<usize>)> = vec![
        (vec![], vec![64, 2000], 0..64 * row),
        (vec![Index::Single(0)], vec![2000], 0..row),
        (vec![Index::Single(-1)], vec![2000], 63 * row..64 * row),
        (
            vec![Index::range(10, 40)],
            vec![30, 2000],
            10 * row..40 * row,
        ),
        (
            vec![Index::Fancy(vec![8, 9, 10])],
            vec![3, 2000],
            8 * row..11 * row,
        ),
        (
            vec![Index::Ellipsis, Index::NewAxis],
            vec![64, 2000, 1],
            0..64 * row,
        ),
        (
            vec![Index::Single(2), Index::Single(3)],
            vec![],
            2 * row + 3..2 * row + 4,
        ),
        (vec![Index::range(5, 5)], vec![0, 2000], 0..0),
    ];

    for (indices, shape, range) in cases {
        let array = client
            .read_selection_as::<f32>(handle, "array", &indices)
            .unwrap_or_else(|e| panic!("{indices:?}: {e}"));
        assert_eq!(array.shape, shape, "{indices:?}");
        assert_eq!(array.data, expected[range], "{indices:?}");
    }
}

#[test]
fn a_selection_that_is_not_one_run_is_gathered() {
    let server = TestServer::start();
    let (rows, cols) = (64u64, 2000u64);
    server.write_counting_npy("grid.npy", &[rows as usize, cols as usize]);
    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");
    let at = |r: u64, c: u64| (r * cols + c) as f32;

    // Each is checked against the source element by element. Some are small
    // enough to come back inline and some go over the data plane; both read
    // through the same layout.
    let every_other_row: Vec<f32> = (0..rows)
        .step_by(2)
        .flat_map(|r| (0..cols).map(move |c| at(r, c)))
        .collect();
    let reversed_thirds: Vec<f32> = (0..rows)
        .flat_map(|r| (0..cols).rev().step_by(3).map(move |c| at(r, c)))
        .collect();
    let rows_159: Vec<f32> = [1, 5, 9]
        .into_iter()
        .flat_map(|r| (0..cols).map(move |c| at(r, c)))
        .collect();
    let cases: Vec<(Vec<Index>, Vec<u64>, Vec<f32>)> = vec![
        (
            vec![Index::full(), Index::range(0, 100)],
            vec![rows, 100],
            (0..rows)
                .flat_map(|r| (0..100).map(move |c| at(r, c)))
                .collect(),
        ),
        (
            vec![Index::Slice {
                start: None,
                stop: None,
                step: Some(2),
            }],
            vec![rows / 2, cols],
            every_other_row,
        ),
        (
            vec![
                Index::full(),
                Index::Slice {
                    start: None,
                    stop: None,
                    step: Some(-3),
                },
            ],
            vec![rows, cols.div_ceil(3)],
            reversed_thirds,
        ),
        (vec![Index::Fancy(vec![1, 5, 9])], vec![3, cols], rows_159),
        (
            vec![Index::Fancy(vec![0, 63]), Index::Fancy(vec![1999, 7])],
            vec![2],
            vec![at(0, 1999), at(63, 7)],
        ),
    ];

    for (indices, shape, expected) in cases {
        let array = client
            .read_selection_as::<f32>(handle, "array", &indices)
            .unwrap_or_else(|e| panic!("{indices:?}: {e}"));
        assert_eq!(array.shape, shape, "{indices:?}");
        assert!(array.data == expected, "{indices:?}");
    }

    // A selection numpy would refuse is still refused.
    let err = client
        .read_selection(handle, "array", &[Index::Single(64)])
        .unwrap_err();
    assert_eq!(err.class(), Some(ErrorClass::Request), "{err}");
}

#[test]
fn reading_into_a_buffer_writes_exactly_that_buffer() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("ocean.npy", &[500, 200]);
    let client = server.connect();
    let handle = client.open("ocean.npy").expect("open");

    let mut bytes = vec![0u8; expected.len() * 4];
    let result = client
        .read_selection_into(handle, "array", &[], &mut bytes)
        .expect("read");
    assert_eq!(result.bytes, bytes.len() as u64);

    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);

    // A buffer of the wrong size is the caller's mistake, and is caught before
    // anything is sent.
    let mut wrong = vec![0u8; 8];
    let err = client
        .read_selection_into(handle, "array", &[], &mut wrong)
        .unwrap_err();
    assert!(
        matches!(err, aex_client::ClientError::BadRequest(_)),
        "{err}"
    );
}

#[test]
fn a_plan_says_what_to_allocate_before_the_fill() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("ocean.npy", &[500, 200]);
    let client = server.connect();
    let handle = client.open("ocean.npy").expect("open");

    // One inline, one over the data plane.
    for indices in [vec![Index::Single(3)], vec![Index::range(10, 400)]] {
        let plan = client.prepare(handle, "array", &indices).expect("prepare");
        assert_eq!(plan.dtype, aex_client::DType::Float32);
        let mut bytes = vec![0u8; plan.total_bytes as usize];
        client
            .fill(&plan, handle, "array", &indices, &mut bytes)
            .expect("fill");

        let first = if plan.shape == [200] { 600 } else { 2000 };
        assert_eq!(plan.shape.iter().product::<u64>() * 4, plan.total_bytes);
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(
            values == expected[first..first + values.len()],
            "{indices:?}"
        );

        // A plan can be filled again, and only into a buffer of its length.
        client
            .fill(&plan, handle, "array", &indices, &mut bytes)
            .expect("fill again");
        let err = client
            .fill(&plan, handle, "array", &indices, &mut bytes[1..])
            .unwrap_err();
        assert!(
            matches!(err, aex_client::ClientError::BadRequest(_)),
            "{err}"
        );
    }
}

#[test]
fn reading_as_the_wrong_type_is_refused() {
    let server = TestServer::start();
    server.write_counting_npy("ocean.npy", &[500, 200]);
    let client = server.connect();
    let handle = client.open("ocean.npy").expect("open");

    let err = client
        .read_selection_as::<f64>(handle, "array", &[])
        .unwrap_err();
    assert!(
        matches!(err, aex_client::ClientError::BadRequest(_)),
        "{err}"
    );

    // The raw bytes are always available, whatever the dtype.
    let array = client.read_selection(handle, "array", &[]).expect("read");
    assert_eq!(array.dtype, aex_core::DType::Float32);
    assert_eq!(array.bytes.len(), 500 * 200 * 4);
}

#[test]
fn one_session_reads_many_selections_over_the_same_connection() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("grid.npy", &[64, 2000]);
    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");

    // The connection carries no state between transfers, so the third read is
    // the same as the first.
    for row in [0usize, 7, 63] {
        let array = client
            .read_selection_as::<f32>(handle, "array", &[Index::Single(row as i64)])
            .expect("read");
        assert_eq!(array.data, expected[row * 2000..(row + 1) * 2000]);
    }
}

#[test]
fn the_synthetic_backend_is_offered_only_when_it_was_asked_for() {
    // It serves data that was never stored anywhere and takes no path under the
    // data roots, so a server that was not told to offer it must not.
    let server = TestServer::start();
    let client = server.connect();
    let err = client.open_as("uint8:1024", "null").unwrap_err();
    assert_eq!(err.class(), Some(ErrorClass::Request), "{err}");
    assert!(err.to_string().contains("enabled"), "{err}");
}

#[test]
fn the_synthetic_backend_serves_the_pattern_it_promises() {
    let server = TestServer::start_with(|config| config.enable_null_backend = true);
    let client = server.connect();

    // Large enough to go over the data plane rather than back with the plan.
    let handle = client.open_as("uint8:1048576", "null").expect("open");
    let array = client.read_selection(handle, "array", &[]).expect("read");
    assert_eq!(array.shape, vec![1048576]);
    assert!(!array.transfer.inline);

    // The byte at position p is p as u8, so a chunk that landed in the wrong
    // place is a mismatch rather than a plausible number.
    let wrong = array.bytes.iter().enumerate().find(|(i, &b)| b != *i as u8);
    assert_eq!(wrong.map(|(i, _)| i), None);

    // A selection of it reads from where it starts, as any other dataset would.
    let rows = client
        .read_selection(handle, "array", &[Index::range(1000, 3000)])
        .expect("read");
    assert_eq!(rows.shape, vec![2000]);
    let expected: Vec<u8> = (1000u64..3000).map(|p| p as u8).collect();
    assert_eq!(rows.bytes, expected);

    // And its metadata crosses the wire like anything else.
    let handle = client.open_as("float32:100x200", "null").expect("open");
    let aex_client::Item::Dataset(info) = client.get_item(handle, "array").expect("item") else {
        panic!("expected a dataset");
    };
    assert_eq!(info.dtype, aex_core::DType::Float32);
    assert_eq!(info.shape, vec![100, 200]);

    // A specification that makes no sense is a bad request, not a crash.
    assert_eq!(
        client.open_as("not a dataset", "null").unwrap_err().class(),
        Some(ErrorClass::Request)
    );
}

#[test]
fn the_data_plane_can_be_reached_at_a_pinned_endpoint() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("ocean.npy", &[1000, 200]);
    let proxy = Proxy::start(server.data_addr, None);

    // What a tunnel looks like: the server advertises one port, the client has
    // to use another.
    let client = server.connect_with(ClientConfig {
        data_endpoint: Some(proxy.endpoint()),
        ..ClientConfig::default()
    });
    assert_eq!(client.session().data_endpoint.1, proxy.addr.port());
    let handle = client.open("ocean.npy").expect("open");
    let array = client
        .read_selection_as::<f32>(handle, "array", &[])
        .expect("read");
    assert_eq!(array.data, expected);
    assert!(proxy.accepted() >= 1);
}
