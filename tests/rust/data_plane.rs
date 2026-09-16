//! The data plane handshake, spoken by hand over a socket.
//!
//! Going through `aex-wire` rather than through the client is deliberate: what
//! these pin down is what the server does with a connection that is *not* well
//! behaved, which a correct client cannot produce.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use aex_core::ErrorClass;
use aex_wire::{
    read_frame_header, write_frame, FrameHeader, FrameType, Hello, Ready, HELLO_LEN, READY_LEN,
};

#[path = "support.rs"]
mod support;

use support::TestServer;

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
    let client = server.connect();
    let session = client.session();
    assert_eq!(session.granted_streams, 2);
    let hello = Hello::new(
        session.id.clone().try_into().unwrap(),
        session.token.clone().try_into().unwrap(),
    );

    let mut open = Vec::new();
    for _ in 0..2 {
        let mut stream = dial(&server);
        assert!(shake_hands(&mut stream, &hello).is_accepted());
        open.push(stream);
    }

    let ready = shake_hands(&mut dial(&server), &hello);
    // Temporary, not a refusal: it clears as the others close.
    assert_eq!(ready.status, ErrorClass::Transient);

    // Closing one makes room again.
    open.pop();
    let mut stream = dial(&server);
    let accepted = (0..100).any(|_| {
        // The server notices the close on its own thread, so give it a moment.
        std::thread::sleep(Duration::from_millis(20));
        TcpStream::connect(server.data_addr)
            .map(|mut fresh| {
                let ready = shake_hands(&mut fresh, &hello);
                if ready.is_accepted() {
                    stream = fresh;
                    true
                } else {
                    false
                }
            })
            .unwrap_or(false)
    });
    assert!(accepted, "a closed connection must free its slot");
    drop(stream);
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
