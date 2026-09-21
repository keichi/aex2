//! Fault injection: broken connections and failed fetches mid-transfer.
//!
//! A proxy between the client and the data plane does the breaking, so the
//! server and the client run exactly the code they run in production.

use std::time::{Duration, Instant};

use aex_client::{ClientConfig, ClientError, ErrorClass};

#[path = "support.rs"]
mod support;

use support::{Fault, Proxy, TestServer};

/// 1.2 MB, so that a transfer is many chunks.
const SHAPE: [usize; 2] = [300, 1000];

fn through(proxy: &Proxy, adjust: impl FnOnce(&mut ClientConfig)) -> ClientConfig {
    let mut config = ClientConfig {
        data_endpoint: Some(proxy.endpoint()),
        chunk_bytes: 64 * 1024,
        ..ClientConfig::default()
    };
    adjust(&mut config);
    config
}

#[test]
fn chunks_lost_with_a_connection_are_fetched_again() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("grid.npy", &SHAPE);
    // Two connections cannot carry 1.2 MB at 500 KB each without one being
    // cut, and each gets through several chunks before it is.
    let proxy = Proxy::start(server.data_addr, Fault::CutAfter(500_000));
    let client = server.connect_with(through(&proxy, |c| c.streams = 2));
    let handle = client.open("grid.npy").expect("open");

    let array = client
        .read_selection_as::<f32>(handle, "array", &[])
        .expect("read");
    assert!(array.data == expected);
    assert!(proxy.cuts() >= 1);
    assert!(array.transfer.retries >= 1);
}

#[test]
fn a_connection_cut_again_and_again_still_finishes_the_transfer() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("grid.npy", &SHAPE);
    // Each rebuilt connection gets through a few chunks before it goes too.
    let proxy = Proxy::start(server.data_addr, Fault::CutAfter(200_000));
    let client = server.connect_with(through(&proxy, |c| {
        c.streams = 1;
        c.credit = 2;
    }));
    let handle = client.open("grid.npy").expect("open");

    let array = client
        .read_selection_as::<f32>(handle, "array", &[])
        .expect("read");
    assert!(array.data == expected);
    assert!(proxy.cuts() >= 5, "{} cuts", proxy.cuts());
    assert_eq!(array.transfer.streams, 1);
}

#[test]
fn deep_credit_does_not_use_up_the_retries() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("grid.npy", &SHAPE);
    // One chunk and a bit per connection: every break takes seven fetches
    // with it that were only waiting their turn.
    let proxy = Proxy::start(server.data_addr, Fault::CutAfter(100_000));
    let client = server.connect_with(through(&proxy, |c| {
        c.streams = 1;
        c.credit = 8;
    }));
    let handle = client.open("grid.npy").expect("open");

    let array = client
        .read_selection_as::<f32>(handle, "array", &[])
        .expect("read");
    assert!(array.data == expected);
    assert!(proxy.cuts() >= 15, "{} cuts", proxy.cuts());
}

#[test]
fn a_connection_that_never_gets_a_chunk_through_fails_the_transfer() {
    let server = TestServer::start();
    server.write_counting_npy("grid.npy", &SHAPE);
    // Past the handshake, short of any chunk.
    let proxy = Proxy::start(server.data_addr, Fault::CutAfter(1_000));
    let client = server.connect_with(through(&proxy, |c| c.streams = 1));
    let handle = client.open("grid.npy").expect("open");

    let started = Instant::now();
    let err = client
        .read_selection(handle, "array", &[])
        .expect_err("no chunk can arrive");
    // Given up after the retries, not hung.
    assert!(started.elapsed() < Duration::from_secs(10));
    assert!(
        matches!(err, ClientError::Io(_)) || err.class() == Some(ErrorClass::Transient),
        "{err}"
    );
}

#[test]
fn a_transient_error_retries_just_that_chunk() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("grid.npy", &SHAPE);
    let proxy = Proxy::start(server.data_addr, Fault::FailFirstFetch);
    let client = server.connect_with(through(&proxy, |c| {
        c.streams = 4;
        c.credit = 1;
    }));
    let handle = client.open("grid.npy").expect("open");

    let array = client
        .read_selection_as::<f32>(handle, "array", &[])
        .expect("read");
    assert!(array.data == expected);
    // One retry per failed fetch, and the connections stayed up. Which
    // connections get a fetch to fail is up to the scheduler, so the retries
    // are read against the proxy's count rather than the stream count.
    assert!(proxy.failed_fetches() > 0);
    assert_eq!(array.transfer.retries as usize, proxy.failed_fetches());
    assert_eq!(proxy.cuts(), 0);
    assert_eq!(proxy.accepted(), 4);
}

#[test]
fn a_file_truncated_under_a_transfer_fails_it_without_retrying() {
    let server = TestServer::start();
    let path = server.write_npy("grid.npy", &SHAPE);
    let expected = server.write_counting_npy("intact.npy", &SHAPE);
    let proxy = Proxy::start(server.data_addr, Fault::None);
    let client = server.connect_with(through(&proxy, |c| c.streams = 4));
    let handle = client.open("grid.npy").expect("open");
    let plan = client.prepare(handle, "array", &[]).expect("prepare");

    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .and_then(|f| f.set_len(1024))
        .expect("truncate");

    let mut bytes = vec![0u8; plan.total_bytes as usize];
    let err = client
        .fill(&plan, handle, "array", &[], &mut bytes)
        .expect_err("the data is gone");
    assert_eq!(err.class(), Some(ErrorClass::Permanent), "{err}");

    // The connections drained rather than broke, and serve the next transfer.
    let intact = client.open("intact.npy").expect("open");
    let array = client
        .read_selection_as::<f32>(intact, "array", &[])
        .expect("read");
    assert!(array.data == expected);
    assert_eq!(proxy.accepted(), 4);
}
