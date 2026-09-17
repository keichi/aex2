//! Transfers over several data connections at once.
//!
//! Parallel bugs show up rarely, so the sweeps repeat and vary everything that
//! decides which connection carries which bytes.

use aex_client::{ClientConfig, Index};

#[path = "support.rs"]
mod support;

use support::TestServer;

const ROWS: usize = 300;
const COLS: usize = 1000;

fn every_third_column() -> Vec<Index> {
    vec![
        Index::full(),
        Index::Slice {
            start: None,
            stop: None,
            step: Some(3),
        },
    ]
}

/// Selections covering both layouts, with what numpy would return.
fn cases(all: &[f32]) -> Vec<(Vec<Index>, Vec<f32>)> {
    let at = |r: usize, c: usize| all[r * COLS + c];
    vec![
        (vec![], all.to_vec()),
        (
            every_third_column(),
            (0..ROWS)
                .flat_map(|r| (0..COLS).step_by(3).map(move |c| at(r, c)))
                .collect(),
        ),
        (
            vec![Index::Fancy(vec![299, 0, 150, 7])],
            [299, 0, 150, 7]
                .iter()
                .flat_map(|&r| (0..COLS).map(move |c| at(r, c)))
                .collect(),
        ),
    ]
}

#[test]
fn every_stream_count_returns_the_same_bytes() {
    let server = TestServer::start_with(|c| c.limits.max_streams_per_session = 16);
    let all = server.write_counting_npy("grid.npy", &[ROWS, COLS]);
    let cases = cases(&all);

    for streams in [1u32, 2, 3, 4, 8, 16] {
        for credit in [1u32, 4, 32] {
            // Not a multiple of the element size, so chunks split elements.
            for chunk_bytes in [10_007u64, 64 * 1024] {
                let client = server.connect_with(ClientConfig {
                    streams,
                    credit,
                    chunk_bytes,
                    ..ClientConfig::default()
                });
                let handle = client.open("grid.npy").expect("open");
                for _ in 0..3 {
                    for (indices, expected) in &cases {
                        let array = client
                            .read_selection_as::<f32>(handle, "array", indices)
                            .unwrap_or_else(|e| panic!("{streams}/{credit}: {e}"));
                        assert!(
                            array.data == *expected,
                            "streams={streams} credit={credit} chunk={chunk_bytes} {indices:?}"
                        );
                        assert_eq!(array.transfer.streams, streams.min(array.transfer.chunks));
                        assert_eq!(array.transfer.retries, 0);
                    }
                }
                client.disconnect().expect("disconnect");
            }
        }
    }
}

#[test]
fn threads_can_share_one_client() {
    let server = TestServer::start();
    let all = server.write_counting_npy("grid.npy", &[ROWS, COLS]);
    let cases = cases(&all);
    let client = server.connect_with(ClientConfig {
        chunk_bytes: 32 * 1024,
        ..ClientConfig::default()
    });
    let handle = client.open("grid.npy").expect("open");

    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..5 {
                    for (indices, expected) in &cases {
                        let array = client
                            .read_selection_as::<f32>(handle, "array", indices)
                            .expect("read");
                        assert!(array.data == *expected, "{indices:?}");
                    }
                }
            });
        }
    });
}

#[test]
fn a_plan_evicted_before_its_fill_is_prepared_again() {
    // One plan per session: preparing a second evicts the first.
    let server = TestServer::start_with(|c| c.limits.max_transfers_per_session = 1);
    let all = server.write_counting_npy("grid.npy", &[ROWS, COLS]);
    let client = server.connect_with(ClientConfig {
        chunk_bytes: 64 * 1024,
        ..ClientConfig::default()
    });
    let handle = client.open("grid.npy").expect("open");

    let plan = client.prepare(handle, "array", &[]).expect("prepare");
    client
        .prepare(handle, "array", &every_third_column())
        .expect("prepare");

    // Every connection has fetches of the evicted plan outstanding when the
    // first error arrives; they drain, and the whole transfer runs again.
    let mut bytes = vec![0u8; plan.total_bytes as usize];
    let result = client
        .fill(&plan, handle, "array", &[], &mut bytes)
        .expect("fill");
    assert!(result.retries >= 1);
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(values == all);

    // The drained connections are still good.
    let again = client
        .read_selection_as::<f32>(handle, "array", &[])
        .expect("read");
    assert!(again.data == all);
}
