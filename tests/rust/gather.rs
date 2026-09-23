//! Several selections resolved in one round trip and fetched as one batch.

use aex_client::{
    Client, ClientError, DType, ErrorClass, FileHandle, Index, Plan, QualitySpec, Selection,
};
use aex_core::Encoding;

#[path = "support.rs"]
mod support;

use std::time::{Duration, Instant};

use aex_client::ClientConfig;
use support::{Fault, Proxy, TestServer};

const ROW: usize = 2000;

fn rows(start: i64, stop: i64) -> Vec<Index> {
    vec![Index::range(start, stop)]
}

/// Fill every plan and return the buffers as f32.
fn fill_all(
    client: &Client,
    plans: &[Plan],
    selections: &[Selection<'_>],
) -> Result<(Vec<Vec<f32>>, aex_client::TransferResult), ClientError> {
    let mut bufs: Vec<Vec<u8>> = plans
        .iter()
        .map(|p| vec![0u8; p.total_bytes as usize])
        .collect();
    let mut dsts: Vec<&mut [u8]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
    let result = client.fill_many(plans, selections, &mut dsts)?;
    let floats = bufs
        .iter()
        .map(|b| {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        })
        .collect();
    Ok((floats, result))
}

fn selections<'a>(handle: FileHandle, keys: &'a [Vec<Index>]) -> Vec<Selection<'a>> {
    keys.iter()
        .map(|indices| Selection::exact(handle, "array", indices))
        .collect()
}

#[test]
fn a_batch_mixes_inline_remote_and_failed_selections() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("grid.npy", &[64, ROW]);
    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");

    // One row is inline, ten rows are not, and row 99 does not exist.
    let keys = [
        rows(0, 1),
        rows(10, 20),
        vec![Index::Single(99)],
        rows(60, 64),
    ];
    let sels = selections(handle, &keys);
    let results = client.prepare_many(&sels).expect("prepare");
    assert_eq!(results.len(), 4);
    assert!(results[0].as_ref().unwrap().is_inline());
    assert!(!results[1].as_ref().unwrap().is_inline());
    let err = results[2].as_ref().unwrap_err();
    assert_eq!(err.class(), Some(ErrorClass::Request), "{err}");

    let ok: Vec<usize> = vec![0, 1, 3];
    let plans: Vec<Plan> = ok
        .iter()
        .map(|&i| results[i].as_ref().unwrap().clone())
        .collect();
    let ok_sels: Vec<_> = ok.iter().map(|&i| sels[i]).collect();
    let (data, transfer) = fill_all(&client, &plans, &ok_sels).expect("fill");

    assert_eq!(data[0], expected[..ROW]);
    assert_eq!(data[1], expected[10 * ROW..20 * ROW]);
    assert_eq!(data[2], expected[60 * ROW..]);
    assert!(!transfer.inline);
    assert_eq!(transfer.bytes, (1 + 10 + 4) as u64 * ROW as u64 * 4);
}

#[test]
fn inline_data_stops_at_what_one_reply_can_carry() {
    let server = TestServer::start_with(|config| {
        config.limits.grpc_max_message_bytes = 256 * 1024;
    });
    let expected = server.write_counting_npy("grid.npy", &[64, ROW]);
    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");

    // Seven rows are 56 KB, under the inline limit, but a reply carries at most
    // half of 256 KiB of inline data, so only two of the four fit.
    let keys: Vec<_> = (0..4).map(|i| rows(i * 7, i * 7 + 7)).collect();
    let sels = selections(handle, &keys);
    let plans: Vec<Plan> = client
        .prepare_many(&sels)
        .expect("prepare")
        .into_iter()
        .map(|r| r.expect("plan"))
        .collect();
    let inline = plans.iter().filter(|p| p.is_inline()).count();
    assert_eq!(inline, 2, "{plans:?}");

    let (data, _) = fill_all(&client, &plans, &sels).expect("fill");
    for (i, got) in data.iter().enumerate() {
        assert_eq!(got[..], expected[i * 7 * ROW..(i + 1) * 7 * ROW]);
    }
}

#[test]
fn evicted_plans_in_a_batch_are_prepared_again() {
    let server = TestServer::start_with(|config| {
        config.limits.max_transfers_per_session = 2;
    });
    let expected = server.write_counting_npy("grid.npy", &[64, ROW]);
    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");

    let keys = [rows(0, 10), rows(10, 20)];
    let sels = selections(handle, &keys);
    let plans: Vec<Plan> = client
        .prepare_many(&sels)
        .expect("prepare")
        .into_iter()
        .map(|r| r.expect("plan"))
        .collect();
    // Two more plans push the first two out.
    let others = [rows(20, 30), rows(30, 40)];
    for r in client.prepare_many(&selections(handle, &others)).unwrap() {
        r.expect("plan");
    }

    let (data, transfer) = fill_all(&client, &plans, &sels).expect("fill");
    assert_eq!(data[0], expected[..10 * ROW]);
    assert_eq!(data[1], expected[10 * ROW..20 * ROW]);
    assert!(transfer.retries >= 1);
}

#[test]
fn a_quality_the_server_lacks_falls_back_to_exact() {
    let server = TestServer::start();
    let expected = server.write_counting_npy("grid.npy", &[64, ROW]);
    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");

    // Float32 does not narrow to float64, and nothing pretends otherwise.
    let widened = QualitySpec {
        encoding: Encoding::DtypeCast,
        cast_dtype: Some(DType::Float64),
        ..QualitySpec::exact()
    };
    let indices = rows(0, 10);
    let selection = Selection {
        quality: &widened,
        ..Selection::exact(handle, "array", &indices)
    };
    let plan = client.prepare_selection(&selection).expect("prepare");
    assert!(plan.applied_quality.is_exact());
    assert_eq!(plan.dtype, DType::Float32);
    let (data, _) = fill_all(&client, &[plan], &[selection]).expect("fill");
    assert_eq!(data[0], expected[..10 * ROW]);
}

#[test]
fn a_gathered_selection_is_cast_on_the_way_out() {
    // The gather walk steps the array's own elements while the stream counts
    // the narrowed ones, so this is where the two itemsizes have to be kept
    // apart.
    let server = TestServer::start();
    let expected = server.write_counting_npy("grid.npy", &[64, ROW]);
    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");

    let narrowed = QualitySpec {
        encoding: Encoding::DtypeCast,
        cast_dtype: Some(DType::Float16),
        ..QualitySpec::exact()
    };
    // Every third row, which is a gather rather than one run.
    let indices = vec![Index::Slice {
        start: Some(0),
        stop: Some(30),
        step: Some(3),
    }];
    let selection = Selection {
        quality: &narrowed,
        ..Selection::exact(handle, "array", &indices)
    };
    let plan = client.prepare_selection(&selection).expect("prepare");
    assert_eq!(plan.applied_quality.encoding, Encoding::DtypeCast);
    assert_eq!(plan.dtype, DType::Float16);
    assert_eq!(plan.shape, vec![10, ROW as u64]);
    assert_eq!(plan.total_bytes, (10 * ROW * 2) as u64);

    let mut buf = vec![0u8; plan.total_bytes as usize];
    client
        .fill_many(&[plan], &[selection], &mut [&mut buf])
        .expect("fill");
    let got: Vec<f32> = buf
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    let want: Vec<f32> = (0..10)
        .flat_map(|r| expected[r * 3 * ROW..(r * 3 + 1) * ROW].iter())
        .map(|v| half::f16::from_f32(*v).to_f32())
        .collect();
    assert_eq!(got, want);
}

#[test]
fn a_batch_costs_one_round_trip_where_one_by_one_costs_many() {
    let server = TestServer::start();
    server.write_counting_npy("grid.npy", &[64, ROW]);
    let control: std::net::SocketAddr = server.url.trim_start_matches("http://").parse().unwrap();
    // 10 ms each way: a 20 ms round trip on the control plane only.
    let proxy = Proxy::start(control, Fault::Delay(Duration::from_millis(10)));
    let client = Client::connect(&format!("http://{}", proxy.addr), ClientConfig::default())
        .expect("connect");
    let handle = client.open("grid.npy").expect("open");

    // Twenty rows, each small enough to come back inline.
    let keys: Vec<_> = (0..20).map(|i| rows(i, i + 1)).collect();
    let sels = selections(handle, &keys);

    let started = Instant::now();
    for selection in &sels {
        client.prepare_selection(selection).expect("prepare");
    }
    let one_by_one = started.elapsed();

    let started = Instant::now();
    let plans: Vec<Plan> = client
        .prepare_many(&sels)
        .expect("prepare")
        .into_iter()
        .map(|r| r.expect("plan"))
        .collect();
    fill_all(&client, &plans, &sels).expect("fill");
    let batch = started.elapsed();

    assert!(
        one_by_one >= Duration::from_millis(20 * 20),
        "{one_by_one:?}"
    );
    assert!(
        batch * 4 < one_by_one,
        "batch {batch:?}, one by one {one_by_one:?}"
    );
}

#[test]
fn a_batch_that_does_not_pair_up_is_refused() {
    let server = TestServer::start();
    server.write_counting_npy("grid.npy", &[64, ROW]);
    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");

    let keys = [rows(0, 10)];
    let sels = selections(handle, &keys);
    let plan = client.prepare_many(&sels).unwrap().remove(0).unwrap();
    let err = client
        .fill_many(&[plan], &sels, &mut [])
        .expect_err("no buffer");
    assert!(matches!(err, ClientError::BadRequest(_)), "{err}");
}
