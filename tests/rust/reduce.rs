//! Reductions computed on the server.

use aex_client::{ClientError, DType, ErrorClass, FunctionArg, Index, Selection};

#[path = "support.rs"]
mod support;

use support::TestServer;

#[test]
fn a_reduction_comes_back_inline() {
    let server = TestServer::start();
    // f32 counting up from 0, 64 x 2000: past the inline limit to transfer.
    server.write_counting_npy("grid.npy", &[64, 2000]);
    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");

    let all = Selection::exact(handle, "array", &[]);
    let sum = client.apply_function(&all, "sum", &[]).expect("sum");
    assert_eq!((sum.dtype, sum.shape.clone()), (DType::Float32, vec![]));
    let n = 64.0 * 2000.0;
    let total = f32::from_le_bytes(sum.data.try_into().unwrap());
    assert!((f64::from(total) - n * (n - 1.0) / 2.0).abs() / (n * n) < 1e-6);

    // The mean of each of rows 10 and 11.
    let rows = [Index::range(10, 12)];
    let selection = Selection::exact(handle, "array", &rows);
    let mean = client
        .apply_function(
            &selection,
            "mean",
            &[
                ("axis", FunctionArg::Int(-1)),
                ("keepdims", FunctionArg::Bool(true)),
            ],
        )
        .expect("mean");
    assert_eq!(
        (mean.dtype, mean.shape.clone()),
        (DType::Float32, vec![2, 1])
    );
    let means: Vec<f32> = mean
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(means, vec![10.0 * 2000.0 + 999.5, 11.0 * 2000.0 + 999.5]);

    let std = client
        .apply_function(
            &selection,
            "std",
            &[("ddof", FunctionArg::Int(1)), ("axis", FunctionArg::None)],
        )
        .expect("std");
    assert_eq!(std.dtype, DType::Float32);
    // A reduction reads no data plane, and moves no bytes through it.
    assert_eq!(client.stats().bytes, 0);
}

#[test]
fn a_bad_reduction_is_a_bad_request() {
    let server = TestServer::start();
    server.write_counting_npy("grid.npy", &[64, 2000]);
    let client = server.connect();
    let handle = client.open("grid.npy").expect("open");
    let all = Selection::exact(handle, "array", &[]);

    let cases: Vec<(&str, Vec<(&str, FunctionArg)>)> = vec![
        ("median", vec![]),
        ("sum", vec![("axis", FunctionArg::Int(2))]),
        ("sum", vec![("out", FunctionArg::None)]),
        // Every element kept is far over the inline limit.
        ("sum", vec![("axis", FunctionArg::Ints(vec![]))]),
    ];
    for (function, kwargs) in cases {
        let err = client.apply_function(&all, function, &kwargs).unwrap_err();
        assert!(
            matches!(
                err,
                ClientError::Server {
                    class: ErrorClass::Request,
                    ..
                }
            ),
            "{function} {kwargs:?}: {err}"
        );
    }
}
