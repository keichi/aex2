//! What a selection costs to walk, with nothing else in the way.
//!
//! The source is already in memory, so what is timed is deciding which bytes
//! go where and copying them: no syscall, no disk, no socket. The pieces are
//! the size the server reads in, so the walk starts from an arbitrary output
//! offset as often as it does in the server.

use std::time::Instant;

use aex_bench::{cpu_seconds, report, Run};
use aex_core::{DType, Index, QualitySpec, SelectionLayout};

/// As big as the server's read buffer.
const PIECE: usize = 512 * 1024;

fn slice(step: i64) -> Index {
    Index::Slice {
        start: None,
        stop: None,
        step: Some(step),
    }
}

fn main() {
    let (rows, cols) = (16384u64, 1024u64);
    let itemsize = 4u64;
    let src: Vec<u8> = (0..rows * cols * itemsize).map(|i| i as u8).collect();

    let cases: Vec<(&str, Vec<Index>)> = vec![
        (
            "inner run [:, 0:512]",
            vec![Index::full(), Index::range(0, 512)],
        ),
        ("stride 2 [:, ::2]", vec![Index::full(), slice(2)]),
        ("stride 16 [:, ::16]", vec![Index::full(), slice(16)]),
        ("column [:, 0]", vec![Index::full(), Index::Single(0)]),
        (
            "fancy [:, [0, 2, 4, ...]]",
            vec![
                Index::full(),
                Index::Fancy((0..cols as i64).step_by(2).collect()),
            ],
        ),
    ];

    for (name, indices) in cases {
        let layout = SelectionLayout::resolve(
            &[rows, cols],
            DType::Float32,
            &indices,
            &QualitySpec::exact(),
        )
        .expect("layout");

        let mut dst = vec![0u8; PIECE];
        let mut runs = Vec::new();
        for _ in 0..5 {
            let cpu = cpu_seconds();
            let started = Instant::now();
            let mut at = 0u64;
            while at < layout.total_bytes {
                let len = PIECE.min((layout.total_bytes - at) as usize);
                layout
                    .read_with(at, &mut dst[..len], |from, buf| {
                        let from = from as usize;
                        buf.copy_from_slice(&src[from..from + buf.len()]);
                        Ok(())
                    })
                    .expect("read");
                at += len as u64;
            }
            runs.push(Run {
                bytes: layout.total_bytes,
                elapsed: started.elapsed(),
                cpu: cpu_seconds() - cpu,
            });
        }
        report(name, &runs);
    }
}
