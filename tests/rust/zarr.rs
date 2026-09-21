//! Zarr v3 stores served end to end.

use std::path::Path;

use aex_client::{ClientConfig, Index, Item};
use aex_core::ErrorClass;
use crc32c::crc32c;

#[path = "support.rs"]
mod support;

use support::{error_class, TestServer};

const ROWS: usize = 300;
const COLS: usize = 1000;
const CHUNK: [usize; 2] = [64, 300];

/// Write `<name>/group/{grid,packed}` holding 0, 1, 2, ... and return the
/// values. `packed` holds the same numbers with its chunks compressed.
fn write_grid(server: &TestServer, name: &str) -> Vec<f32> {
    let store = server.root().join(name);
    node(
        &store,
        "",
        r#"{"zarr_format": 3, "node_type": "group",
            "attributes": {"Conventions": "CF-1.8"}}"#,
    );
    node(
        &store.join("group"),
        "",
        r#"{"zarr_format": 3, "node_type": "group"}"#,
    );
    let values: Vec<f32> = (0..ROWS * COLS).map(|i| i as f32).collect();
    write_array(&store.join("group"), "grid", &values, false);
    write_array(&store.join("group"), "packed", &values, true);
    values
}

/// Write one node's metadata.
fn node(store: &Path, at: &str, json: &str) {
    let dir = if at.is_empty() {
        store.to_path_buf()
    } else {
        store.join(at)
    };
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("zarr.json"), json).expect("write");
}

/// Write `values` as a `ROWS x COLS` float32 array in chunks of `CHUNK`.
///
/// The chunks are laid out by hand rather than with a library: the point of
/// the test is that the server reads what a store really holds.
fn write_array(parent: &Path, name: &str, values: &[f32], packed: bool) {
    let codecs = if packed {
        r#", {"name": "zstd"}, {"name": "crc32c"}"#
    } else {
        ""
    };
    node(
        parent,
        name,
        &format!(
            r#"{{"zarr_format": 3, "node_type": "array", "data_type": "float32",
                 "shape": [{ROWS}, {COLS}],
                 "chunk_grid": {{"name": "regular",
                                 "configuration": {{"chunk_shape": {CHUNK:?}}}}},
                 "chunk_key_encoding": {{"name": "default"}},
                 "fill_value": 0.0,
                 "attributes": {{"units": "K"}},
                 "codecs": [{{"name": "bytes",
                              "configuration": {{"endian": "little"}}}}{codecs}]}}"#
        ),
    );
    let dir = parent.join(name);
    let grid = [ROWS.div_ceil(CHUNK[0]), COLS.div_ceil(CHUNK[1])];
    for cr in 0..grid[0] {
        for cc in 0..grid[1] {
            let mut chunk = vec![0f32; CHUNK[0] * CHUNK[1]];
            for r in 0..CHUNK[0] {
                for c in 0..CHUNK[1] {
                    let (row, col) = (cr * CHUNK[0] + r, cc * CHUNK[1] + c);
                    if row < ROWS && col < COLS {
                        chunk[r * CHUNK[1] + c] = values[row * COLS + col];
                    }
                }
            }
            let key = dir.join("c").join(cr.to_string());
            std::fs::create_dir_all(&key).expect("mkdir");
            let mut bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
            if packed {
                bytes = zstd::bulk::compress(&bytes, 3).expect("zstd");
                let sum = crc32c(&bytes);
                bytes.extend(sum.to_le_bytes());
            }
            std::fs::write(key.join(cc.to_string()), bytes).expect("write");
        }
    }
}

#[test]
fn an_array_arrives_whole_over_every_stream_count() {
    let server = TestServer::start_with(|c| c.limits.max_streams_per_session = 8);
    let all = write_grid(&server, "grid.zarr");
    let odd_rows: Vec<f32> = (1..ROWS)
        .step_by(2)
        .flat_map(|r| all[r * COLS..(r + 1) * COLS].to_vec())
        .collect();
    let every_other_row = [Index::Slice {
        start: Some(1),
        stop: None,
        step: Some(2),
    }];

    for streams in [1u32, 2, 8] {
        let client = server.connect_with(ClientConfig {
            streams,
            // Not a multiple of the element size, so chunks split elements.
            chunk_bytes: 10_007,
            ..ClientConfig::default()
        });
        let handle = client.open("grid.zarr").expect("open");
        for name in ["/group/grid", "group/packed"] {
            let array = client
                .read_selection_as::<f32>(handle, name, &[])
                .expect("read");
            assert_eq!(array.shape, [ROWS as u64, COLS as u64]);
            assert!(array.data == all, "{name} streams={streams}");
            let array = client
                .read_selection_as::<f32>(handle, name, &every_other_row)
                .expect("read");
            assert!(array.data == odd_rows, "{name} streams={streams}");
        }
        client.disconnect().expect("disconnect");
    }
}

#[test]
fn the_hierarchy_is_browsable() {
    let server = TestServer::start();
    write_grid(&server, "grid.zarr");
    let client = server.connect();
    let handle = client.open("grid.zarr").expect("open");

    let root = client.list_children(handle, "/").expect("list /");
    assert_eq!(root.len(), 1);
    assert_eq!(root[0].0, "group");
    assert!(matches!(root[0].1, Item::Group(_)));

    let children = client.list_children(handle, "group").expect("list group");
    let names: Vec<&str> = children.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["grid", "packed"]);

    let Item::Dataset(info) = client.get_item(handle, "/group/grid").expect("grid") else {
        panic!("grid must be a dataset");
    };
    assert_eq!(info.shape, [ROWS as u64, COLS as u64]);
    assert_eq!(text_attr(&info.attrs, "units").as_deref(), Some("K"));
    assert_eq!(
        error_class(client.get_item(handle, "/group/missing")),
        ErrorClass::Request
    );

    // A store's own attributes sit on its root, as netCDF's globals do.
    let Item::Group(attrs) = client.get_item(handle, "/").expect("root") else {
        panic!("the root must be a group");
    };
    assert_eq!(text_attr(&attrs, "Conventions").as_deref(), Some("CF-1.8"));
}

/// The text attribute called `name`, or `None`.
fn text_attr(attrs: &[(String, aex_client::AttrValue)], name: &str) -> Option<String> {
    attrs.iter().find(|(n, _)| n == name).map(|(_, v)| match v {
        aex_client::AttrValue::Text(text) => text.clone(),
        other => panic!("{name} is {other:?}, not text"),
    })
}

#[test]
fn a_store_is_opened_by_extension_or_by_an_explicit_format() {
    let server = TestServer::start();
    write_grid(&server, "named.zarr");
    write_grid(&server, "unnamed");
    server.write_npy("data.npy", &[4]);
    let client = server.connect();

    assert!(client.open("named.zarr").is_ok());
    // Without the extension the client has to say what the directory holds.
    assert_eq!(error_class(client.open("unnamed")), ErrorClass::Request);
    assert!(client.open_as("unnamed", "zarr").is_ok());

    // A file is not a store, and a directory is not a file.
    assert_eq!(
        error_class(client.open_as("data.npy", "zarr")),
        ErrorClass::Request
    );
    assert_eq!(
        error_class(client.open_as("named.zarr", "npy")),
        ErrorClass::Request
    );
}

#[test]
fn a_directory_that_is_not_a_store_is_refused() {
    let server = TestServer::start();
    std::fs::create_dir(server.root().join("empty.zarr")).expect("mkdir");
    let client = server.connect();
    assert_eq!(error_class(client.open("empty.zarr")), ErrorClass::Request);
}

#[test]
fn a_store_cannot_serve_what_lies_outside_it() {
    let server = TestServer::start();
    write_grid(&server, "grid.zarr");
    // The link's target is inside a data root, so the server's own path check
    // would not object; the backend refuses it because it leaves the store.
    let outside = server.root().join("beside");
    std::fs::create_dir_all(&outside).expect("mkdir");
    std::fs::write(outside.join("zarr.json"), b"{}").expect("write");
    std::os::unix::fs::symlink(&outside, server.root().join("grid.zarr/away")).expect("symlink");

    let client = server.connect();
    let handle = client.open("grid.zarr").expect("open");
    assert_eq!(
        error_class(client.get_item(handle, "away")),
        ErrorClass::Permanent
    );
    assert_eq!(
        error_class(client.get_item(handle, "../beside")),
        ErrorClass::Permanent
    );
    // The store itself still serves.
    assert!(client.get_item(handle, "/group/grid").is_ok());
}
