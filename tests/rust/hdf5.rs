//! HDF5 files served end to end.

use aex_client::{ClientConfig, Index, Item};

#[path = "support.rs"]
mod support;

use support::{error_class, TestServer};

const ROWS: usize = 300;
const COLS: usize = 1000;

/// Write `/group/grid` holding 0, 1, 2, ... and return the values, along with
/// `/group/packed`, the same values compressed in chunks.
fn write_grid(server: &TestServer, name: &str) -> Vec<f32> {
    let values: Vec<f32> = (0..ROWS * COLS).map(|i| i as f32).collect();
    // Without libhdf5's file lock, as the server opens it: it refuses a file
    // whose locking setting disagrees with a handle already open.
    let file = hdf5::File::with_options()
        .with_fapl(|fapl| fapl.file_locking(false))
        .create(server.root().join(name))
        .expect("create");
    let group = file.create_group("group").expect("group");
    group
        .new_dataset::<f32>()
        .shape([ROWS, COLS])
        .create("grid")
        .expect("dataset")
        .write_raw(&values)
        .expect("write");
    group
        .new_dataset::<f32>()
        .shape([ROWS, COLS])
        .chunk([64, 300])
        .shuffle()
        .deflate(4)
        .create("packed")
        .expect("dataset")
        .write_raw(&values)
        .expect("write");
    values
}

#[test]
fn a_dataset_arrives_whole_over_every_stream_count() {
    let server = TestServer::start_with(|c| c.limits.max_streams_per_session = 8);
    let all = write_grid(&server, "grid.h5");
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
        let handle = client.open("grid.h5").expect("open");
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
    write_grid(&server, "grid.h5");
    let client = server.connect();
    let handle = client.open("grid.h5").expect("open");

    let children = client.list_children(handle, "group").expect("list group");
    let names: Vec<&str> = children.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["grid", "packed"]);
    let root = client.list_children(handle, "/").expect("list /");
    assert_eq!(root.len(), 1);
    assert_eq!(root[0].0, "group");
    assert!(matches!(root[0].1, Item::Group));
    let Item::Dataset(info) = client.get_item(handle, "/group/grid").expect("grid") else {
        panic!("grid must be a dataset");
    };
    assert_eq!(info.shape, [ROWS as u64, COLS as u64]);
    assert_eq!(
        error_class(client.get_item(handle, "/group/missing")),
        aex_core::ErrorClass::Request
    );
}

#[test]
fn netcdf4_extensions_and_names_open_as_hdf5() {
    let server = TestServer::start();
    write_grid(&server, "grid.nc");
    std::fs::copy(
        server.root().join("grid.nc"),
        server.root().join("grid.bin"),
    )
    .unwrap();
    let client = server.connect();
    for handle in [
        client.open("grid.nc").expect("open .nc"),
        client
            .open_as("grid.bin", "netcdf4")
            .expect("open as netcdf4"),
    ] {
        assert!(matches!(
            client.get_item(handle, "group/grid"),
            Ok(Item::Dataset(_))
        ));
    }
    // An npy is not HDF5, whatever it is called.
    server.write_npy("data.npy", &[4]);
    assert_eq!(
        error_class(client.open_as("data.npy", "hdf5")),
        aex_core::ErrorClass::Request
    );
}
