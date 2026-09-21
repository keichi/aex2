"""HDF5 files, written by h5py and checked against what h5py reads back.

The file is v1's test file: nested groups, one dataset filled and the rest
never written, so those read back as their fill value.
"""

import h5py
import numpy as np
import pytest
from test_selection import DTYPES, KEYS

from aex import AexNotFoundError, ArrayProxy, GroupProxy

SHAPE = (12, 7, 5)


@pytest.fixture(scope="module")
def h5_path(data_dir):
    path = data_dir / "test.h5"
    with h5py.File(path, "w") as f:
        g1 = f.create_group("g1")
        ds1 = g1.create_dataset("ds1", (100, 200), dtype=np.float32)
        g1.create_dataset("ds2", (100, 200), dtype=np.float64)
        g3 = g1.create_group("g3")
        g3.create_dataset("ds3", (100, 200), dtype=np.int8)
        g3.create_dataset("ds4", (100, 200), dtype=np.int16, fillvalue=-3)
        g2 = f.create_group("g2")
        g2.create_dataset("ds5", (100, 200), dtype=np.int32)
        g2.create_dataset("ds6", (100, 200), dtype=np.int64)
        x, y = np.meshgrid(np.arange(200), np.arange(100))
        ds1[...] = (x + y * 200).astype(np.float32)
        f.create_dataset("selection", data=np.arange(np.prod(SHAPE), dtype=np.int32).reshape(SHAPE))
        f.create_dataset("text", data=["a", "b"])
    return str(path)


@pytest.fixture
def h5(client, h5_path):
    return client.open(h5_path)


def test_nested_groups_are_walked(h5):
    g1 = h5["g1"]
    assert isinstance(g1, GroupProxy)
    assert g1.name == "/g1"
    assert h5["g1"]["g3"]["ds3"].name == "/g1/g3/ds3"
    assert g1["g3/ds4"].name == "/g1/g3/ds4"
    assert h5["/g2/ds5"].name == "/g2/ds5"


def test_groups_list_their_members(h5):
    # The string dataset is left out: it cannot be served.
    assert list(h5) == ["g1", "g2", "selection"]
    assert [item.name for item in h5.values()] == ["/g1", "/g2", "/selection"]
    assert [item.name for item in h5["g1"].values()] == ["/g1/ds1", "/g1/ds2", "/g1/g3"]
    assert len(h5["g1/g3"]) == 2
    assert "g3/ds3" in h5["g1"]
    assert "g3/nonexistent" not in h5["g1"]


def test_missing_and_unservable_items(h5):
    with pytest.raises(AexNotFoundError):
        h5["g1/nonexistent"]
    with pytest.raises(KeyError):
        h5["/nonexistent"]
    with pytest.raises(ValueError):
        h5["text"]


def test_written_data_reads_back(h5):
    ds1 = h5["g1/ds1"]
    assert isinstance(ds1, ArrayProxy)
    assert ds1.dtype == np.float32
    assert ds1.shape == (100, 200)
    assert ds1[5, 3] == 3 + 5 * 200
    assert ds1[10:20:2, 0].tolist() == [2000, 2400, 2800, 3200, 3600]


def test_unwritten_data_reads_as_the_fill_value(h5):
    np.testing.assert_array_equal(h5["g2/ds6"][:], np.zeros((100, 200), np.int64))
    np.testing.assert_array_equal(h5["g1/g3/ds4"][3], np.full(200, -3, np.int16))


@pytest.mark.parametrize("key", KEYS, ids=repr)
def test_selection_matches_numpy(key, h5, h5_path):
    with h5py.File(h5_path, "r") as f:
        expected = f["selection"][()][key]
    actual = h5["selection"][key]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_every_dtype_round_trips(dtype, client, data_dir):
    rng = np.random.default_rng(0)
    array = (rng.random((40, 30)) * 100).astype(dtype)
    if np.dtype(dtype).kind == "c":
        array = array + 1j * array[::-1]
    path = data_dir / f"dtype_{np.dtype(dtype).name}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=array)

    proxy = client.open(str(path))["data"]
    assert proxy.dtype == array.dtype
    for key in [slice(None), (slice(None), slice(None, None, 3)), [3, 1]]:
        actual = proxy[key]
        assert actual.dtype == array.dtype
        np.testing.assert_array_equal(actual, array[key])


@pytest.mark.parametrize(
    "options",
    [
        {"chunks": (7, 3)},
        {"chunks": (40, 30), "compression": "gzip"},
        {"chunks": (16, 30), "compression": "gzip", "shuffle": True, "fletcher32": True},
        {"maxshape": (None, 30)},
    ],
    ids=repr,
)
def test_chunked_datasets_match_numpy(options, client, data_dir):
    rng = np.random.default_rng(1)
    array = rng.random((40, 30))
    path = data_dir / f"chunked_{abs(hash(repr(options)))}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=array, **options)

    proxy = client.open(str(path))["data"]
    for key in [slice(None), (slice(3, 30, 4), [29, 0, 7]), 17, (Ellipsis, 2)]:
        np.testing.assert_array_equal(proxy[key], array[key])


def test_partly_written_chunks_read_as_the_fill_value(client, data_dir):
    path = data_dir / "partial.h5"
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            "data", (50, 20), dtype=np.int32, chunks=(10, 10), compression="gzip", fillvalue=9
        )
        ds[12:18, 5:15] = 1
    with h5py.File(path, "r") as f:
        expected = f["data"][()]
    np.testing.assert_array_equal(client.open(str(path))["data"][:], expected)


def test_unknown_filters_are_refused(client, data_dir):
    path = data_dir / "lzf.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.arange(100), compression="lzf")
    with pytest.raises(ValueError):
        client.open(str(path))["data"]


def test_big_endian_is_refused(client, data_dir):
    path = data_dir / "big.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.arange(10, dtype=">i4"))
    with pytest.raises(ValueError):
        client.open(str(path))["data"]


def test_external_links_are_not_followed(client, data_dir, h5_path):
    path = data_dir / "link.h5"
    with h5py.File(path, "w") as f:
        f["leak"] = h5py.ExternalLink(h5_path, "/g1/ds1")
    proxy = client.open(str(path))
    assert "leak" not in proxy
    with pytest.raises(KeyError):
        proxy["leak"]


def test_a_netcdf4_file_is_served(client, data_dir):
    path = data_dir / "test.nc"
    array = np.arange(24, dtype=np.float64).reshape(4, 6)
    with h5py.File(path, "w") as f:
        f.create_dataset("t", data=array)
    np.testing.assert_array_equal(client.open(str(path))["t"][1:3], array[1:3])
