"""Zarr v3 stores, written by zarr-python and checked against what it reads back.

zarr-python is the oracle here: the backend is a reader of its own, so the only
claim worth testing is that the two agree on the same store.
"""

import numpy as np
import pytest
import zarr
from test_selection import DTYPES, KEYS

from aex import AexNotFoundError, AexValueError, ArrayProxy, GroupProxy

SHAPE = (12, 7, 5)


@pytest.fixture(scope="module")
def zarr_path(data_dir):
    """A store shaped like the HDF5 fixture, so the two can be read alike."""
    path = data_dir / "test.zarr"
    root = zarr.create_group(path, zarr_format=3)
    g1 = root.create_group("g1")
    x, y = np.meshgrid(np.arange(200), np.arange(100))
    ds1 = g1.create_array("ds1", shape=(100, 200), dtype="float32", chunks=(32, 64))
    ds1[...] = (x + y * 200).astype(np.float32)
    g1.create_array("ds2", shape=(100, 200), dtype="float64", chunks=(32, 64))
    g3 = g1.create_group("g3")
    g3.create_array("ds3", shape=(100, 200), dtype="int8", chunks=(32, 64))
    # Never written, so every chunk is missing and reads as the fill value.
    g3.create_array("ds4", shape=(100, 200), dtype="int16", chunks=(32, 64), fill_value=-3)
    g2 = root.create_group("g2")
    g2.create_array("ds5", shape=(100, 200), dtype="int32", chunks=(32, 64))
    g2.create_array("ds6", shape=(100, 200), dtype="int64", chunks=(32, 64))
    selection = root.create_array("selection", shape=SHAPE, dtype="int32", chunks=(5, 3, 2))
    selection[...] = np.arange(np.prod(SHAPE), dtype=np.int32).reshape(SHAPE)
    return str(path)


@pytest.fixture
def store(client, zarr_path):
    return client.open(zarr_path)


def test_nested_groups_are_walked(store):
    g1 = store["g1"]
    assert isinstance(g1, GroupProxy)
    assert g1.name == "/g1"
    assert store["g1"]["g3"]["ds3"].name == "/g1/g3/ds3"
    assert g1["g3/ds4"].name == "/g1/g3/ds4"
    assert store["/g2/ds5"].name == "/g2/ds5"


def test_groups_list_their_members(store):
    assert sorted(store.keys()) == ["g1", "g2", "selection"]
    assert sorted(store["g1"].keys()) == ["ds1", "ds2", "g3"]
    assert isinstance(store["g1"]["ds1"], ArrayProxy)
    with pytest.raises(AexNotFoundError):
        store["absent"]


def test_an_array_matches_what_zarr_python_reads(store, zarr_path):
    expected = zarr.open_array(f"{zarr_path}/g1/ds1")[...]
    got = store["g1/ds1"][...]
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_unwritten_chunks_read_as_the_fill_value(store, zarr_path):
    expected = zarr.open_array(f"{zarr_path}/g1/g3/ds4")[...]
    assert (expected == -3).all()
    np.testing.assert_array_equal(store["g1/g3/ds4"][...], expected)


@pytest.mark.parametrize("key", KEYS, ids=repr)
def test_selections_match_zarr_python(key, store, zarr_path):
    reference = zarr.open_array(f"{zarr_path}/selection")[...]
    proxy = store["selection"]
    np.testing.assert_array_equal(proxy[key], reference[key])


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_every_dtype_matches_zarr_python(dtype, client, data_dir):
    """Every element type survives, in a shape no chunk covers evenly."""
    name = np.dtype(dtype).name
    path = data_dir / f"dtype-{name}.zarr"
    values = np.arange(35).reshape(5, 7)
    if np.dtype(dtype) == np.bool_:
        values = values % 3 == 0
    array = zarr.create_array(str(path), shape=(5, 7), dtype=dtype, chunks=(2, 3))
    array[...] = values.astype(dtype)

    # zarr-python puts a bare array at the root of its own store, so the file
    # proxy's own path is the array.
    proxy = client.open(str(path))[""]
    assert isinstance(proxy, ArrayProxy)
    got = proxy[...]
    assert got.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(got, zarr.open_array(str(path))[...])


CHAINS = {
    "uncompressed": None,
    # What zarr-python writes when it is not told otherwise.
    "default": "auto",
    "gzip": [zarr.codecs.GzipCodec()],
    "zstd+crc32c": [zarr.codecs.ZstdCodec(), zarr.codecs.Crc32cCodec()],
}


@pytest.mark.parametrize("chain", list(CHAINS), ids=list(CHAINS))
def test_every_codec_chain_matches_zarr_python(chain, client, data_dir):
    path = data_dir / f"codec-{chain}.zarr"
    compressors = CHAINS[chain]
    # A shape the chunks do not cover evenly, so edge chunks are padded before
    # they are compressed.
    values = np.arange(35, dtype=np.float32).reshape(5, 7)
    array = zarr.create_array(
        str(path), shape=(5, 7), dtype="float32", chunks=(2, 3), compressors=compressors
    )
    array[...] = values

    proxy = client.open(str(path))[""]
    np.testing.assert_array_equal(proxy[...], zarr.open_array(str(path))[...])
    # Reading a piece at a time goes back to the same decoded chunks.
    np.testing.assert_array_equal(proxy[1:4, ::2], values[1:4, ::2])


@pytest.mark.parametrize("location", ["end", "start"])
def test_a_sharded_array_matches_zarr_python(location, client, data_dir):
    """zarr-python is the oracle for the shard index, which is the whole point."""
    path = data_dir / f"sharded-{location}.zarr"
    values = np.arange(35, dtype=np.int32).reshape(5, 7)
    # The serializer alone says how a shard is divided; `shards=` as well
    # would nest a second sharding codec inside the first.
    array = zarr.create_array(
        str(path),
        shape=(5, 7),
        dtype="int32",
        chunks=(4, 4),
        compressors=None,
        serializer=zarr.codecs.ShardingCodec(chunk_shape=(2, 2), index_location=location),
    )
    array[...] = values

    proxy = client.open(str(path))[""]
    np.testing.assert_array_equal(proxy[...], zarr.open_array(str(path))[...])
    # A selection that crosses both shard and inner-chunk boundaries.
    np.testing.assert_array_equal(proxy[1:5, ::3], values[1:5, ::3])


def test_a_partly_written_shard_matches_zarr_python(client, data_dir):
    path = data_dir / "sparse-shard.zarr"
    array = zarr.create_array(
        str(path), shape=(8, 8), dtype="int32", shards=(4, 4), chunks=(2, 2), fill_value=-3
    )
    # One shard written, the rest never touched.
    array[0:4, 0:4] = np.arange(16, dtype=np.int32).reshape(4, 4)

    proxy = client.open(str(path))[""]
    np.testing.assert_array_equal(proxy[...], zarr.open_array(str(path))[...])


def test_a_store_that_is_not_one_is_refused(client, data_dir):
    empty = data_dir / "empty.zarr"
    empty.mkdir(exist_ok=True)
    with pytest.raises(AexValueError):
        client.open(str(empty))
