"""v1's client tests, ported onto the npy hierarchy.

v1 served an HDF5 file with nested groups. A .npy is a root group holding one
dataset named ``array``, so the datasets are separate files here. The nested
group tests live in test_hdf5.py.
"""

from collections.abc import Mapping

import numpy as np
import pytest

import aex
from aex import AexError, AexNotFoundError, ArrayProxy, FileProxy, GroupProxy

# ============================================================
# ArrayProxy
# ============================================================


def test_array_proxy_attributes(array_proxy):
    assert array_proxy.dtype == np.float32
    assert array_proxy.shape == (100, 200)
    assert array_proxy.ndim == 2
    assert array_proxy.size == 20000
    assert len(array_proxy) == 100
    assert array_proxy.name == "/array"


def test_array_proxy_getitem_single_index(array_proxy):
    result = array_proxy[5]
    assert isinstance(result, np.ndarray)
    assert result.shape == (200,)
    assert result.dtype == np.float32
    assert not result.flags.writeable
    assert result[0] == 0 + 5 * 200
    assert result[3] == 3 + 5 * 200


def test_array_proxy_getitem_negative_index(array_proxy):
    result = array_proxy[-1]
    assert result.shape == (200,)
    assert result[0] == 0 + 99 * 200
    assert result[1] == 1 + 99 * 200


def test_array_proxy_getitem_slice(array_proxy):
    result = array_proxy[10:20]
    assert result.shape == (10, 200)
    assert result.dtype == np.float32
    assert not result.flags.writeable
    assert result[0, 0] == 0 + 10 * 200
    assert result[0, 3] == 3 + 10 * 200


def test_array_proxy_getitem_slice_with_step(array_proxy):
    result = array_proxy[10:20:2]
    assert result.shape == (5, 200)
    assert result[0, 0] == 0 + 10 * 200
    assert result[1, 0] == 0 + 12 * 200


def test_array_proxy_getitem_slice_defaults(array_proxy):
    result = array_proxy[:10]
    assert result.shape == (10, 200)
    assert result[0, 0] == 0

    result = array_proxy[90:]
    assert result.shape == (10, 200)
    assert result[0, 0] == 0 + 90 * 200

    assert array_proxy[:].shape == (100, 200)


def test_array_proxy_getitem_fancy_list(array_proxy):
    result = array_proxy[[1, 5, 10]]
    assert result.shape == (3, 200)
    assert result.dtype == np.float32
    assert list(result[:, 0]) == [200, 1000, 2000]


def test_array_proxy_getitem_fancy_numpy_array(array_proxy):
    result = array_proxy[np.array([1, 5, 10])]
    assert result.shape == (3, 200)
    assert list(result[:, 0]) == [200, 1000, 2000]


def test_array_proxy_getitem_2d_slice(array_proxy):
    result = array_proxy[10:20, 30:40]
    assert result.shape == (10, 10)
    assert not result.flags.writeable
    assert result[0, 0] == 30 + 10 * 200
    assert result[5, 3] == 33 + 15 * 200


def test_array_proxy_getitem_mixed_indexing(array_proxy):
    result = array_proxy[5, 10:20]
    assert result.shape == (10,)
    assert result[0] == 10 + 5 * 200
    assert result[5] == 15 + 5 * 200


def test_array_proxy_getitem_data_pattern_verification(array_proxy):
    result = array_proxy[0:3, 0:3]
    for i in range(3):
        for j in range(3):
            assert result[i, j] == j + i * 200


def test_array_proxy_getitem_empty_selection(array_proxy):
    result = array_proxy[10:10]
    assert result.shape == (0, 200)


def test_array_proxy_getitem_ellipsis_and_newaxis(array_proxy):
    # v1 could not do either.
    assert array_proxy[..., 0].shape == (100,)
    assert array_proxy[..., 0][7] == 7 * 200
    assert array_proxy[:, None].shape == (100, 1, 200)


def test_array_proxy_getitem_boolean_mask(array_proxy):
    mask = np.zeros(100, dtype=bool)
    mask[[3, 50]] = True
    result = array_proxy[mask]
    assert result.shape == (2, 200)
    assert list(result[:, 1]) == [1 + 3 * 200, 1 + 50 * 200]


def test_array_proxy_asarray(array_proxy):
    data = np.asarray(array_proxy)
    assert data.shape == (100, 200)
    assert data[99, 199] == 199 + 99 * 200
    assert np.asarray(array_proxy, dtype=np.float64).dtype == np.float64


def test_array_proxy_iter(array_proxy):
    rows = list(array_proxy)
    assert len(rows) == 100
    assert rows[42][0] == 42 * 200


def test_array_proxy_repr(array_proxy):
    text = repr(array_proxy)
    assert "ArrayProxy" in text
    assert "/array" in text
    assert "(100, 200)" in text
    assert "float32" in text


def test_array_proxy_out_of_bounds(array_proxy):
    with pytest.raises(aex.AexValueError) as info:
        array_proxy[100]
    assert info.value.error_class == "REQUEST"
    assert isinstance(info.value, ValueError)


# ============================================================
# GroupProxy
# ============================================================


def test_group_proxy_getitem_dataset(file_proxy):
    item = file_proxy["array"]
    assert isinstance(item, ArrayProxy)
    assert item.name == "/array"


def test_group_proxy_getitem_absolute(file_proxy):
    assert isinstance(file_proxy["/array"], ArrayProxy)


def test_group_proxy_getitem_root_is_a_group(file_proxy):
    root = file_proxy["/"]
    assert isinstance(root, GroupProxy)
    assert not isinstance(root, FileProxy)


def test_group_proxy_getitem_nonexistent(file_proxy):
    with pytest.raises(KeyError):
        file_proxy["nonexistent"]
    with pytest.raises(AexNotFoundError) as info:
        file_proxy["nonexistent"]
    assert info.value.error_class == "REQUEST"


def test_group_proxy_iter(file_proxy):
    assert list(file_proxy) == ["array"]


def test_group_proxy_iter_multiple_times(file_proxy):
    assert len(list(file_proxy)) == len(list(file_proxy)) == 1


def test_group_proxy_is_a_mapping(file_proxy):
    assert isinstance(file_proxy, Mapping)


def test_group_proxy_keys(file_proxy):
    assert list(file_proxy.keys()) == ["array"]


def test_group_proxy_values(file_proxy):
    values = list(file_proxy.values())
    assert len(values) == 1
    assert isinstance(values[0], ArrayProxy)
    assert values[0].name == "/array"


def test_group_proxy_items(file_proxy):
    (name, item) = next(iter(file_proxy.items()))
    assert name == "array"
    assert isinstance(item, ArrayProxy)


class _ListingOnly:
    """The native client with get_item taken away, to prove it is not used."""

    def __init__(self, native):
        self._native = native

    def list_children(self, *args):
        return self._native.list_children(*args)

    def get_item(self, *args):
        raise AssertionError("the children were asked for one at a time")


def test_group_proxy_values_come_from_one_listing(file_proxy):
    file_proxy._native = _ListingOnly(file_proxy._native)
    assert [item.name for item in file_proxy.values()] == ["/array"]
    assert [name for name, _ in file_proxy.items()] == ["array"]


def test_group_proxy_dict(file_proxy):
    mapping = dict(file_proxy)
    assert list(mapping) == ["array"]
    assert isinstance(mapping["array"], ArrayProxy)


def test_group_proxy_get(file_proxy):
    assert isinstance(file_proxy.get("array"), ArrayProxy)
    assert file_proxy.get("nonexistent") is None


def test_group_proxy_len(file_proxy):
    assert len(file_proxy) == 1


def test_group_proxy_contains(file_proxy):
    assert "array" in file_proxy
    assert "/array" in file_proxy
    assert "nonexistent" not in file_proxy
    assert "/nonexistent" not in file_proxy
    assert "array/nonexistent" not in file_proxy


def test_group_proxy_join_names_relative():
    assert GroupProxy._join_names("/g1", "ds1") == "/g1/ds1"


def test_group_proxy_join_names_absolute_replaces():
    assert GroupProxy._join_names("/g1", "/g2") == "/g2"


def test_group_proxy_join_names_trailing_slash():
    assert GroupProxy._join_names("/g1/", "ds1") == "/g1/ds1"


def test_group_proxy_join_names_multiple():
    assert GroupProxy._join_names("/", "g1", "g3", "ds3") == "/g1/g3/ds3"


def test_group_proxy_repr(file_proxy):
    text = repr(file_proxy["/"])
    assert "GroupProxy" in text
    assert '"/"' in text


# ============================================================
# FileProxy
# ============================================================


def test_file_proxy_initialization(client, ds_paths):
    proxy = client.open(ds_paths["ds1"])
    assert proxy.name == "/"
    assert isinstance(proxy.handle, int)
    proxy.close()


def test_file_proxy_is_group_proxy(file_proxy):
    assert isinstance(file_proxy, GroupProxy)


def test_file_proxy_close(client, ds_paths):
    proxy = client.open(ds_paths["ds1"])
    proxy.close()
    # The handle is gone.
    with pytest.raises(AexError):
        proxy["array"]


def test_file_proxy_context_manager(client, ds_paths):
    with client.open(ds_paths["ds1"]) as proxy:
        assert isinstance(proxy["array"], ArrayProxy)
    with pytest.raises(AexError):
        proxy["array"]


def test_file_proxy_close_twice(file_proxy):
    file_proxy.close()
    with pytest.raises(AexError):
        file_proxy.close()


def test_file_proxy_repr(file_proxy):
    text = repr(file_proxy)
    assert "FileProxy" in text
    assert str(file_proxy.handle) in text


def test_open_nonexistent_file(client, data_dir):
    with pytest.raises(AexNotFoundError):
        client.open(str(data_dir / "missing.npy"))


# ============================================================
# Workflows
# ============================================================


def test_workflow_open_navigate_read(client, ds_paths):
    f = client.open(ds_paths["ds1"])
    ds = f["array"]
    assert isinstance(ds, ArrayProxy)
    data = ds[10:20, 30:40]
    assert data.shape == (10, 10)
    assert data[0, 0] == 30 + 10 * 200
    f.close()


def test_workflow_iteration_and_access(file_proxy):
    for name in file_proxy:
        item = file_proxy[name]
        if isinstance(item, ArrayProxy) and item.name == "/array":
            data = item[0]
            assert data.shape == (200,)
            assert data[0] == 0.0


def test_workflow_contains_then_access(file_proxy):
    if "array" in file_proxy:
        data = file_proxy["array"][0:10]
        assert data.shape == (10, 200)
        assert data[0, 0] == 0.0


def test_workflow_other_dtypes(open_array):
    ds3 = open_array("ds3")
    assert ds3.dtype == np.int8
    data = ds3[0:5, 0:5]
    assert data.shape == (5, 5)
    assert data.dtype == np.int8


def test_workflow_multiple_accesses_same_proxy(array_proxy):
    data1 = array_proxy[0:10]
    data2 = array_proxy[50:60]
    data3 = array_proxy[[5, 10, 15]]
    assert data1.shape == (10, 200)
    assert data2.shape == (10, 200)
    assert data3.shape == (3, 200)
    assert not np.shares_memory(data1, data2)
    assert not np.shares_memory(data1, data3)


# ============================================================
# NumPy functions
#
# Every function is computed locally until the server can reduce; the values
# must match either way.
# ============================================================

TOTAL = 100 * (200 * 199 // 2) + (100 * 99 // 2) * 200 * 200


def test_numpy_sum_full_reduction(array_proxy):
    result = np.sum(array_proxy)
    assert np.ndim(result) == 0
    # The server adds in stream order and numpy pairwise, so float32 sums differ.
    assert np.isclose(result, np.sum(np.asarray(array_proxy)), rtol=1e-6)
    assert np.isclose(float(result), TOTAL)


def test_numpy_sum_axis_0(array_proxy):
    result = np.sum(array_proxy, axis=0)
    assert result.shape == (200,)
    assert result[0] == 200 * (100 * 99 // 2)
    assert result[1] == 100 * 1 + 200 * (100 * 99 // 2)


def test_numpy_sum_axis_1(array_proxy):
    result = np.sum(array_proxy, axis=1)
    assert result.shape == (100,)
    assert result[0] == 200 * 199 // 2
    assert result[5] == 200 * 199 // 2 + 40000 * 5


def test_numpy_sum_keepdims(array_proxy):
    assert np.sum(array_proxy, axis=0, keepdims=True).shape == (1, 200)


def test_numpy_mean(array_proxy):
    assert np.isclose(float(np.mean(array_proxy)), TOTAL / 20000)


def test_numpy_mean_axis(array_proxy):
    result = np.mean(array_proxy, axis=0)
    assert result.shape == (200,)
    assert np.isclose(result[0], 200 * (100 * 99 // 2) / 100)


def test_numpy_std_var(array_proxy):
    data = np.asarray(array_proxy)
    assert np.isclose(np.std(array_proxy), np.std(data))
    assert np.isclose(np.var(array_proxy), np.var(data))


def test_numpy_max_min(array_proxy):
    assert float(np.max(array_proxy)) == 199 + 99 * 200
    assert float(np.min(array_proxy)) == 0


def test_numpy_prod(open_array):
    result = np.prod(open_array("ds3"), axis=0)
    assert result.shape == (200,)
    assert (result == 1).all()


def test_numpy_argmax_argmin(array_proxy):
    assert int(np.argmax(array_proxy)) == 99 * 200 + 199
    assert int(np.argmin(array_proxy)) == 0


def test_numpy_all_any(open_array):
    ds6 = open_array("ds6")
    assert not bool(np.all(ds6))
    assert bool(np.any(ds6))


def test_numpy_function_0d_result_to_scalar(array_proxy):
    assert isinstance(np.sum(array_proxy).item(), (int, float))


def test_numpy_ufunc(array_proxy):
    result = np.sqrt(array_proxy[0:5, 0:5])
    assert result.shape == (5, 5)
    assert np.allclose(result, np.sqrt(np.asarray(array_proxy[0:5, 0:5])))
    # The proxy itself, too.
    assert np.allclose(np.sqrt(array_proxy), np.sqrt(np.asarray(array_proxy)))


@pytest.mark.parametrize(
    "func, kwargs",
    [
        (np.median, {}),
        (np.median, {"axis": 0}),
        (np.nanmedian, {}),
        (np.cumsum, {"axis": 0}),
        (np.count_nonzero, {}),
        (np.count_nonzero, {"axis": 0}),
        (np.ptp, {}),
        (np.ptp, {"axis": 1}),
        (np.sort, {"axis": 0}),
        (np.argsort, {"axis": 0}),
    ],
)
def test_numpy_functions_match_local(array_proxy, func, kwargs):
    expected = func(np.asarray(array_proxy), **kwargs)
    result = func(array_proxy, **kwargs)
    assert np.shape(result) == np.shape(expected)
    assert np.array_equal(result, expected)


def test_numpy_cumprod(array_proxy):
    result = np.cumprod(array_proxy[0:5, 0:5], axis=0)
    assert np.allclose(result, np.cumprod(np.asarray(array_proxy[0:5, 0:5]), axis=0))


def test_numpy_function_with_proxies_in_a_list(array_proxy):
    result = np.concatenate([array_proxy, array_proxy])
    assert result.shape == (200, 200)


# ============================================================
# Errors
# ============================================================


def test_connection_errors_are_os_errors():
    # Retry wrappers catch ConnectionError / OSError, not our base class.
    assert issubclass(aex.AexConnectionError, ConnectionError)
    assert issubclass(aex.AexTransferError, OSError)
    assert str(aex.AexConnectionError("down", "TRANSIENT")) == "down"


def test_is_retryable():
    assert aex.AexTransferError("later", "TRANSIENT").is_retryable
    assert not aex.AexTransferError("never").is_retryable
