"""Server-side reductions, checked against numpy.

Integers and booleans must match exactly; floating results within a tolerance,
since numpy adds pairwise and the server adds in stream order.
"""

import warnings

import numpy as np
import pytest

import aex

SHAPE = (6, 5, 4)
DTYPES = ["?", "i1", "i4", "i8", "u1", "u8", "f2", "f4", "f8", "c8", "c16"]
RTOL = {"f": {2: 1e-3, 4: 1e-5, 8: 1e-10}, "c": {8: 1e-5, 16: 1e-10}}
# numpy sums float16 in float16, rounding at every step; up to 24 steps of
# about 1e-3 each here, which relative tolerance cannot absorb near zero.
F16_ATOL = 0.03

FUNCTIONS = [
    np.sum,
    np.prod,
    np.mean,
    np.max,
    np.min,
    np.amax,
    np.std,
    np.var,
    np.all,
    np.any,
    np.argmax,
    np.argmin,
    np.nansum,
    np.nanmean,
    np.nanmax,
    np.nanmin,
]
AXES = [None, 0, 1, -1, (0, 2), (), (2, 1, 0)]


def make(dtype: str, nan: bool) -> np.ndarray:
    rng = np.random.default_rng(len(dtype) + ord(dtype[0]))
    dt = np.dtype(dtype)
    size = int(np.prod(SHAPE))
    if dt.kind == "b":
        data = rng.random(size) < 0.7
    elif dt.kind in "iu":
        info = np.iinfo(dt)
        data = rng.integers(info.min, info.max, size, dtype=dt, endpoint=True)
    else:
        # Near one, so that a product neither overflows nor vanishes in float16.
        data = rng.uniform(0.8, 1.25, size) * rng.choice([-1, 1], size)
        if dt.kind == "c":
            data = data + 1j * rng.uniform(-1, 1, size)
        data = data.astype(dt)
        if nan:
            data[rng.random(size) < 0.2] = np.nan
    data = data.reshape(SHAPE)
    if nan:
        # A slice that is nothing but NaN.
        data[2, :, 1] = np.nan
    return data


@pytest.fixture(scope="module")
def arrays(client_module, data_dir):
    """(numpy array, proxy) per dataset name."""
    out = {}
    for dtype in DTYPES:
        for nan in (False, True):
            if nan and np.dtype(dtype).kind not in "fc":
                continue
            name = f"reduce-{dtype}{'-nan' if nan else ''}"
            data = make(dtype, nan)
            path = data_dir / f"{name}.npy"
            np.save(path, data)
            out[name] = (data, client_module.open(str(path))["array"])
    return out


@pytest.fixture(scope="module")
def client_module(server):
    with aex.Client(server) as client:
        yield client


def outcome(func, target, **kwargs):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return func(target, **kwargs)
        except Exception as e:  # noqa: BLE001
            return e


def check(expected, actual):
    if isinstance(expected, Exception):
        kind = next(k for k in (TypeError, ValueError, IndexError) if isinstance(expected, k))
        assert isinstance(actual, kind), (expected, actual)
        return
    assert not isinstance(actual, Exception), actual
    # A scalar where numpy gives one, an array where it gives one.
    assert type(actual) is type(expected)
    expected, actual_arr = np.asarray(expected), np.asarray(actual)
    assert actual_arr.dtype == expected.dtype
    assert actual_arr.shape == expected.shape
    kind = expected.dtype.kind
    if kind in "fc":
        size = expected.dtype.itemsize
        atol = F16_ATOL if expected.dtype == np.float16 else 0
        np.testing.assert_allclose(
            actual_arr, expected, rtol=RTOL[kind][size], atol=atol, equal_nan=True
        )
    else:
        np.testing.assert_array_equal(actual_arr, expected)


def cases():
    for dtype in DTYPES:
        for nan in (False, True):
            if nan and np.dtype(dtype).kind not in "fc":
                continue
            name = f"reduce-{dtype}{'-nan' if nan else ''}"
            for func in FUNCTIONS:
                for axis in AXES:
                    for keepdims in (False, True):
                        yield pytest.param(
                            name,
                            func,
                            {"axis": axis, "keepdims": keepdims},
                            id=f"{name}-{func.__name__}-{axis}-{keepdims}",
                        )


@pytest.mark.parametrize("name, func, kwargs", list(cases()))
def test_reduction_matches_numpy(arrays, name, func, kwargs):
    data, proxy = arrays[name]
    check(outcome(func, data, **kwargs), outcome(func, proxy, **kwargs))


@pytest.mark.parametrize("func", [np.std, np.var])
@pytest.mark.parametrize("ddof", [0, 1, 2.0, 6])
@pytest.mark.parametrize("name", ["reduce-f8", "reduce-i4", "reduce-c16"])
def test_ddof_matches_numpy(arrays, func, ddof, name):
    data, proxy = arrays[name]
    for axis in (None, 0, (1, 2)):
        check(outcome(func, data, axis=axis, ddof=ddof), outcome(func, proxy, axis=axis, ddof=ddof))


@pytest.mark.parametrize(
    "key",
    [np.s_[0:3], np.s_[1], np.s_[:, ::2, [0, 3]], np.s_[..., None, 2], np.s_[4:4]],
    ids=["slice", "single", "fancy", "newaxis", "empty"],
)
@pytest.mark.parametrize("func", [np.sum, np.max, np.argmin, np.mean])
def test_a_view_reduces_its_selection(arrays, key, func):
    data, proxy = arrays["reduce-f8"]
    view = proxy.view[key]
    assert view.shape == data[key].shape
    assert view.dtype == data.dtype
    for axis in (None, 0, -1):
        check(outcome(func, data[key], axis=axis), outcome(func, view, axis=axis))


def test_a_reduction_transfers_nothing(client, ds_paths):
    arr = client.open(ds_paths["ds1"])["array"]
    np.sum(arr.view[10:90], axis=0)
    np.argmax(arr)
    assert client.stats()["bytes"] == 0


def test_a_large_result_is_computed_locally(client, ds_paths):
    arr = client.open(ds_paths["ds1"])["array"]
    # 20000 float64 elements are past the server's inline limit.
    result = np.mean(arr, axis=())
    np.testing.assert_array_equal(result, np.asarray(arr).astype(np.float64))
    assert client.stats()["bytes"] > 0


@pytest.mark.parametrize(
    "kwargs",
    [{"dtype": np.float64}, {"where": True}, {"initial": 0}],
)
def test_arguments_the_server_lacks_run_locally(client, ds_paths, kwargs):
    arr = client.open(ds_paths["ds1"])["array"]
    expected = np.sum(np.asarray(arr), **kwargs)
    assert np.sum(arr, **kwargs) == expected
    assert client.stats()["bytes"] > 0


def test_positional_arguments_reach_the_server(client, ds_paths):
    arr = client.open(ds_paths["ds1"])["array"]
    data = np.asarray(arr)
    np.testing.assert_allclose(np.sum(arr, 0), np.sum(data, 0))
    np.testing.assert_allclose(np.std(arr, 1, None, None, 1), np.std(data, 1, None, None, 1))


def test_warnings_follow_numpy(arrays):
    data, proxy = arrays["reduce-f8-nan"]
    with pytest.warns(RuntimeWarning, match="All-NaN"):
        np.nanmax(proxy, axis=(1,))
    with pytest.warns(RuntimeWarning, match="empty slice"):
        np.mean(proxy.view[0:0])
    with pytest.warns(RuntimeWarning, match="Degrees of freedom"):
        np.var(proxy, axis=0, ddof=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.nanmax(proxy)


def test_indexing_a_view_downloads_it(arrays):
    data, proxy = arrays["reduce-i4"]
    view = proxy.view[1:4]
    np.testing.assert_array_equal(view[1, ::2], data[1:4][1, ::2])
    np.testing.assert_array_equal(np.asarray(view), data[1:4])
    np.testing.assert_array_equal(list(view), list(data[1:4]))
    np.testing.assert_array_equal(np.cumsum(view), np.cumsum(data[1:4]))


def test_a_view_forbids_what_needs_the_whole_array(arrays):
    _, proxy = arrays["reduce-i4"]
    view = proxy.view[1:4]
    for call in (
        lambda: view.view[0],
        lambda: view.gather([0]),
        lambda: view.read_into(np.empty(0)),
        lambda: view.at(dtype="f4"),
    ):
        with pytest.raises(TypeError):
            call()
    assert "view" in repr(view)


def test_indexing_a_view_follows_the_fallback_policy(arrays):
    _, proxy = arrays["reduce-i4"]
    aex.set_fallback_policy("error")
    try:
        with pytest.raises(aex.AexFallbackError):
            proxy.view[1:4][0]
    finally:
        aex.set_fallback_policy("warn")
