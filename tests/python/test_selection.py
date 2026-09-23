"""Differential tests: every selection must match what numpy returns."""

import threading
import warnings

import numpy as np
import pytest

import aex
from aex import _aex
from aex.array_proxy import _to_wire

SHAPE = (12, 7, 5)

KEYS = [
    5,
    -1,
    np.int64(3),
    slice(None),
    slice(2, 8),
    slice(2, 8, 2),
    slice(None, None, -1),
    slice(8, 2, -3),
    slice(-3, None),
    slice(100, 200),
    slice(None, None, 5),
    [1, 5, 10],
    [-1, -2],
    [3, 3, 0],
    [],
    np.array([2, 0], dtype=np.uint8),
    Ellipsis,
    None,
    (),
    (5, slice(None)),
    (slice(None), 3),
    (Ellipsis, 0),
    (slice(None), None),
    (slice(2, 8), [1, 3, 5]),
    (0, 1, 2),
    (-1, -1, -1),
    (slice(None), slice(1, 6, 2), slice(None, None, -1)),
    ([0, 2], [1, 3]),
    ([0, 2], slice(None), [4, 1]),
    ([1], [0, 1, 2]),
    (1, slice(None), [0, 3]),
    (Ellipsis, [4, 0]),
    (None, [2, 1], None),
    ([0, 11], [6, 0], [4, 4]),
    (slice(None, None, 2), 3, [1, 2]),
    (None, Ellipsis, None),
    (Ellipsis, None, 2),
    np.arange(12) % 3 == 0,
    (slice(None), np.arange(7) > 4),
    np.arange(84).reshape(12, 7) % 5 == 0,
    (Ellipsis, np.array([True, False, True, False, True])),
    (np.arange(12) < 2, slice(None), [0, 4]),
]


@pytest.fixture(scope="module")
def source(data_dir):
    array = np.arange(np.prod(SHAPE), dtype=np.int32).reshape(SHAPE)
    path = data_dir / "selection.npy"
    np.save(path, array)
    return str(path), array


@pytest.fixture
def proxy(client, source):
    return client.open(source[0])["array"]


@pytest.mark.parametrize("key", KEYS, ids=repr)
def test_selection_matches_numpy(key, source, proxy):
    expected = source[1][key]
    actual = proxy[key]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape


@pytest.mark.parametrize("key", KEYS, ids=repr)
def test_resolve_matches_numpy(key):
    """Takes no client: a view resolves its shape without a server."""
    expected = np.empty(SHAPE, np.int32)[key]
    descr, shape = _aex.resolve(SHAPE, "<i4", _to_wire(key, SHAPE))
    assert tuple(shape) == expected.shape
    assert descr == expected.dtype.str


@pytest.mark.parametrize(
    "key",
    [12, -13, (0, 7), [0, 12], (0, 0, 0, 0), ([0, 1], [0, 1, 2])],
    ids=repr,
)
def test_resolve_refuses_what_numpy_refuses(key):
    with pytest.raises(aex.AexValueError):
        _aex.resolve(SHAPE, "<i4", _to_wire(key, SHAPE))


@pytest.mark.parametrize(
    "key",
    [12, -13, (0, 7), [0, 12], (0, 0, 0, 0), ([0, 1], [0, 1, 2])],
    ids=repr,
)
def test_selection_numpy_refuses_is_refused(key, source, proxy):
    with pytest.raises(IndexError):
        source[1][key]
    with pytest.raises(aex.AexValueError):
        proxy[key]


@pytest.mark.parametrize(
    "key",
    [1.5, "a", True, np.ones((12, 2), dtype=bool), np.zeros((2, 2), dtype=int)],
    ids=repr,
)
def test_selection_that_cannot_be_sent_is_an_index_error(key, proxy):
    with pytest.raises(IndexError):
        proxy[key]


DTYPES = [
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_every_dtype_round_trips(dtype, client, data_dir):
    rng = np.random.default_rng(0)
    array = (rng.random((40, 30)) * 100).astype(dtype)
    if np.dtype(dtype).kind == "c":
        array = array + 1j * array[::-1]
    path = data_dir / f"dtype_{np.dtype(dtype).name}.npy"
    np.save(path, array)

    proxy = client.open(str(path))["array"]
    assert proxy.dtype == array.dtype
    for key in [slice(None), (slice(None), slice(None, None, 3)), [3, 1]]:
        actual = proxy[key]
        assert actual.dtype == array.dtype
        np.testing.assert_array_equal(actual, array[key])


@pytest.fixture(scope="module")
def large(data_dir):
    # Well past the inline limit, so the bytes go over the data plane.
    array = np.arange(3_000_000, dtype=np.float64).reshape(1000, 3000)
    path = data_dir / "large.npy"
    np.save(path, array)
    return str(path), array


@pytest.mark.parametrize(
    "key",
    [
        slice(None),
        slice(100, 900),
        (slice(None), slice(None, None, 7)),
        [999, 0, 500],
        (slice(None, None, -2), 17),
    ],
    ids=repr,
)
def test_large_selection_over_the_data_plane(key, client, large):
    np.testing.assert_array_equal(client.open(large[0])["array"][key], large[1][key])


def test_threads_share_a_client(client, large):
    proxy = client.open(large[0])["array"]
    errors = []

    def read(row):
        try:
            np.testing.assert_array_equal(proxy[row::97], large[1][row::97])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=read, args=(row,)) for row in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors


# Fallback policy


@pytest.fixture
def policy():
    yield aex
    aex.set_fallback_policy("warn")
    aex.set_fallback_threshold(64 * 1024 * 1024)


def test_small_fallback_is_silent(policy, proxy):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.median(proxy)


def test_fallback_over_the_threshold_warns(policy, proxy):
    policy.set_fallback_threshold(0)
    with pytest.warns(aex.AexFallbackWarning, match=r"(?s)np\.median .*MiB") as record:
        np.median(proxy)
    # Pointing at the caller, not at aex.
    assert record[0].filename == __file__
    with pytest.warns(aex.AexFallbackWarning, match=r"np\.sqrt"):
        np.sqrt(proxy)


def test_fallback_can_be_allowed(policy, proxy):
    policy.set_fallback_threshold(0)
    policy.set_fallback_policy("allow")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.median(proxy)


def test_fallback_can_be_forbidden(policy, proxy):
    policy.set_fallback_policy("error")
    with pytest.raises(aex.AexFallbackError):
        np.median(proxy)
    # Indexing is explicit, so it is still allowed.
    assert np.sum(proxy[...]) == np.sum(np.arange(np.prod(SHAPE)))


def test_unknown_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        policy.set_fallback_policy("sometimes")


# The buffer the extension writes into


@pytest.fixture
def plan(proxy):
    key = (slice(0, 2),)
    return proxy, key, proxy._native.prepare(proxy.handle, proxy.name, key)


def fill(plan, out):
    proxy, key, p = plan
    proxy._native.fill(p, proxy.handle, proxy.name, key, out)


def test_plan_describes_the_result(plan):
    _, _, p = plan
    assert p.shape == (2, 7, 5)
    assert p.dtype == "<i4"
    assert p.total_bytes == 2 * 7 * 5 * 4
    assert p.is_inline


def test_fill_writes_a_matching_buffer(plan, source):
    out = np.empty((2, 7, 5), np.int32)
    fill(plan, out)
    np.testing.assert_array_equal(out, source[1][:2])
    # Any shape with the right dtype and length will do.
    flat = np.empty(70, np.int32)
    fill(plan, flat)
    np.testing.assert_array_equal(flat, source[1][:2].ravel())


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.empty((2, 7, 5), np.int64),
        lambda: np.empty((2, 7, 5), ">i4"),
        lambda: np.empty(69, np.int32),
        lambda: np.empty((5, 7, 2), np.int32).T,
        lambda: np.empty(140, np.int32)[::2],
    ],
    ids=["dtype", "big-endian", "length", "fortran", "strided"],
)
def test_fill_refuses_a_buffer_it_cannot_write(plan, make):
    out = make()
    with pytest.raises(aex.AexValueError):
        fill(plan, out)


def test_fill_refuses_a_read_only_buffer(plan):
    out = np.empty(70, np.int32)
    out.flags.writeable = False
    with pytest.raises(aex.AexValueError):
        fill(plan, out)


def test_extension_module_classes():
    assert _aex.Client.__module__ == "aex._aex"
