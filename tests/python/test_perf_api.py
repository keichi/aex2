"""The performance APIs: read_into, gather, get_async, at, stats."""

import warnings

import numpy as np
import pytest

import aex


def expected_ds1() -> np.ndarray:
    y, x = np.meshgrid(np.arange(100), np.arange(200), indexing="ij")
    return (x + y * 200).astype(np.float32)


# ============================================================
# read_into
# ============================================================


@pytest.mark.parametrize("key", [np.s_[0:10], np.s_[5], np.s_[:, ::3], np.s_[[1, 7], 2:4]])
def test_read_into_fills_the_buffer(array_proxy, key):
    expected = expected_ds1()[key]
    out = np.full(expected.shape, -1, np.float32)
    array_proxy.read_into(out, key)
    np.testing.assert_array_equal(out, expected)


def test_read_into_defaults_to_everything(array_proxy):
    out = np.empty((100, 200), np.float32)
    array_proxy.read_into(out)
    np.testing.assert_array_equal(out, expected_ds1())


def test_read_into_goes_over_the_data_plane_too(array_proxy):
    # 100 * 200 * 4 bytes is over the inline limit.
    out = np.empty((100, 200), np.float32)
    for _ in range(3):
        out[:] = 0
        array_proxy.read_into(out, np.s_[:])
        np.testing.assert_array_equal(out, expected_ds1())


@pytest.mark.parametrize(
    "out",
    [
        np.empty((10, 200), np.float64),  # wrong dtype, wrong size
        np.empty((10, 200), np.int32),  # wrong dtype, right size
        np.empty((10, 100), np.float32),  # too small
        np.empty((10, 400), np.float32)[:, ::2],  # not contiguous
        np.empty((10, 200), ">f4"),  # big-endian
    ],
)
def test_read_into_refuses_a_buffer_it_cannot_write_straight_into(array_proxy, out):
    with pytest.raises(aex.AexValueError):
        array_proxy.read_into(out, np.s_[0:10])


def test_read_into_refuses_a_read_only_buffer(array_proxy):
    out = np.empty((10, 200), np.float32)
    out.flags.writeable = False
    with pytest.raises(aex.AexValueError):
        array_proxy.read_into(out, np.s_[0:10])


# ============================================================
# gather
# ============================================================


def test_gather_matches_indexing_one_by_one(array_proxy):
    keys = [np.s_[0:10], np.s_[5], np.s_[:], np.s_[..., 3], np.s_[[1, 50], ::7], np.s_[2:2]]
    results = array_proxy.gather(keys)
    assert len(results) == len(keys)
    for key, result in zip(keys, results, strict=True):
        expected = expected_ds1()[key]
        np.testing.assert_array_equal(result, expected)
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        assert not result.flags.writeable


def test_gather_of_nothing_is_nothing(array_proxy):
    assert array_proxy.gather([]) == []


def test_gather_spans_several_batches(array_proxy):
    # More than one PrepareSelections call's worth, many of them remote.
    keys = [np.s_[i % 100 : i % 100 + 1 + i % 3] for i in range(150)]
    for key, result in zip(keys, array_proxy.gather(keys), strict=True):
        np.testing.assert_array_equal(result, expected_ds1()[key])


def test_gather_raises_the_first_failure(array_proxy):
    with pytest.raises(aex.AexValueError, match="500"):
        array_proxy.gather([np.s_[0], np.s_[500], np.s_[600]])


# ============================================================
# get_async
# ============================================================


def test_get_async_gives_what_indexing_would(array_proxy):
    keys = [np.s_[i : i + 30] for i in range(0, 100, 10)]
    futures = [array_proxy.get_async(key) for key in keys]
    for key, future in zip(keys, futures, strict=True):
        np.testing.assert_array_equal(future.result(timeout=30), expected_ds1()[key])


def test_get_async_reports_failure_through_the_future(array_proxy):
    future = array_proxy.get_async(np.s_[500])
    with pytest.raises(aex.AexValueError):
        future.result(timeout=30)


def test_close_waits_for_pending_transfers(server, ds_paths):
    client = aex.Client(server)
    arr = client.open(ds_paths["ds1"])["array"]
    futures = [arr.get_async(np.s_[:]) for _ in range(8)]
    client.close()
    for future in futures:
        np.testing.assert_array_equal(future.result(timeout=0), expected_ds1())


# ============================================================
# at
# ============================================================


@pytest.mark.parametrize(
    # float64 is not a narrowing of float32, and this server has no codec for
    # an error bound.
    "kwargs",
    [{"dtype": np.float64}, {"abs_error": 1e-3}, {"rel_error": 1e-2}],
)
def test_at_falls_back_to_exact_and_warns(array_proxy, kwargs):
    view = array_proxy.at(**kwargs)
    assert view.applied_quality is None
    with pytest.warns(aex.AexQualityWarning):
        data = view[0:10]
    np.testing.assert_array_equal(data, expected_ds1()[0:10])
    assert data.dtype == np.float32
    assert view.applied_quality == {"encoding": "exact", "codec": "raw"}


def test_at_warns_once_per_view(array_proxy):
    view = array_proxy.at(dtype="f8")
    with pytest.warns(aex.AexQualityWarning):
        view[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        view[1]


@pytest.mark.parametrize("kwargs", [{}, {"dtype": "f2", "abs_error": 1e-3}])
def test_at_takes_at_most_one_kind_of_quality(array_proxy, kwargs):
    with pytest.raises(ValueError):
        array_proxy.at(**kwargs)


def test_at_carries_the_codec_alongside_the_bound(array_proxy):
    # The codec says how a bound travels, not which quality is asked for, so
    # it rides along with abs_error rather than counting as a second kind.
    assert array_proxy.at(abs_error=1e-3, codec="zfp").quality == {
        "abs_error": 1e-3,
        "codec": "zfp",
    }


def test_at_rejects_a_codec_that_is_not_one(array_proxy):
    with pytest.raises(ValueError):
        array_proxy.at(abs_error=1e-3, codec="deflate")[0:10]


def test_a_cast_narrows_the_elements(array_proxy):
    # float32 to float16 halves the transfer; the array that comes back is
    # float16 too, so the saving is kept rather than undone on arrival.
    view = array_proxy.at(dtype="f2")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = view[:]
    assert data.dtype == np.float16
    assert view.applied_quality == {
        "encoding": "dtype_cast",
        "dtype": "<f2",
        "codec": "raw",
    }
    np.testing.assert_array_equal(data, expected_ds1().astype(np.float16))


def test_gzip_is_asked_for_by_codec_alone(array_proxy):
    # It is lossless, so there is no quality to go with it and no warning:
    # only the wire is smaller.
    view = array_proxy.at(codec="gzip")
    assert view.quality == {"codec": "gzip"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = view[:]
    np.testing.assert_array_equal(data, expected_ds1())
    assert view.applied_quality == {"encoding": "exact", "codec": "gzip"}


def test_at_accepts_both_error_bounds(array_proxy):
    assert array_proxy.at(abs_error=1, rel_error=0.1).quality == {
        "abs_error": 1.0,
        "rel_error": 0.1,
    }


# ============================================================
# stats
# ============================================================


def test_stats_count_what_was_transferred(client, ds_paths):
    before = client.stats()
    assert before["bytes"] == 0
    assert before["streams"] >= 1
    # Connect is a call, so there is already a round trip to report.
    assert before["rtt_ms"] > 0

    arr = client.open(ds_paths["ds1"])["array"]
    arr[:]
    arr[0]
    after = client.stats()
    assert after["bytes"] == 100 * 200 * 4 + 200 * 4
    assert after["chunks"] >= 1
    assert after["elapsed"] > 0
    assert after["throughput_mibps"] > 0
    assert after["rtt_ms"] <= before["rtt_ms"]
    assert set(after) == {
        "bytes",
        "elapsed",
        "throughput_mibps",
        "streams",
        "chunks",
        "retries",
        "rtt_ms",
    }
