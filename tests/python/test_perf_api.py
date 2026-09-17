"""The performance APIs: read_into, stats."""

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
