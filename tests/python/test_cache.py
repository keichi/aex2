"""The result cache: every answer it gives must be what numpy gives."""

import numpy as np
import pytest

import aex
from aex.array_proxy import _to_wire
from aex.cache import Cache

SHAPE = (6, 7, 5)
SOURCE = np.arange(np.prod(SHAPE), dtype=np.int32).reshape(SHAPE)


def _valid(make):
    """Call `make` until numpy accepts what it returns."""
    while True:
        key = make()
        try:
            SOURCE[key]
        except IndexError:
            continue
        return key


def _random_basic_key(rng):
    """Integers and slices only, with new axes and an ellipsis mixed in."""
    entries = []
    for dim in SHAPE[: rng.integers(0, len(SHAPE) + 1)]:
        if rng.random() < 0.3:
            entries.append(int(rng.integers(-dim, dim)))
        else:
            step = None if rng.random() < 0.3 else int(rng.choice([-3, -2, -1, 1, 2, 3]))
            entries.append(slice(_bound(rng, dim), _bound(rng, dim), step))
    return _sprinkle(rng, entries)


def _bound(rng, dim):
    # Past either end too: a slice clamps.
    return None if rng.random() < 0.3 else int(rng.integers(-dim - 2, dim + 2))


def _random_key(rng):
    """Anything a request may be, index arrays and masks included."""
    entries = []
    # Index arrays in one key have to broadcast.
    count = rng.integers(0, 4)
    for axis, dim in enumerate(SHAPE[: rng.integers(0, len(SHAPE) + 1)]):
        r = rng.random()
        if r < 0.25:
            entries.append(int(rng.integers(-dim, dim)))
        elif r < 0.5:
            entries.append(rng.integers(-dim, dim, size=count))
        elif r < 0.6 and axis == 0:
            entries.append(rng.random(dim) < 0.5)
        else:
            step = int(rng.choice([-2, -1, 1, 2]))
            entries.append(slice(int(rng.integers(0, dim)), None, step))
    return _sprinkle(rng, entries)


def _within(rng, cached):
    """A request inside what `cached` selected, built from its coordinates."""
    entries = []
    count = rng.integers(0, 4)
    for dim, coords in zip(SHAPE, _coords(cached), strict=True):
        n = len(coords)
        r = rng.random()
        if r < 0.3:
            i = int(coords[rng.integers(n)])
            entries.append(i - dim if rng.random() < 0.3 else i)
        elif r < 0.6:
            entries.append(rng.choice(coords, size=count))
        else:
            # Every m-th of the cached coordinates, from position p0, either way.
            m = int(rng.choice([-2, -1, 1, 2]))
            p0 = int(rng.integers(n))
            positions = np.arange(p0, n if m > 0 else -1, m)[: rng.integers(1, n + 1)]
            first, by = int(coords[positions[0]]), int(coords[1] - coords[0]) * m if n > 1 else 1
            stop = first + by * len(positions)
            entries.append(slice(first, None if stop < 0 else stop, by))
    return _sprinkle(rng, entries)


def _coords(key):
    """The source coordinates a basic key takes along each axis."""
    marker = np.arange(np.prod(SHAPE)).reshape(SHAPE)
    picked = np.asarray(marker[key])
    return [np.unique(np.unravel_index(picked.ravel(), SHAPE)[axis]) for axis in range(3)]


def _sprinkle(rng, entries):
    for _ in range(rng.integers(0, 2)):
        entries.insert(int(rng.integers(0, len(entries) + 1)), None)
    if rng.random() < 0.3:
        entries.insert(int(rng.integers(0, len(entries) + 1)), Ellipsis)
    return tuple(entries)


def _store(cache, key, quality=None):
    wire = _to_wire(key, SHAPE)
    found = cache.lookup(1, "a", SHAPE, "<i4", wire, quality)
    out = np.array(SOURCE[key])
    out.flags.writeable = False
    cache.store(1, "a", quality, found.axes, out, {"encoding": "exact"})


def _ask(cache, key, quality=None):
    return cache.lookup(1, "a", SHAPE, "<i4", _to_wire(key, SHAPE), quality).hit


def _check(hit, key):
    expected = np.asarray(SOURCE[key])
    np.testing.assert_array_equal(hit, expected)
    assert hit.shape == expected.shape
    assert hit.dtype == expected.dtype
    assert not hit.flags.writeable


def test_any_answer_matches_numpy():
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(3000):
        cache = Cache(1 << 20)
        cached = _valid(lambda: _random_basic_key(rng))
        _store(cache, cached)
        if 0 in np.asarray(SOURCE[cached]).shape:
            continue
        for key in (_within(rng, cached), _valid(lambda: _random_key(rng))):
            hit = _ask(cache, key)
            if hit is not None:
                _check(hit, key)
                hits += 1
    # The requests built inside an entry must mostly hit, or this tests little.
    assert hits > 2500


def test_a_request_inside_an_entry_always_hits():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        cache = Cache(1 << 20)
        cached = _valid(lambda: _random_basic_key(rng))
        if 0 in np.asarray(SOURCE[cached]).shape:
            continue
        _store(cache, cached)
        key = _within(rng, cached)
        hit = _ask(cache, key)
        assert hit is not None, (cached, key)
        _check(hit, key)


@pytest.mark.parametrize(
    "cached, key",
    [
        # Between the coordinates the entry has.
        (np.s_[0:6:2], np.s_[1]),
        (np.s_[0:6:2], np.s_[0:4]),
        # Past its end, or partly so.
        (np.s_[0:3], np.s_[2:5]),
        (np.s_[0:3], np.s_[[0, 4]]),
        # An axis the entry fixed, asked at another index.
        (np.s_[2], np.s_[3]),
        # The same coordinates on a different axis.
        (np.s_[:, 0:2], np.s_[:, 2:4]),
    ],
    ids=repr,
)
def test_a_request_outside_every_entry_misses(cached, key):
    cache = Cache(1 << 20)
    _store(cache, cached)
    assert _ask(cache, key) is None


def test_an_integer_kept_apart_from_an_index_array_moves_the_axis_first():
    # Dropping the fixed axis instead would give (5, 3), the transpose.
    cache = Cache(1 << 20)
    _store(cache, np.s_[3])
    key = np.s_[3, :, [0, 2, 4]]
    _check(_ask(cache, key), key)


def test_quality_must_match_exactly():
    cache = Cache(1 << 20)
    _store(cache, np.s_[:], {"abs_error": 0.1})
    assert _ask(cache, np.s_[0]) is None
    assert _ask(cache, np.s_[0], {"abs_error": 0.2}) is None
    assert _ask(cache, np.s_[0], {"abs_error": 0.1}) is not None


def test_results_of_index_arrays_are_not_stored():
    cache = Cache(1 << 20)
    _store(cache, np.s_[[0, 1]])
    assert cache.stats()["entries"] == 0


def test_the_least_recently_used_entry_goes_first():
    row = SOURCE[0].nbytes
    cache = Cache(2 * row)
    _store(cache, np.s_[0])
    _store(cache, np.s_[1])
    assert _ask(cache, np.s_[0, 0]) is not None  # 0 is now the more recent
    _store(cache, np.s_[2])
    assert _ask(cache, np.s_[1, 0]) is None
    assert _ask(cache, np.s_[0, 0]) is not None
    assert cache.stats()["evictions"] == 1


def test_an_entry_the_new_one_covers_is_dropped():
    cache = Cache(1 << 20)
    _store(cache, np.s_[1])
    _store(cache, np.s_[2:4])
    _store(cache, np.s_[0:4])
    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["bytes"] == SOURCE[0:4].nbytes
    assert stats["evictions"] == 0


def test_a_result_larger_than_the_cache_is_not_stored():
    cache = Cache(SOURCE[0].nbytes)
    _store(cache, np.s_[0])
    _store(cache, np.s_[0:2])
    assert _ask(cache, np.s_[0, 0]) is not None


def test_a_closed_file_takes_its_entries_with_it():
    cache = Cache(1 << 20)
    _store(cache, np.s_[:])
    cache.drop(1)
    assert _ask(cache, np.s_[0]) is None
    assert cache.stats()["bytes"] == 0


def test_a_disabled_cache_resolves_nothing():
    cache = Cache(0)
    found = cache.lookup(1, "a", SHAPE, "<i4", _to_wire(np.s_[0], SHAPE), None)
    assert found.hit is None and found.axes is None


# Through a client and a server: what is answered here is not transferred.


@pytest.fixture
def cached(server, ds_paths):
    with aex.Client(server, cache_bytes=1 << 20) as client:
        f = client.open(ds_paths["ds1"])
        yield client, f, f["array"], np.load(ds_paths["ds1"])


def _sent(client):
    return client.stats()["bytes"]


def test_a_read_inside_an_earlier_one_is_not_transferred(cached):
    client, _, arr, source = cached
    arr[0:50]
    before = _sent(client)
    key = np.s_[10:20:2, [5, 0, 7]]
    np.testing.assert_array_equal(arr[key], source[key])
    assert _sent(client) == before
    assert client.stats()["cache"]["hits"] == 1


def test_a_view_reads_through_the_cache(cached):
    client, _, arr, source = cached
    arr[:]
    before = _sent(client)
    np.testing.assert_array_equal(arr.view[0:50][3], source[0:50][3])
    assert _sent(client) == before


def test_gather_transfers_only_what_it_misses(cached):
    client, _, arr, source = cached
    arr[0:10]
    before = _sent(client)
    a, b = arr.gather([np.s_[2:4], np.s_[60:62]])
    np.testing.assert_array_equal(a, source[2:4])
    np.testing.assert_array_equal(b, source[60:62])
    assert _sent(client) - before == b.nbytes
    # The miss was stored like any other.
    arr[61]
    assert _sent(client) - before == b.nbytes


def test_read_into_copies_a_hit_but_never_stores(cached):
    client, _, arr, source = cached
    buf = np.empty(200, np.float32)
    arr.read_into(buf, np.s_[3])
    arr[3]
    assert client.stats()["cache"]["hits"] == 0
    before = _sent(client)
    out = np.empty(200, np.float32)
    arr.read_into(out, np.s_[3])
    np.testing.assert_array_equal(out, source[3])
    assert _sent(client) == before


def test_read_into_leaves_a_buffer_it_cannot_take_to_the_transfer(cached):
    _, _, arr, _ = cached
    arr[:]
    with pytest.raises(aex.AexValueError):
        arr.read_into(np.empty(199, np.float32), np.s_[3])


def test_get_async_reads_through_the_cache(cached):
    client, _, arr, source = cached
    arr[:]
    before = _sent(client)
    np.testing.assert_array_equal(arr.get_async(np.s_[5:7]).result(), source[5:7])
    assert _sent(client) == before


def test_a_quality_reads_only_what_was_read_at_it(cached):
    client, _, arr, _ = cached
    view = arr.at(codec="gzip")
    view[0:10]
    before = _sent(client)
    view[1]
    assert _sent(client) == before
    arr[1]
    assert _sent(client) > before


def test_closing_the_file_forgets_its_results(cached):
    client, f, arr, _ = cached
    arr[:]
    assert client.stats()["cache"]["entries"] == 1
    f.close()
    assert client.stats()["cache"]["entries"] == 0


def test_a_negative_capacity_is_refused(server):
    with pytest.raises(ValueError):
        aex.Client(server, cache_bytes=-1)
