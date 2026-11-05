"""Results already transferred, and the requests they can answer.

An entry is the result of a selection made only of integers and slices, so
each of its axes is an arithmetic progression of source coordinates. A later
request whose coordinates all fall on those progressions is cut out of it with
numpy instead of crossing the network.
"""

import threading
from collections import OrderedDict
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from . import _aex
from .errors import AexError

Key = tuple[Any, ...]
# (start, step, len) of the source coordinates along one axis.
Progression = tuple[int, int, int]


class _Entry(NamedTuple):
    handle: int
    name: str
    quality: Any
    axes: tuple[Progression, ...]
    # The result with its integer axes kept at length 1 and its new axes
    # dropped, so that it has one axis per source axis.
    data: npt.NDArray[Any]
    applied: dict[str, Any]


class Lookup(NamedTuple):
    """What a request found: the answer on a hit, and its resolved axes."""

    hit: npt.NDArray[Any] | None
    applied: dict[str, Any] | None
    axes: list[tuple[Any, ...]] | None


_MISS = Lookup(None, None, None)


class Cache:
    """A byte-bounded LRU of transferred results, shared by a client's arrays.

    Holds the results themselves, which are read-only, so storing one copies
    nothing. The bound counts what the cache refers to; a result the caller
    still holds stays in memory after it is evicted.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._lock = threading.Lock()
        # Least recently used first.
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._next_id = 0
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def lookup(
        self, handle: int, name: str, shape: tuple[int, ...], dtype: str, key: Key, quality: Any
    ) -> Lookup:
        """Answer `key` from an entry, or say how it resolved for `store`."""
        if self.capacity == 0:
            return _MISS
        try:
            _, _, axes = _aex.resolve(shape, dtype, key)
        except AexError:
            # The server raises the error the caller expects.
            return _MISS
        wanted = [_as_int64(axis) for axis in axes]
        quality = _quality_key(quality)
        with self._lock:
            best = None
            for entry_id, entry in self._entries.items():
                if (entry.handle, entry.name, entry.quality) != (handle, name, quality):
                    continue
                if best is not None and entry.data.nbytes >= best[1].data.nbytes:
                    continue
                if all(
                    _contains(have, want) for have, want in zip(entry.axes, wanted, strict=True)
                ):
                    best = (entry_id, entry)
            if best is None:
                self._misses += 1
                return Lookup(None, None, wanted)
            self._entries.move_to_end(best[0])
            self._hits += 1
        entry = best[1]
        # A transfer of a single element is a 0-d array, not a numpy scalar.
        hit = np.asarray(entry.data[_translate(key, entry.axes, wanted)])
        hit.flags.writeable = False
        return Lookup(hit, entry.applied, wanted)

    def store(
        self,
        handle: int,
        name: str,
        quality: Any,
        axes: list[tuple[Any, ...]] | None,
        out: npt.NDArray[Any],
        applied: dict[str, Any],
    ) -> None:
        """Keep `out`, the result of a request `lookup` missed."""
        if axes is None or out.nbytes > self.capacity:
            return
        if any(axis[0] == "fancy" for axis in axes):
            # Index arrays pick points, not a product of axes.
            return
        progressions = tuple(_progression(axis) for axis in axes)
        data = out.reshape([n for _, _, n in progressions])
        entry = _Entry(handle, name, _quality_key(quality), progressions, data, applied)
        with self._lock:
            # An entry the new one covers would never be the smallest to answer.
            for entry_id, old in list(self._entries.items()):
                if old[:3] == entry[:3] and all(
                    _contains(new, ("range", s, n, k))
                    for new, (s, k, n) in zip(progressions, old.axes, strict=True)
                ):
                    self._remove(entry_id)
            self._entries[self._next_id] = entry
            self._next_id += 1
            self._bytes += data.nbytes
            while self._bytes > self.capacity:
                self._remove(next(iter(self._entries)))
                self._evictions += 1

    def drop(self, handle: int) -> None:
        """Forget a file's entries, when it is closed."""
        with self._lock:
            for entry_id, entry in list(self._entries.items()):
                if entry.handle == handle:
                    self._remove(entry_id)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "entries": len(self._entries),
                "bytes": self._bytes,
            }

    def _remove(self, entry_id: int) -> None:
        self._bytes -= self._entries.pop(entry_id).data.nbytes


def _quality_key(quality: dict[str, Any] | None) -> Any:
    return None if quality is None else tuple(sorted(quality.items()))


def _as_int64(axis: tuple[Any, ...]) -> tuple[Any, ...]:
    # Unsigned indices would wrap when an offset is subtracted.
    if axis[0] == "fancy":
        return ("fancy", axis[1].astype(np.int64))
    return axis


def _progression(axis: tuple[Any, ...]) -> Progression:
    if axis[0] == "point":
        return (axis[1], 1, 1)
    _, start, length, step = axis
    return (start, step, length)


def _offset(have: Progression, i: int) -> int | None:
    """Where coordinate `i` sits in `have`, if it is one of its coordinates."""
    start, step, length = have
    position, rest = divmod(i - start, step)
    return position if rest == 0 and 0 <= position < length else None


def _contains(have: Progression, want: tuple[Any, ...]) -> bool:
    kind = want[0]
    if kind == "point":
        return _offset(have, want[1]) is not None
    if kind == "range":
        _, start, length, step = want
        if length == 0:
            return True
        last = start + step * (length - 1)
        if _offset(have, start) is None or _offset(have, last) is None:
            return False
        # Both ends on the progression and a step that is a multiple of its
        # step put every coordinate in between on it too.
        return bool(length == 1 or step % have[1] == 0)
    indices = want[1]
    if indices.size == 0:
        return True
    start, step, length = have
    lo, hi = sorted((start, start + step * (length - 1)))
    if indices.min() < lo or indices.max() > hi:
        return False
    return bool(((indices - start) % step == 0).all())


def _translate(key: Key, have: tuple[Progression, ...], wanted: list[tuple[Any, ...]]) -> Key:
    """`key` rewritten for the entry, with the same structure.

    Each entry keeps its kind and position, and so does every ellipsis and new
    axis: where numpy puts the dimension of an index array depends on nothing
    else, so the result comes out in the order `dataset[key]` would have.
    """
    consuming = sum(1 for k in key if k is not None and k is not Ellipsis)
    implied = len(have) - consuming
    axis = 0
    out: list[Any] = []

    def take() -> Any:
        nonlocal axis
        rewritten = _rewrite(have[axis], wanted[axis])
        axis += 1
        return rewritten

    for k in key:
        if k is None:
            out.append(None)
        elif k is Ellipsis:
            out.extend(take() for _ in range(implied))
            # Kept, standing for no axes now, because it still separates.
            out.append(Ellipsis)
        else:
            out.append(take())
    while axis < len(have):
        out.append(take())
    return tuple(out)


def _rewrite(have: Progression, want: tuple[Any, ...]) -> Any:
    start, step, _ = have
    kind = want[0]
    if kind == "point":
        return (want[1] - start) // step
    if kind == "range":
        _, first, length, by = want
        if length == 0:
            return slice(0, 0)
        at = (first - start) // step
        by = by // step if length > 1 else 1
        stop = at + by * length
        return slice(at, None if stop < 0 else stop, by)
    return (want[1] - start) // step
