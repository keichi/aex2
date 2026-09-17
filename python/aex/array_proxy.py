"""ArrayProxy: a remote array that indexes like a numpy array."""

import operator
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from . import _aex
from .errors import AexFallbackError, AexFallbackWarning

__all__ = ["ArrayProxy", "set_fallback_policy", "set_fallback_threshold"]

FallbackPolicy = Literal["warn", "allow", "error"]

_fallback_policy: FallbackPolicy = "warn"
_fallback_threshold = 64 * 1024 * 1024


def set_fallback_policy(policy: FallbackPolicy) -> None:
    """What to do when a numpy function has to download an array.

    ``"warn"`` warns above the threshold, ``"allow"`` stays silent, and
    ``"error"`` refuses every such download.
    """
    global _fallback_policy
    if policy not in ("warn", "allow", "error"):
        raise ValueError(f"unknown fallback policy {policy!r}")
    _fallback_policy = policy


def set_fallback_threshold(nbytes: int) -> None:
    """Downloads larger than this warn under the ``"warn"`` policy."""
    global _fallback_threshold
    _fallback_threshold = operator.index(nbytes)


Key = tuple[Any, ...]

# The server's default max_transfers_per_session: a larger batch would evict
# its own plans before they are fetched.
_GATHER_BATCH = 64


class ArrayProxy:
    """An array on the server. Indexing it transfers the selection."""

    def __init__(
        self,
        native: _aex.Client,
        handle: int,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
    ) -> None:
        self._native = native
        self.handle = handle
        self.name = name
        self.dtype = np.dtype(dtype)
        self.shape = shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __iter__(self) -> Iterator[npt.NDArray[Any]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key: Any) -> npt.NDArray[Any]:
        """Transfer a selection. The result is read-only."""
        wire_key = _to_wire(key, self.shape)
        plan = self._native.prepare(self.handle, self.name, wire_key)
        # Shape and dtype are the server's answer, not worked out here.
        out = np.empty(plan.shape, dtype=plan.dtype)
        self._native.fill(plan, self.handle, self.name, wire_key, out)
        out.flags.writeable = False
        return out

    def read_into(self, out: npt.NDArray[Any], key: Any = Ellipsis) -> None:
        """Transfer a selection into ``out`` without allocating.

        ``out`` must be C-contiguous, writable, of the array's dtype and exactly
        the selection's size; nothing is copied to make it fit. If the transfer
        fails, the contents of ``out`` are undefined.
        """
        wire_key = _to_wire(key, self.shape)
        plan = self._native.prepare(self.handle, self.name, wire_key)
        self._native.fill(plan, self.handle, self.name, wire_key, out)

    def gather(self, keys: Sequence[Any]) -> list[npt.NDArray[Any]]:
        """Transfer several selections, resolving them in one round trip.

        Raises the error of the first selection that fails. The results are
        read-only.
        """
        results: list[npt.NDArray[Any]] = []
        for start in range(0, len(keys), _GATHER_BATCH):
            wire_keys = [_to_wire(key, self.shape) for key in keys[start : start + _GATHER_BATCH]]
            plans = self._native.prepare_many(self.handle, self.name, wire_keys)
            for plan in plans:
                if isinstance(plan, BaseException):
                    raise plan
            ready = cast(list[_aex.Plan], plans)
            outs = [np.empty(plan.shape, dtype=plan.dtype) for plan in ready]
            self._native.fill_many(ready, self.handle, self.name, wire_keys, outs)
            for out in outs:
                out.flags.writeable = False
            results.extend(outs)
        return results

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray[Any]:
        data = self[...]
        if dtype is not None:
            data = data.astype(dtype, copy=False)
        return data.copy() if copy else data

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # ponytail: every function runs locally until the server can reduce
        # (ApplyFunction); dispatch the supported ones there then.
        args, kwargs = _download(func, (args, kwargs))
        return func(*args, **kwargs)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        inputs, kwargs = _download(ufunc, (inputs, kwargs))
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __repr__(self) -> str:
        return f'<ArrayProxy name "{self.name}", shape {self.shape}, type {self.dtype}>'


def _to_wire(key: Any, shape: tuple[int, ...]) -> Key:
    """Turn a numpy key into what the server takes.

    Only spelling is changed here; normalising and checking against the shape
    is the server's job. The exception is a boolean mask, which becomes one
    index array per axis it covers and is checked against those axes, since
    the server never sees it.
    """
    entries = key if isinstance(key, tuple) else (key,)

    # How many axes each entry consumes, to place the masks.
    converted: list[Any] = []
    widths: list[int] = []
    for entry in entries:
        if entry is None or entry is Ellipsis:
            converted.append(entry)
            widths.append(0)
        elif isinstance(entry, slice):
            converted.append(
                slice(
                    *(
                        None if v is None else operator.index(v)
                        for v in (entry.start, entry.stop, entry.step)
                    )
                )
            )
            widths.append(1)
        elif isinstance(entry, (bool, np.bool_)):
            raise IndexError("a scalar boolean index is not supported")
        elif isinstance(entry, (int, np.integer)):
            converted.append(operator.index(entry))
            widths.append(1)
        else:
            array = np.asarray(entry)
            if array.dtype == np.bool_:
                converted.append(array)
                widths.append(max(array.ndim, 1))
            elif array.dtype.kind in "iu" or array.size == 0:
                if array.ndim == 0:
                    converted.append(int(array))
                elif array.ndim == 1:
                    converted.append(array.astype(np.int64))
                else:
                    raise IndexError(
                        "index arrays with more than one dimension are not supported; "
                        "flatten the array and reshape the result"
                    )
                widths.append(1)
            else:
                raise IndexError(
                    "only integers, slices, ellipsis, None, and integer or boolean "
                    f"arrays are valid indices, not {type(entry).__name__}"
                )

    implied = len(shape) - sum(widths)
    wire: list[Any] = []
    axis = 0
    for entry, width in zip(converted, widths, strict=True):
        if entry is Ellipsis:
            axis += max(implied, 0)
        if isinstance(entry, np.ndarray) and entry.dtype == np.bool_:
            covered = shape[axis : axis + width]
            if entry.shape != covered:
                raise IndexError(
                    f"boolean index of shape {entry.shape} does not match "
                    f"the indexed axes of shape {covered}"
                )
            wire.extend(i.astype(np.int64) for i in np.nonzero(entry))
        else:
            wire.append(entry)
        axis += width
    return tuple(wire)


def _download(func: Any, tree: Any) -> Any:
    """Replace every ArrayProxy in ``tree`` with its data, as the policy allows."""
    proxies: list[ArrayProxy] = []
    _collect(tree, proxies)
    nbytes = sum(p.nbytes for p in {id(p): p for p in proxies}.values())
    name = f"np.{getattr(func, '__name__', func)}"

    if _fallback_policy == "error":
        raise AexFallbackError(
            f"{name} is not supported server-side, and the fallback policy forbids "
            f"downloading {nbytes / 2**20:.1f} MiB to compute locally.",
            "REQUEST",
        )
    if _fallback_policy == "warn" and nbytes > _fallback_threshold:
        warnings.warn(
            f"{name} is not supported server-side.\n"
            f"  Downloading {nbytes / 2**20:.1f} MiB to compute locally.\n"
            "  Use arr[...] explicitly to silence, or set aex.set_fallback_policy(...).",
            AexFallbackWarning,
            stacklevel=3,
        )
    return _replace(tree)


def _collect(tree: Any, found: list[ArrayProxy]) -> None:
    if isinstance(tree, ArrayProxy):
        found.append(tree)
    elif isinstance(tree, (list, tuple)):
        for item in tree:
            _collect(item, found)
    elif isinstance(tree, dict):
        for item in tree.values():
            _collect(item, found)


def _replace(tree: Any) -> Any:
    if isinstance(tree, ArrayProxy):
        return tree[...]
    if isinstance(tree, (list, tuple)):
        return type(tree)(_replace(item) for item in tree)
    if isinstance(tree, dict):
        return {key: _replace(item) for key, item in tree.items()}
    return tree
