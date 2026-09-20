"""ArrayProxy: a remote array that indexes like a numpy array."""

import inspect
import math
import operator
import warnings
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt

from . import _aex
from .errors import AexFallbackError, AexFallbackWarning, AexQualityWarning

if TYPE_CHECKING:
    from .client import Client

__all__ = ["ArrayProxy", "QualityView", "set_fallback_policy", "set_fallback_threshold"]

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

# The server's default inline_limit_bytes, past which it refuses a reduction.
_INLINE_LIMIT = 64 * 1024

# numpy functions the server computes, by the name it knows them by.
_SERVER_FUNCTIONS: dict[Callable[..., Any], str] = {
    getattr(np, name): name
    for name in (
        "sum",
        "prod",
        "mean",
        "max",
        "min",
        "std",
        "var",
        "all",
        "any",
        "argmax",
        "argmin",
        "nansum",
        "nanmean",
        "nanmax",
        "nanmin",
    )
}
_SERVER_FUNCTIONS[np.amax] = "max"
_SERVER_FUNCTIONS[np.amin] = "min"
_SERVER_ARGUMENTS = {"axis", "keepdims", "ddof"}

# Returned by a reduction that has to run locally instead.
_LOCAL = object()


class ArrayProxy:
    """An array on the server. Indexing it transfers the selection.

    ``arr.view[key]`` is a proxy for a selection that transfers nothing, so
    that ``np.sum(arr.view[0:100])`` reads 100 rows on the server.
    """

    def __init__(
        self,
        client: "Client",
        handle: int,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        key: Key | None = None,
    ) -> None:
        self._client = client
        self._native = client._native
        self.handle = handle
        self.name = name
        self.dtype = np.dtype(dtype)
        self.shape = shape
        # The selection of the array this is a view of, in wire form.
        self._key = key

    @property
    def view(self) -> "_Viewer":
        """Select without transferring: ``arr.view[key]``."""
        self._require_base("view")
        return _Viewer(self)

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
        if self._key is not None:
            yield from self[...]
            return
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key: Any) -> npt.NDArray[Any]:
        """Transfer a selection. The result is read-only.

        A view is transferred whole and ``key`` applied to it here, since the
        server cannot select from a selection.
        """
        if self._key is not None:
            _check_fallback(f"indexing {self!r}", self.nbytes)
            return np.asarray(self._fetch_all()[key])
        return self._read(key, None)[0]

    def _read(self, key: Any, quality: dict[str, Any] | None) -> tuple[npt.NDArray[Any], _aex.Plan]:
        return self._fetch(_to_wire(key, self.shape), quality)

    def _fetch_all(self) -> npt.NDArray[Any]:
        return self._fetch((Ellipsis,) if self._key is None else self._key, None)[0]

    def _fetch(
        self, wire_key: Key, quality: dict[str, Any] | None
    ) -> tuple[npt.NDArray[Any], _aex.Plan]:
        plan = self._native.prepare(self.handle, self.name, wire_key, quality)
        # Shape and dtype are the server's answer, not worked out here.
        out = np.empty(plan.shape, dtype=plan.dtype)
        self._native.fill(plan, self.handle, self.name, wire_key, out)
        out.flags.writeable = False
        return out, plan

    def at(
        self,
        *,
        dtype: npt.DTypeLike | None = None,
        abs_error: float | None = None,
        rel_error: float | None = None,
        codec: str | None = None,
    ) -> "QualityView":
        """A view that asks the server for a cheaper encoding of the data.

        Give one of: ``dtype`` to narrow the elements, or ``abs_error`` /
        ``rel_error`` for lossy compression. A server that cannot do it sends
        the exact data and the view warns; ``applied_quality`` says what was
        done. To take every n-th element, slice with a step instead: that is
        an ordinary selection and needs no quality at all.

        Only ``abs_error`` is implemented, on float32 and float64, and only by
        a server and an extension module built with the ``sz`` or ``zfp``
        feature. A bound relative to the value range would have to mean the
        range of the whole selection, and a compressed block only ever sees its
        own.

        ``codec`` picks which error-bounded codec carries it, ``"sz"`` or
        ``"zfp"``; left out, the server uses whichever it was built with.
        ``applied_quality["codec"]`` says which one it actually used.
        """
        self._require_base("at")
        quality: dict[str, Any] = {}
        if dtype is not None:
            quality["dtype"] = np.dtype(dtype).newbyteorder("<").str
        if abs_error is not None:
            quality["abs_error"] = float(abs_error)
        if rel_error is not None:
            quality["rel_error"] = float(rel_error)
        kinds = {"error" if k.endswith("_error") else k for k in quality}
        if len(kinds) != 1:
            raise ValueError("give exactly one of dtype or abs_error / rel_error")
        # The codec is how a quality travels, not which quality it is, so it
        # does not count towards the check above.
        if codec is not None:
            quality["codec"] = str(codec)
        return QualityView(self, quality)

    def read_into(self, out: npt.NDArray[Any], key: Any = Ellipsis) -> None:
        """Transfer a selection into ``out`` without allocating.

        ``out`` must be C-contiguous, writable, of the array's dtype and exactly
        the selection's size; nothing is copied to make it fit. If the transfer
        fails, the contents of ``out`` are undefined.
        """
        self._require_base("read_into")
        wire_key = _to_wire(key, self.shape)
        plan = self._native.prepare(self.handle, self.name, wire_key)
        self._native.fill(plan, self.handle, self.name, wire_key, out)

    def get_async(self, key: Any) -> "Future[npt.NDArray[Any]]":
        """Start transferring a selection and return at once.

        The transfer runs on the client's worker threads; ``result()`` gives
        what ``self[key]`` would have.
        """
        return self._client._submit(self.__getitem__, key)

    def gather(self, keys: Sequence[Any]) -> list[npt.NDArray[Any]]:
        """Transfer several selections, resolving them in one round trip.

        Raises the error of the first selection that fails. The results are
        read-only.
        """
        self._require_base("gather")
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
        data = self._fetch_all()
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
        name = _SERVER_FUNCTIONS.get(func)
        if name is not None:
            result = self._reduce(func, name, args, kwargs)
            if result is not _LOCAL:
                return result
        args, kwargs = _download(func, (args, kwargs))
        return func(*args, **kwargs)

    def _reduce(
        self, func: Callable[..., Any], name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Run a reduction on the server, or return _LOCAL if it cannot."""
        try:
            params = dict(inspect.signature(func).bind(*args, **kwargs).arguments)
        except TypeError:
            return _LOCAL
        if params.pop("a", None) is not self:
            return _LOCAL
        for default_only in ("dtype", "out"):
            if params.get(default_only, 0) is None:
                del params[default_only]
        if not params.keys() <= _SERVER_ARGUMENTS:
            return _LOCAL

        # numpy itself checks the arguments and names the result's dtype, on an
        # array of the same rank and type with one element.
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            probe = np.asarray(func(np.zeros((1,) * self.ndim, self.dtype), **params))
        axis = params.get("axis")
        if axis is None:
            reduced = set(range(self.ndim))
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            reduced = {operator.index(a) % self.ndim for a in axes}
        cells = math.prod(n for d, n in enumerate(self.shape) if d not in reduced)
        if cells * probe.dtype.itemsize > _INLINE_LIMIT:
            return _LOCAL

        wire: dict[str, Any] = {}
        if axis is not None:
            wire["axis"] = (
                tuple(operator.index(a) for a in axis)
                if isinstance(axis, tuple)
                else operator.index(axis)
            )
        if "keepdims" in params:
            wire["keepdims"] = bool(params["keepdims"])
        ddof = params.get("ddof", 0)
        if "ddof" in params:
            wire["ddof"] = int(ddof) if isinstance(ddof, (int, np.integer)) else float(ddof)

        key = () if self._key is None else self._key
        descr, shape, data = self._native.apply_function(self.handle, self.name, key, name, wire)
        out = np.frombuffer(data, dtype=descr).reshape(shape)
        _warn_as_numpy(name, out, self.size // cells if cells else 1, ddof)
        return out[()] if out.ndim == 0 else out.copy()

    def _require_base(self, what: str) -> None:
        if self._key is not None:
            raise TypeError(f"{what} is not supported on a view; use the array it came from")

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        inputs, kwargs = _download(ufunc, (inputs, kwargs))
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __repr__(self) -> str:
        kind = "ArrayProxy" if self._key is None else "ArrayProxy view"
        return f'<{kind} name "{self.name}", shape {self.shape}, type {self.dtype}>'


class _Viewer:
    """What ``ArrayProxy.view`` returns; indexing it makes a view."""

    def __init__(self, array: ArrayProxy) -> None:
        self._array = array

    def __getitem__(self, key: Any) -> ArrayProxy:
        array = self._array
        wire_key = _to_wire(key, array.shape)
        # The server resolves the selection, so it says what shape it has.
        plan = array._native.prepare(array.handle, array.name, wire_key)
        return ArrayProxy(
            array._client, array.handle, array.name, plan.dtype, tuple(plan.shape), wire_key
        )


def _warn_as_numpy(name: str, out: npt.NDArray[Any], count: int, ddof: Any) -> None:
    """Warn where numpy would have, which the server cannot."""
    message = None
    if name in ("mean", "var", "std") and count == 0:
        message = "Mean of empty slice."
    elif name in ("var", "std") and count - ddof <= 0:
        message = "Degrees of freedom <= 0 for slice"
    elif name in ("nanmax", "nanmin", "nanmean") and np.isnan(out).any():
        message = "All-NaN slice encountered"
    if message is not None:
        warnings.warn(message, RuntimeWarning, stacklevel=4)


class QualityView:
    """An array read at a requested quality. Made by ``ArrayProxy.at``."""

    def __init__(self, array: ArrayProxy, quality: dict[str, Any]) -> None:
        self.array = array
        self.quality = quality
        # What the server applied on the last read; None before one.
        self.applied_quality: dict[str, Any] | None = None

    def __getitem__(self, key: Any) -> npt.NDArray[Any]:
        """Transfer a selection at this view's quality. The result is read-only."""
        out, plan = self.array._read(key, self.quality)
        applied = plan.applied_quality
        if applied["encoding"] == "exact" and self.applied_quality is None:
            warnings.warn(
                f"the server cannot apply {self.quality}; the data is exact",
                AexQualityWarning,
                stacklevel=2,
            )
        self.applied_quality = applied
        return out

    def __repr__(self) -> str:
        return f"<QualityView of {self.array!r}, quality {self.quality}>"


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
    _check_fallback(f"np.{getattr(func, '__name__', func)}", nbytes, stacklevel=4)
    return _replace(tree)


def _check_fallback(what: str, nbytes: int, stacklevel: int = 3) -> None:
    """Refuse or warn about a download, as the policy says."""
    if _fallback_policy == "error":
        raise AexFallbackError(
            f"{what} is not supported server-side, and the fallback policy forbids "
            f"downloading {nbytes / 2**20:.1f} MiB to compute locally.",
            "REQUEST",
        )
    if _fallback_policy == "warn" and nbytes > _fallback_threshold:
        warnings.warn(
            f"{what} is not supported server-side.\n"
            f"  Downloading {nbytes / 2**20:.1f} MiB to compute locally.\n"
            "  Use arr[...] explicitly to silence, or set aex.set_fallback_policy(...).",
            AexFallbackWarning,
            stacklevel=stacklevel,
        )


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
        return tree._fetch_all()
    if isinstance(tree, (list, tuple)):
        return type(tree)(_replace(item) for item in tree)
    if isinstance(tree, dict):
        return {key: _replace(item) for key, item in tree.items()}
    return tree
