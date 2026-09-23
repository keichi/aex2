"""The client, and proxies for the files and groups it opens."""

import posixpath
from collections.abc import Callable, ItemsView, Iterator, Mapping, ValuesView
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, Self, TypeVar

from . import _aex
from .array_proxy import ArrayProxy, _attrs_from_wire
from .errors import AexNotFoundError, AexValueError

__all__ = ["ArrayProxy", "Client", "FileProxy", "GroupProxy"]

T = TypeVar("T")


class Client:
    """A session with an AEX server.

    ``url`` is ``host:port``, optionally with a scheme. Transfer settings come
    from the ``AEX_*`` environment variables.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._native = _aex.Client(url if "://" in url else f"http://{url}")
        # Starts no thread until the first get_async.
        self._executor = ThreadPoolExecutor(thread_name_prefix="aex")

    def open(self, path: str) -> "FileProxy":
        """Open a file, at a path relative to the server's data root."""
        return FileProxy(self, self._native.open(path))

    def _submit(self, fn: Callable[..., T], *args: Any) -> "Future[T]":
        """Run ``fn`` on this client's worker threads.

        The calls release the GIL while they wait, so the workers transfer
        while the caller computes.
        """
        return self._executor.submit(fn, *args)

    def stats(self) -> dict[str, Any]:
        """Totals over every transfer this client has made.

        ``rtt_ms`` is the fastest control plane call so far, an upper bound on
        the round trip.
        """
        return self._native.stats()

    def close(self) -> None:
        """End the session. The proxies it produced stop working.

        Waits for pending get_async transfers first.
        """
        self._executor.shutdown()
        self._native.disconnect()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<Client url "{self.url}">'


def _proxy(
    client: Client,
    handle: int,
    name: str,
    item: tuple[str, list[int]] | None,
    attrs: dict[str, Any],
) -> "ArrayProxy | GroupProxy":
    decoded = _attrs_from_wire(attrs)
    if item is None:
        return GroupProxy(client, handle, name, decoded)
    dtype, shape = item
    return ArrayProxy(client, handle, name, dtype, tuple(shape), None, decoded)


class GroupProxy(Mapping[str, "ArrayProxy | GroupProxy"]):
    """A group in an open file, as a read-only mapping from name to item.

    ``group[name]`` is the item at that path: an ``ArrayProxy`` for a dataset,
    a ``GroupProxy`` for a group. A leading ``/`` makes the path absolute, and
    ``in`` tests for a path without raising. Iterating and ``keys()`` yield the
    child names like ``h5py``; ``values()`` and ``items()`` give the proxies,
    and cost one request for the whole group rather than one per child.
    """

    def __init__(
        self, client: Client, handle: int, name: str, attrs: dict[str, Any] | None = None
    ) -> None:
        self._client = client
        self._native = client._native
        self.handle = handle
        self.name = name
        # Fetched on first use when whoever built this proxy had none.
        self._attrs = attrs

    @property
    def attrs(self) -> dict[str, Any]:
        """The group's HDF5/netCDF attributes.

        The root group carries the netCDF global attributes. Empty for a format
        without any.
        """
        if self._attrs is None:
            self._attrs = _attrs_from_wire(self._native.get_item(self.handle, self.name)[1])
        return self._attrs

    def __getitem__(self, name: str) -> "ArrayProxy | GroupProxy":
        path = posixpath.join(self.name, name)
        return _proxy(self._client, self.handle, path, *self._native.get_item(self.handle, path))

    def _children(self) -> dict[str, "ArrayProxy | GroupProxy"]:
        """Every child, built from one listing: it already carries the metadata."""
        return {
            name: _proxy(self._client, self.handle, posixpath.join(self.name, name), item, attrs)
            for name, item, attrs in self._native.list_children(self.handle, self.name)
        }

    def values(self) -> ValuesView["ArrayProxy | GroupProxy"]:
        """The child proxies, as of now."""
        return self._children().values()

    def items(self) -> ItemsView[str, "ArrayProxy | GroupProxy"]:
        """The child names and proxies, as of now."""
        return self._children().items()

    def __iter__(self) -> Iterator[str]:
        for name, _, _ in self._native.list_children(self.handle, self.name):
            yield name

    def __len__(self) -> int:
        return len(self._native.list_children(self.handle, self.name))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            self._native.get_item(self.handle, posixpath.join(self.name, name))
        except (AexNotFoundError, AexValueError):
            return False
        return True

    def __repr__(self) -> str:
        return f'<GroupProxy name "{self.name}">'


class FileProxy(GroupProxy):
    """An open file, which is also its root group."""

    def __init__(self, client: Client, handle: int) -> None:
        super().__init__(client, handle, "/")

    def close(self) -> None:
        """Release the file on the server. The proxies it produced stop working."""
        self._native.close_file(self.handle)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<FileProxy handle "{self.handle}">'
