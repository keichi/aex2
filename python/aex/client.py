"""The client, and proxies for the files and groups it opens."""

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, Self, TypeVar

from . import _aex
from .array_proxy import ArrayProxy
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
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    def open(self, path: str) -> "FileProxy":
        return FileProxy(self, self._native.open(path))

    def _submit(self, fn: Callable[..., T], *args: Any) -> "Future[T]":
        """Run ``fn`` on this client's worker threads.

        The calls release the GIL while they wait, so the workers transfer
        while the caller computes.
        """
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(thread_name_prefix="aex")
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
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown()
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
    client: Client, handle: int, name: str, item: tuple[str, list[int]] | None
) -> "ArrayProxy | GroupProxy":
    if item is None:
        return GroupProxy(client, handle, name)
    dtype, shape = item
    return ArrayProxy(client, handle, name, dtype, tuple(shape))


class GroupProxy:
    """A group in an open file. Indexing it with a path yields its items."""

    def __init__(self, client: Client, handle: int, name: str) -> None:
        self._client = client
        self._native = client._native
        self.handle = handle
        self.name = name

    @staticmethod
    def _join_names(*names: str) -> str:
        joined = ""
        for name in names:
            if name.startswith("/"):
                joined = name
            elif joined.endswith("/"):
                joined += name
            else:
                joined += "/" + name
        return joined

    def __getitem__(self, name: str) -> "ArrayProxy | GroupProxy":
        path = self._join_names(self.name, name)
        return _proxy(self._client, self.handle, path, self._native.get_item(self.handle, path))

    def __iter__(self) -> Iterator["ArrayProxy | GroupProxy"]:
        for name, item in self._native.list_children(self.handle, self.name):
            yield _proxy(self._client, self.handle, self._join_names(self.name, name), item)

    def __len__(self) -> int:
        return len(self._native.list_children(self.handle, self.name))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            self._native.get_item(self.handle, self._join_names(self.name, name))
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
        self._native.close_file(self.handle)

    def __repr__(self) -> str:
        return f'<FileProxy handle "{self.handle}">'
