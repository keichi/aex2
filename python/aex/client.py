"""The client, and proxies for the files and groups it opens."""

from collections.abc import Iterator
from types import TracebackType
from typing import Self

from . import _aex
from .array_proxy import ArrayProxy
from .errors import AexNotFoundError, AexValueError

__all__ = ["ArrayProxy", "Client", "FileProxy", "GroupProxy"]


class Client:
    """A session with an AEX server.

    ``url`` is ``host:port``, optionally with a scheme. Transfer settings come
    from the ``AEX_*`` environment variables.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._native = _aex.Client(url if "://" in url else f"http://{url}")

    def open(self, path: str) -> "FileProxy":
        return FileProxy(self._native, self._native.open(path))

    def close(self) -> None:
        """End the session. The proxies it produced stop working."""
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
    native: _aex.Client, handle: int, name: str, item: tuple[str, list[int]] | None
) -> "ArrayProxy | GroupProxy":
    if item is None:
        return GroupProxy(native, handle, name)
    dtype, shape = item
    return ArrayProxy(native, handle, name, dtype, tuple(shape))


class GroupProxy:
    """A group in an open file. Indexing it with a path yields its items."""

    def __init__(self, native: _aex.Client, handle: int, name: str) -> None:
        self._native = native
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
        return _proxy(self._native, self.handle, path, self._native.get_item(self.handle, path))

    def __iter__(self) -> Iterator["ArrayProxy | GroupProxy"]:
        for name, item in self._native.list_children(self.handle, self.name):
            yield _proxy(self._native, self.handle, self._join_names(self.name, name), item)

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

    def __init__(self, native: _aex.Client, handle: int) -> None:
        super().__init__(native, handle, "/")

    def close(self) -> None:
        self._native.close_file(self.handle)

    def __repr__(self) -> str:
        return f'<FileProxy handle "{self.handle}">'
