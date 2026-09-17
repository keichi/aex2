# Types of the extension module built from crates/aex-py.

from typing import Any

import numpy as np

Item = tuple[str, list[int]] | None

class Plan:
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def total_bytes(self) -> int: ...
    @property
    def applied_quality(self) -> dict[str, Any]: ...
    @property
    def is_inline(self) -> bool: ...

class Client:
    def __init__(self, url: str) -> None: ...
    def open(self, path: str) -> int: ...
    def close_file(self, handle: int) -> None: ...
    def get_item(self, handle: int, name: str) -> Item: ...
    def list_children(self, handle: int, name: str) -> list[tuple[str, Item]]: ...
    def prepare(
        self,
        handle: int,
        name: str,
        key: tuple[Any, ...],
        quality: dict[str, Any] | None = None,
    ) -> Plan: ...
    def fill(
        self,
        plan: Plan,
        handle: int,
        name: str,
        key: tuple[Any, ...],
        out: np.ndarray[Any, Any],
    ) -> None: ...
    def apply_function(
        self,
        handle: int,
        name: str,
        key: tuple[Any, ...],
        function: str,
        kwargs: dict[str, Any],
    ) -> tuple[str, list[int], bytes]: ...
    def stats(self) -> dict[str, Any]: ...
    def prepare_many(
        self, handle: int, name: str, keys: list[tuple[Any, ...]]
    ) -> list[Plan | BaseException]: ...
    def fill_many(
        self,
        plans: list[Plan],
        handle: int,
        name: str,
        keys: list[tuple[Any, ...]],
        outs: list[np.ndarray[Any, Any]],
    ) -> None: ...
    def disconnect(self) -> None: ...
