"""Fixtures: a real aex-server on loopback, and files for it to serve.

The server is the binary from this workspace, built on first use, so the tests
exercise the same control plane and data plane a user would.
"""

import os
import re
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import aex

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def server(data_dir: Path) -> Iterator[str]:
    """Start aex-server and yield its control plane address."""
    binary = os.environ.get("AEX_SERVER_BIN")
    if binary is None:
        subprocess.run(
            ["cargo", "build", "-q", "-p", "aex-server", "--features", "hdf5"], cwd=REPO, check=True
        )
        binary = str(REPO / "target" / "debug" / "aex-server")

    proc = subprocess.Popen(
        [
            binary,
            "--control-addr",
            "127.0.0.1:0",
            "--data-addr",
            "127.0.0.1:0",
            "--root",
            str(data_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "NO_COLOR": "1"},
    )
    assert proc.stdout is not None
    # Port 0 lets the OS pick; the server logs what it got.
    address = None
    for line in proc.stdout:
        match = re.search(r"control=(\S+)", line)
        if match:
            address = match.group(1)
            break
    if address is None:
        proc.kill()
        pytest.fail("aex-server exited before it was listening")
    # Keep reading, or the server blocks on a full pipe once it has logged enough.
    threading.Thread(target=proc.stdout.read, daemon=True).start()
    try:
        yield address
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def client(server: str) -> Iterator[aex.Client]:
    with aex.Client(server) as client:
        yield client


@pytest.fixture(scope="session")
def ds_paths(data_dir: Path) -> dict[str, str]:
    """The datasets of v1's test file, one .npy each.

    ds1 is ``col + row * 200``, so every element says where it came from.
    """
    y, x = np.meshgrid(np.arange(100), np.arange(200), indexing="ij")
    arrays = {
        "ds1": (x + y * 200).astype(np.float32),
        "ds2": np.zeros((100, 200), np.float64),
        "ds3": np.ones((100, 200), np.int8),
        "ds4": np.zeros((100, 200), np.int16),
        "ds5": np.zeros((100, 200), np.int32),
        "ds6": (x % 2).astype(np.int64),
    }
    paths = {}
    for name, array in arrays.items():
        path = data_dir / f"{name}.npy"
        np.save(path, array)
        paths[name] = str(path)
    return paths


@pytest.fixture
def file_proxy(client: aex.Client, ds_paths: dict[str, str]) -> aex.FileProxy:
    return client.open(ds_paths["ds1"])


@pytest.fixture
def array_proxy(file_proxy: aex.FileProxy) -> aex.ArrayProxy:
    proxy = file_proxy["array"]
    assert isinstance(proxy, aex.ArrayProxy)
    return proxy


@pytest.fixture
def open_array(client: aex.Client, ds_paths: dict[str, str]):  # type: ignore[no-untyped-def]
    """Open one of the datasets by name."""

    def open_array(name: str) -> aex.ArrayProxy:
        proxy = client.open(ds_paths[name])["array"]
        assert isinstance(proxy, aex.ArrayProxy)
        return proxy

    return open_array
