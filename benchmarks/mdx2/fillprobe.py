"""Split a read into its control plane and data plane round trips.

Each request is read after a one-second idle gap, then again at once, so that
what the kernel does to an idle connection shows in the first fill and not the
second.

Usage: fillprobe.py URL PATH LABEL     (PATH is wave.npy or any 2-D array of
at least 8192 x 2048)
"""

import sys
import time

import numpy as np

import aex
from aex.array_proxy import _to_wire

url, path, label = sys.argv[1:4]
client = aex.Client(url)
arr = client.open(path)["array"]
native = arr._native
keys = {
    "strided-1MiB": np.s_[0:8192:8, ::8],
    "tile-4MiB": np.s_[0:512, :],
    "rows-8MiB": np.sort(np.random.default_rng(0).choice(8192, 1024, replace=False)),
}
arr[0:64]


def read(key: object) -> tuple[float, float]:
    """Milliseconds spent in prepare and in fill."""
    wire = _to_wire(key, arr.shape)
    t0 = time.perf_counter()
    plan = native.prepare(arr.handle, arr.name, wire, None)
    t1 = time.perf_counter()
    native.fill(plan, arr.handle, arr.name, wire, np.empty(plan.shape, dtype=plan.dtype))
    return (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3


for _ in range(5):
    for kind, key in keys.items():
        time.sleep(1.0)
        idle = read(key)
        warm = read(key)
        print(
            f"{label} {kind} idle prepare {idle[0]:.0f} fill {idle[1]:.0f} | "
            f"warm prepare {warm[0]:.0f} fill {warm[1]:.0f}",
            flush=True,
        )
client.close()
