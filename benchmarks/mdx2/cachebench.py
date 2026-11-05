"""Time reads the client cache answers against the same reads transferred.

Three clients read the same requests, alternating within each rep because the
shared link drifts:

- hit:      keeps results, and has the region every request falls in
- transfer: keeps nothing
- probe:    a cache of one byte, so it resolves and scans every request but
            stores nothing; probe minus transfer is what a miss costs

Usage: cachebench.py URL PATH [--reps N] [--region-rows R] [--label L]
PATH is a 2-D array with at least R rows and 2048 columns (wave.npy).
"""

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import aex


def requests(rows: int, cols: int, rng: np.random.Generator) -> dict[str, Callable[[int], Any]]:
    """Request kinds inside rows [0, rows), each a function of the rep."""
    return {
        # One row's first 4 KiB, the smallest read worth caching.
        "row-4KiB": lambda i: np.s_[(i * 37) % rows, 0:1024],
        # A 512-row tile: 4 MiB.
        "tile-4MiB": lambda i: np.s_[_tile(i, rows) : _tile(i, rows) + 512, :],
        # Every 8th row and column: 1 MiB at 8192 rows.
        "strided-1MiB": lambda i: np.s_[i % 8 : rows : 8, i % 8 :: 8],
        # 1024 rows picked at random: 8 MiB.
        "rows-8MiB": lambda i: np.sort(rng.choice(rows, 1024, replace=False)),
        # 100,000 scattered elements, an index array per axis.
        "points-100k": lambda i: (rng.integers(0, rows, 100_000), rng.integers(0, cols, 100_000)),
    }


def _tile(i: int, rows: int) -> int:
    return (i * 512) % (rows - 512)


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    out = fn()
    return time.perf_counter() - start, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("path")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--region-rows", type=int, default=8192)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    clients = {
        "hit": aex.Client(args.url, cache_bytes=1 << 30),
        "transfer": aex.Client(args.url),
        "probe": aex.Client(args.url, cache_bytes=1),
    }
    arrays = {name: client.open(args.path)["array"] for name, client in clients.items()}
    rows = args.region_rows
    cols = arrays["hit"].shape[1]

    fill, region = timed(lambda: arrays["hit"][0:rows, :])
    print(f"{args.label} fill {region.nbytes / 2**20:.0f}MiB {fill * 1e3:.1f}ms", flush=True)
    sent = clients["hit"].stats()["bytes"]

    rng = np.random.default_rng(0)
    kinds = requests(rows, cols, rng)
    times: dict[tuple[str, str], list[float]] = {}
    sizes: dict[str, int] = {}
    for rep in range(args.reps):
        for kind, make in kinds.items():
            key = make(rep)
            results = {}
            for name, array in arrays.items():
                elapsed, results[name] = timed(lambda: array[key])  # noqa: B023
                times.setdefault((kind, name), []).append(elapsed)
            sizes[kind] = results["hit"].nbytes
            if rep == 0:
                for name in ("transfer", "probe"):
                    np.testing.assert_array_equal(results["hit"], results[name])

    # Every request was inside the region, so the cached client sent nothing more.
    assert clients["hit"].stats()["bytes"] == sent
    assert clients["hit"].stats()["cache"]["misses"] == 1
    assert clients["probe"].stats()["cache"]["entries"] == 0

    for (kind, name), samples in times.items():
        ms = sorted(t * 1e3 for t in samples)
        print(
            f"{args.label} {kind} {name} median {statistics.median(ms):.4f}ms "
            f"min {ms[0]:.4f}ms max {ms[-1]:.4f}ms n {len(ms)} "
            f"size {sizes[kind] / 2**20:.3f}MiB",
            flush=True,
        )
    for client in clients.values():
        client.close()


if __name__ == "__main__":
    main()
