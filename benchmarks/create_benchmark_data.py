#!/usr/bin/env python3
"""Generate the .npy files run_benchmarks.py reads.

.npy is the one format v1 and v2 both serve, so the same files measure both.
"""

import argparse
from pathlib import Path

import numpy as np

SIZES = {"10M": 10, "50M": 50, "100M": 100, "500M": 500, "1G": 1024, "2G": 2048}
DTYPES = ["float32", "float64", "int32", "int64"]


def shape_for(size_mib: int, dtype: np.dtype) -> tuple[int, int]:
    """A near-square 2-d shape of about ``size_mib`` MiB."""
    elements = size_mib * 1024 * 1024 // dtype.itemsize
    side = int(np.sqrt(elements))
    return side, elements // side


def create(path: Path, size_mib: int, dtype: np.dtype) -> None:
    rows, cols = shape_for(size_mib, dtype)
    print(f"{path.name}: shape {(rows, cols)}")
    # Written through a memmap so a large file never sits in memory at once.
    # Each element is its flat index, as in v1.
    out = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(rows, cols))
    step = max(1, (64 * 1024 * 1024) // (cols * dtype.itemsize))
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        out[start:stop] = np.arange(start * cols, stop * cols).reshape(-1, cols)
    out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="where to write the files (default: benchmarks/data)",
    )
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    parser.add_argument("--dtypes", nargs="+", default=DTYPES, choices=DTYPES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        for dtype in args.dtypes:
            path = args.output_dir / f"benchmark_{size}_{dtype}.npy"
            create(path, SIZES[size], np.dtype(dtype))


if __name__ == "__main__":
    main()
