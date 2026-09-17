"""Write the HDF5 fixtures for the mdx2 measurements.

Each file holds one float32 dataset named ``array``, which is what aexbench
reads. ``counting`` matches mem.npy (element i is i), so aexbench can check
it; ``noisy`` compresses about as poorly as real data and is read with
``--pattern none``.

Usage: python mkh5.py OUTPUT ELEMENTS {contiguous,gzip} {counting,noisy}
"""

import sys

import h5py
import numpy as np

BATCH = 1 << 24
CHUNK = 1 << 20  # 4 MiB of float32, the server's default fetch size.


def main() -> None:
    path, elements, layout, content = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    options = {}
    if layout == "gzip":
        options = dict(chunks=(CHUNK,), compression="gzip", compression_opts=1, shuffle=True)
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("array", (elements,), dtype=np.float32, **options)
        for start in range(0, elements, BATCH):
            i = np.arange(start, min(start + BATCH, elements))
            if content == "counting":
                values = i.astype(np.float32)
            else:
                values = (i % 1000 + rng.integers(0, 16, i.size)).astype(np.float32)
            ds[start : start + i.size] = values


if __name__ == "__main__":
    main()
