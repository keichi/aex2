"""Write the Zarr fixtures for the mdx2 measurements.

Each store holds one float32 array named ``array``, which is what aexbench
reads. ``counting`` matches mem.npy (element i is i), so aexbench can check
it; ``noisy`` compresses about as poorly as real data and is read with
``--pattern none``.

``plain`` is one file per chunk, which is what Zarr costs on a filesystem:
one open per chunk on the transfer's hot path. ``sharded`` holds the same
chunks in far fewer files, which is the comparison the layout exists for.

Usage: python mkzarr.py OUTPUT ELEMENTS {plain,sharded} {counting,noisy}
"""

import sys

import numpy as np
import zarr

BATCH = 1 << 24
CHUNK = 1 << 20  # 4 MiB of float32, the server's default fetch size.
SHARD = CHUNK * 64  # 256 MiB a file, so one store is a handful of them.


def main() -> None:
    path, elements, layout, content = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    shape = {"chunks": (CHUNK,)}
    if layout == "sharded":
        shape = {"chunks": (CHUNK,), "shards": (SHARD,)}
    rng = np.random.default_rng(0)
    # The array is a child named ``array``, not the store itself: that is the
    # name aexbench asks for, as it does for the HDF5 fixtures.
    root = zarr.create_group(path, overwrite=True)
    array = root.create_array("array", shape=(elements,), dtype="float32", **shape)
    for start in range(0, elements, BATCH):
        i = np.arange(start, min(start + BATCH, elements))
        if content == "counting":
            values = i.astype(np.float32)
        else:
            values = (i % 1000 + rng.integers(0, 16, i.size)).astype(np.float32)
        array[start : start + i.size] = values


if __name__ == "__main__":
    main()
