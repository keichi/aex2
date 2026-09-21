"""Write the Zarr fixtures for the mdx2 measurements.

Each store holds one float32 array named ``array``, which is what aexbench
reads. ``counting`` matches mem.npy (element i is i), so aexbench can check
it; ``noisy`` compresses about as poorly as real data and is read with
``--pattern none``. ``wave`` is the field mknpy writes with ``--field wave``,
two-dimensional like wave.npy, and is the only one an error-bounded codec can
be measured on: a ramp compresses unboundedly and says nothing.

``plain`` is one file per chunk, which is what Zarr costs on a filesystem:
one open per chunk on the transfer's hot path. ``sharded`` holds the same
chunks in far fewer files, which is the comparison the layout exists for.

``big`` is the case the decode cache was designed for: a chunk far larger
than one fetch, so the pieces of it go to different connections and the
cache is what stops them decoding it again each.

An optional fifth argument quantizes the values to that many significant
digits before storing them. The store is then an ordinary lossless one that
happens to hold reduced-precision floats, which is what a user does who
accepts loss in storage rather than in transfer.

Usage: python mkzarr.py OUTPUT ELEMENTS {plain,sharded,big} {counting,noisy,wave} [DIGITS]
"""

import sys

import numcodecs
import numpy as np
import zarr

BATCH = 1 << 24
CHUNK = 1 << 20  # 4 MiB of float32, the server's default fetch size.
SHARD = CHUNK * 64  # 256 MiB a file, so one store is a handful of them.
BIG = CHUNK * 16  # 64 MiB, sixteen fetches to a chunk.
ROW = 2048  # Elements in a row of the wave field, as wave.npy has.


def wave(i):
    """The field `mknpy --field wave` writes, computed in float32 as it is."""
    x, y = (i % ROW).astype(np.float32), (i // ROW).astype(np.float32)
    smooth = np.sin(x / 37) * 100 + np.cos(y / 53) * 40 + np.sin((x + y) / 211) * 10
    wiggle = i.astype(np.uint64) * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
    return (smooth + ((wiggle >> np.uint64(40)).astype(np.float32) / (1 << 24) - 0.5)).astype(
        np.float32
    )


def main() -> None:
    path, elements, layout, content = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    digits = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    quantize = numcodecs.Quantize(digits, "float32") if digits else None
    shape = {"chunks": (CHUNK,)}
    if layout == "sharded":
        shape = {"chunks": (CHUNK,), "shards": (SHARD,)}
    elif layout == "big":
        shape = {"chunks": (BIG,)}
    rng = np.random.default_rng(0)
    # The array is a child named ``array``, not the store itself: that is the
    # name aexbench asks for, as it does for the HDF5 fixtures.
    root = zarr.create_group(path, overwrite=True)
    if content == "wave":
        # Two dimensions, like wave.npy: an error-bounded codec predicts from
        # neighbours in both, and a single long row is not the same job.
        shape = {name: (size // ROW, ROW) for name, (size,) in shape.items()}
        array = root.create_array("array", shape=(elements // ROW, ROW), dtype="float32", **shape)
    else:
        array = root.create_array("array", shape=(elements,), dtype="float32", **shape)
    for start in range(0, elements, BATCH):
        i = np.arange(start, min(start + BATCH, elements))
        if content == "counting":
            values = i.astype(np.float32)
        elif content == "wave":
            values = wave(i)
        else:
            values = (i % 1000 + rng.integers(0, 16, i.size)).astype(np.float32)
        if quantize:
            values = quantize.encode(values)
        if content == "wave":
            array[start // ROW : (start + i.size) // ROW] = values.reshape(-1, ROW)
        else:
            array[start : start + i.size] = values


if __name__ == "__main__":
    main()
