"""Serve an HDF5 file through HSDS without copying its data.

HSDS can keep the chunks where they are and store only a table of offsets into
the original file, which is how it should be compared against a server that
reads that same file. `hsload --link` writes that table, but this h5pyd cannot:
it sends a maxshape that HSDS refuses for a reference layout, and for a
contiguous dataset it writes a layout with no chunk shape, which makes HSDS
answer every read by fetching the whole dataset. The table is a few lines, so
it is written here instead.

The file has to sit in the bucket directory HSDS was started on, and the
fixtures are hard links into it so that both servers read the same bytes.

Usage: python hslink.py FILE DOMAIN [CHUNK_ELEMENTS]
       CHUNK_ELEMENTS sets the chunk shape for a contiguous dataset.
"""

import os
import sys

import h5py
import h5pyd
import numpy as np
from h5pyd._apps.utillib import get_chunktable_dtype

BUCKET = os.environ.get("HS_BUCKET", "hsds")
ROOT = os.environ.get("HS_ROOT", "/mnt/aexram")


def chunk_table(dset, chunk):
    """Where each chunk of `dset` lives in the file, and what it costs."""
    count = -(-dset.shape[0] // chunk)
    table = np.zeros(count, dtype=get_chunktable_dtype())
    if dset.chunks is None:
        # Contiguous: the chunks are a slice of the one region, in order.
        stride = chunk * dset.dtype.itemsize
        table["offset"] = dset.id.get_offset() + np.arange(count, dtype=np.int64) * stride
        table["size"] = stride
        return table
    space = dset.id.get_space()
    for i in range(dset.id.get_num_chunks(space)):
        info = dset.id.get_chunk_info(i, space)
        table[info.chunk_offset[0] // chunk] = (info.byte_offset, info.size)
    return table


def main() -> None:
    name, domain = sys.argv[1], sys.argv[2]
    with h5py.File(f"{ROOT}/{name}", "r") as f:
        dset = f["array"]
        if len(dset.shape) != 1:
            sys.exit("only one-dimensional datasets are linked here")
        chunk = dset.chunks[0] if dset.chunks else int(sys.argv[3])
        filters = {
            "compression": dset.compression,
            "compression_opts": dset.compression_opts,
            "shuffle": dset.shuffle,
        }
        table = chunk_table(dset, chunk)
        shape, dtype = dset.shape, dset.dtype

    with h5pyd.File(domain, "w", bucket=BUCKET) as out:
        anon = out.create_dataset(None, table.shape, dtype=table.dtype)
        anon[...] = table
        layout = {
            "class": "H5D_CHUNKED_REF_INDIRECT",
            "file_uri": f"{BUCKET}/{name}",
            "dims": [chunk],
            "chunk_table": anon.id.id,
        }
        out.create_dataset("array", shape=shape, dtype=dtype, chunks=layout, **filters)
    print(f"{name} -> {domain}: {len(table)} chunks of {chunk} elements, {filters}")


if __name__ == "__main__":
    main()
