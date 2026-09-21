"""Where zarr-python's time goes, on the machine that holds the store.

Three questions, in order: what one core of zstd can decompress (the floor
every reader shares), whether zarr-python uses more than one core when asked
to, and how much of its time is the framework rather than read + decode.

Usage: python zarr-why.py STORE [ELEMENTS]
"""

import functools
import pathlib
import statistics
import sys
import threading
import time

import numcodecs
import numpy as np
import zarr

MIB = 1 << 20


def timed(fn, delivered, reps=3):
    """Median MiB/s of `reps` runs, each delivering `delivered` bytes."""
    runs = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        runs.append(delivered / MIB / (time.perf_counter() - start))
    return statistics.median(runs), min(runs), max(runs)


def codec_ceiling(store, codec):
    """MiB/s one core decompresses at, with nothing else in the way."""
    files = sorted(pathlib.Path(store, "array", "c").rglob("*"))
    raw = files[len(files) // 2].read_bytes()
    decoded = len(codec.decode(raw))
    # Enough repeats to run for a moment rather than a few microseconds.
    reps = max(1, 512 * MIB // decoded)
    start = time.perf_counter()
    for _ in range(reps):
        codec.decode(raw)
    return reps * decoded / MIB / (time.perf_counter() - start), files


def read_in_threads(array, out, threads):
    """Read the whole array with `threads` callers taking disjoint slices."""
    step = len(out) // threads

    def work(k, out=out, step=step):
        out[k * step : (k + 1) * step] = array[k * step : (k + 1) * step]

    workers = [threading.Thread(target=work, args=(k,)) for k in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def read_without_zarr(files, codec, out):
    """Read the chunk files and decode them, with zarr-python out of the way."""
    at = 0
    for path in files:
        chunk = np.frombuffer(codec.decode(path.read_bytes()), out.dtype)
        take = min(len(chunk), len(out) - at)
        out[at : at + take] = chunk[:take]
        at += take


def main() -> None:
    store = sys.argv[1]
    elements = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 30
    array = zarr.open_array(f"{store}/array", mode="r")
    codec = numcodecs.Zstd()
    delivered = elements * array.dtype.itemsize
    print(f"== {store}  {array.nchunks} chunks of {array.chunks[0]}  {array.compressors}")

    ceiling, files = codec_ceiling(store, codec)
    print(f"   one core, numcodecs zstd decode only        {ceiling:8.0f} MiB/s")

    for threads in [1, 2, 4, 8, 16]:
        out = np.empty(elements, array.dtype)
        cpu = time.process_time()
        read = functools.partial(read_in_threads, array, out, threads)
        median, low, high = timed(read, delivered)
        # CPU seconds per GiB delivered, over the three runs `timed` made.
        per_gib = (time.process_time() - cpu) / 3 / (delivered / (1 << 30))
        print(
            f"   zarr-python, {threads:2d} thread(s)                  "
            f"{median:8.0f} MiB/s ({low:.0f}-{high:.0f})  cpu {per_gib:.2f} s/GiB"
        )
        del out

    out = np.empty(elements, array.dtype)
    read = functools.partial(read_without_zarr, files, codec, out)
    median, low, high = timed(read, delivered)
    print(
        f"   read + decode, one thread, no zarr-python   {median:8.0f} MiB/s ({low:.0f}-{high:.0f})"
    )


if __name__ == "__main__":
    main()
