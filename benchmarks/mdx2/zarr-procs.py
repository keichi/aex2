"""Is zarr-python's ceiling the machine, or one Python process?

Each worker process reads its own disjoint slice of the same store. If N
processes go N times faster, then neither the storage nor the codec was the
limit -- one interpreter was. Sweeping past the core count shows where the
machine takes over.

Usage: python zarr-procs.py STORE [ELEMENTS]
"""

import multiprocessing as mp
import statistics
import sys
import time

import zarr

MIB = 1 << 20


def work(store, k, procs, elements, done):
    array = zarr.open_array(f"{store}/array", mode="r")
    step = elements // procs
    done.put(array[k * step : (k + 1) * step].nbytes)


def run(store, procs, elements):
    done = mp.Queue()
    workers = [
        mp.Process(target=work, args=(store, k, procs, elements, done)) for k in range(procs)
    ]
    start = time.perf_counter()
    for worker in workers:
        worker.start()
    # Read the queue before joining: a full pipe would deadlock the children.
    delivered = sum(done.get() for _ in workers)
    for worker in workers:
        worker.join()
    return delivered / MIB / (time.perf_counter() - start)


def main() -> None:
    store = sys.argv[1]
    elements = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 30
    print(f"== {store}")
    for procs in [1, 2, 4, 8, 16]:
        runs = [run(store, procs, elements) for _ in range(3)]
        print(
            f"   zarr-python, {procs:2d} process(es)                 "
            f"{statistics.median(runs):8.0f} MiB/s "
            f"({min(runs):.0f}-{max(runs):.0f})"
        )


if __name__ == "__main__":
    main()
