"""Read the same array through readers that all cross the same link.

zarr-python against a store served over HTTP, h5py and h5pyd against the same
HDF5 file, and AEX2's Python client against the same data on the same machine,
in one loop with one timing, so that what is compared is the readers and not
several benchmarks.

Also the question this started as: is a reader's ceiling the machine, or one
Python process? Each worker process reads its own disjoint slice. If N
processes go N times faster, then neither the storage nor the codec was the
limit -- one interpreter was. Sweeping past the core count shows where the
machine takes over.

Usage:
  python read-procs.py /mnt/aexram/mem.zarr                  # local, as before
  python read-procs.py http://SERVER:8080/mem.zarr --via obstore
  python read-procs.py http://SERVER:50391/mem.zarr --backend aex --streams 16
  python read-procs.py http://SERVER:8080/mem.h5 --backend h5py --cache none
  python read-procs.py http://SERVER:5101/home/test/mem.h5 --backend h5pyd
"""

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import time

import numpy as np

MIB = 1 << 20
GIB = 1 << 30
# Slices the --check comparison reads, as fractions of the array.
CHECK_AT = (0.0, 0.5, 0.99)
CHECK_ELEMENTS = 1 << 18


def open_array(args, store=None, backend=None):
    """Open `store`'s `array` through the reader `backend` names.

    Returns the session too, which has to outlive the array it produced.
    """
    store = store or args.store
    backend = backend or args.backend
    if backend == "aex":
        # Read at connect time, so it has to be set before the client exists.
        os.environ["AEX_STREAMS"] = str(args.streams)
        import aex

        url, name = store.rsplit("/", 1)
        client = aex.Client(url)
        array = client.open(name)["array"]
        if args.abs_error or args.codec:
            array = array.at(abs_error=args.abs_error or None, codec=args.codec or None)
        return array, client

    if backend == "h5py":
        import fsspec
        import h5py

        # libhdf5 reads the file itself, over a file object that turns its
        # seeks into range requests. What that costs depends on how the object
        # caches, so the setting is swept rather than defaulted.
        handle = fsspec.open(store, block_size=args.block, cache_type=args.cache).open()
        f = h5py.File(handle, "r")
        return f["array"], f

    if backend == "h5pyd":
        import urllib.parse

        import h5pyd

        # The endpoint is the server, the path is the domain within it. The
        # rest comes from the environment, which h5pyd only reads from a
        # config file of its own.
        #
        # A standalone HSDS runs one service node, and everything a client
        # reads passes through that one Python process. Several of them are
        # started on consecutive ports to let it use the machine, and the
        # readers are spread over them.
        url = urllib.parse.urlsplit(store)
        host, _, port = url.netloc.partition(":")
        port = int(port) + args.rank % args.endpoints
        f = h5pyd.File(
            url.path,
            "r",
            endpoint=f"{url.scheme}://{host}:{port}",
            username=os.environ.get("HS_USERNAME"),
            password=os.environ.get("HS_PASSWORD"),
            bucket=os.environ.get("HS_BUCKET"),
        )
        return f["array"], f

    import zarr

    if args.concurrency:
        zarr.config.set({"async.concurrency": args.concurrency})
    if args.via == "obstore":
        import obstore
        from zarr.storage import ObjectStore

        url, name = store.rsplit("/", 1)
        # Without allow_http obstore refuses the plain-http URL, and its way of
        # saying so is ten retries of a request it never sent.
        remote = obstore.store.from_url(url, client_options={"allow_http": True})
        return zarr.open_array(store=ObjectStore(remote), path=f"{name}/array", mode="r"), None
    # A local path or an http URL; zarr-python picks the store from the scheme.
    return zarr.open_array(f"{store}/array", mode="r"), None


def read(array, start, stop, piece):
    """Read rows start..stop and return the bytes delivered.

    In `piece` rows at a time when that is set: HSDS assembles a whole
    selection in memory before it answers, so a quarter of a gigabyte asked
    for at once by each of sixteen readers gets it killed by the kernel.
    Reading in pieces is what its own documentation has a client do; the
    readers that stream a request do not need it and are not given it.

    The pieces are not joined, because joining them would charge the reader
    for a copy the others never make.
    """
    if not piece:
        return array[start:stop].nbytes
    return sum(array[at : min(at + piece, stop)].nbytes for at in range(start, stop, piece))


def work(args, k, procs, done):
    """Read process k's slice of the array and report what it cost."""
    session = None
    try:
        args.rank = k
        array, session = open_array(args)
        # The split is along the leading axis, which for a two-dimensional field
        # is rows rather than elements. A quality view wraps the array with the
        # shape rather than having one.
        rows = min(args.elements, getattr(array, "array", array).shape[0])
        step = rows // procs
        cpu = time.process_time()
        delivered = read(array, k * step, (k + 1) * step, args.piece)
        # The server drops a quality it cannot apply and sends exact data, so a
        # run that posts a good number may not have compressed anything.
        applied = getattr(array, "applied_quality", None)
        done.put((delivered, time.process_time() - cpu, applied))
    except BaseException as error:  # noqa: BLE001 - reported through the queue
        # The parent blocks on the queue, so a child that dies quietly hangs
        # the sweep rather than failing it.
        done.put(f"{type(error).__name__}: {error}")
    finally:
        # A session the server still counts is a session the next run cannot
        # open: it stops at 64 of them.
        if session is not None:
            session.close()


def run(args, procs):
    """One measurement: `procs` readers, each on its own slice, at once."""
    done = mp.Queue()
    workers = [mp.Process(target=work, args=(args, k, procs, done)) for k in range(procs)]
    start = time.perf_counter()
    for worker in workers:
        worker.start()
    # Read the queue before joining: a full pipe would deadlock the children.
    results = [done.get() for _ in workers]
    elapsed = time.perf_counter() - start
    for worker in workers:
        worker.join()
    for result in results:
        if isinstance(result, str):
            sys.exit(f"a reader failed: {result}")
    if args.abs_error or args.codec:
        applied = results[0][2] or {}
        wanted = (args.codec or "sz").lower()
        if applied.get("codec", "").lower() != wanted:
            sys.exit(f"asked for {wanted}, server applied {applied}")
    delivered = sum(nbytes for nbytes, _, _ in results)
    return delivered / MIB / elapsed, sum(cpu for _, cpu, _ in results) / (delivered / GIB)


def check(args, spec):
    """Require the two readers to deliver the same bytes, or stop."""
    backend, _, other = spec.partition(":")
    mine, mine_session = open_array(args)
    theirs, their_session = open_array(args, other, backend)
    for where in CHECK_AT:
        # A quality view has no shape of its own; it wraps the array that has.
        at = int(getattr(mine, "array", mine).shape[0] * where)
        got, want = mine[at : at + CHECK_ELEMENTS], theirs[at : at + CHECK_ELEMENTS]
        if not np.array_equal(got, want):
            sys.exit(f"{args.store} and {other} differ at element {at}")
    for session in (mine_session, their_session):
        if session is not None:
            session.close()
    print(f"   {args.store} and {other} agree at {len(CHECK_AT)} places")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", help="Store path, or URL whose last segment names the store")
    parser.add_argument("elements", nargs="?", type=int, default=1 << 30)
    parser.add_argument("--backend", choices=["zarr", "aex", "h5py", "h5pyd"], default="zarr")
    parser.add_argument(
        "--via", choices=["fsspec", "obstore"], default="fsspec", help="How zarr-python reaches it"
    )
    parser.add_argument(
        "--cache", default="none", help="h5py: how the fsspec file object caches (fsspec name)"
    )
    parser.add_argument(
        "--block", type=int, default=4 << 20, help="h5py: fsspec block size in bytes"
    )
    parser.add_argument(
        "--endpoints", type=int, default=1, help="h5pyd: HSDS servers, on consecutive ports"
    )
    parser.add_argument(
        "--piece", type=int, default=0, help="Rows per read; 0 asks for the whole slice at once"
    )
    parser.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--streams", type=int, default=1, help="AEX2 connections per process")
    parser.add_argument(
        "--concurrency", type=int, default=0, help="zarr-python async.concurrency; 0 keeps its own"
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--check", metavar="BACKEND:STORE", help="Require this reader to deliver the same bytes"
    )
    parser.add_argument("--label", default="", help="Prefix each result line, as aexbench does")
    parser.add_argument(
        "--abs-error", type=float, default=0.0, help="AEX2 error bound; 0 asks for exact data"
    )
    parser.add_argument("--codec", default="", help="AEX2 wire codec: sz, zfp or gzip")
    args = parser.parse_args()
    # Which worker this is; only the HSDS endpoints are chosen from it.
    args.rank = 0

    # The label has to say what would otherwise be invisible in a sweep log.
    if args.backend == "aex":
        quality = ""
        if args.abs_error or args.codec:
            quality = f" {args.codec or 'sz'}{args.abs_error or ''}"
        how = f"aex x{args.streams}{quality}"
    elif args.backend == "h5py":
        how = f"h5py {args.cache}/{args.block >> 20}M"
    elif args.backend == "h5pyd":
        how = f"h5pyd e{args.endpoints}"
    else:
        how = f"zarr/{args.via}" + (f" c{args.concurrency}" if args.concurrency else "")
    if not args.label:
        print(f"== {args.store}  {how}")
    if args.check:
        check(args, args.check)
        return
    for procs in args.procs:
        runs = [run(args, procs) for _ in range(args.reps)]
        rates = [rate for rate, _ in runs]
        cpu = statistics.median(cpu for _, cpu in runs)
        # A sweep's label already says which reader and how many of it.
        head = args.label if args.label else f"   {how:20} {procs:2d} process(es)"
        print(
            f"{head}  {statistics.median(rates):8.0f} MiB/s "
            f"({min(rates):.0f}-{max(rates):.0f})  cpu {cpu:.2f} s/GiB"
        )


if __name__ == "__main__":
    main()
