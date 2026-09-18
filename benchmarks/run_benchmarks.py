#!/usr/bin/env python3
"""Time whole-array transfers, with v1 or v2 on the same files.

Ported from v1. The client API is the same in both, so this one script
measures either: by default the installed aex (v2), or v1 with ``--v1 PATH``
pointing at a v1 checkout. Run v1 in v1's own environment, since it needs grpcio:

    # v2
    aex-server --root benchmarks/data &
    python benchmarks/run_benchmarks.py --host localhost:50051

    # v1, same files
    (cd ../aex && uv run python -m aex.server) &
    uv run --project ../aex python benchmarks/run_benchmarks.py --v1 ../aex \\
        --data-dir "$PWD/benchmarks/data"

v2's tuning knobs are environment variables (AEX_STREAMS, AEX_CREDIT,
AEX_CHUNK_BYTES, ...), so a sweep needs no flags here; whichever are set are
printed with the results, since a number without its settings is not a record.
"""

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Result:
    file_name: str
    shape: tuple[int, ...]
    size_mib: float
    elapsed_sec: float

    @property
    def throughput_mibps(self) -> float:
        return self.size_mib / self.elapsed_sec


def run_once(client: Any, path: str, dataset: str) -> Result:
    f = client.open(path)
    array = f[dataset]
    start = time.perf_counter()
    data = array[:]
    elapsed = time.perf_counter() - start
    f.close()
    return Result(Path(path).name, tuple(array.shape), data.nbytes / 2**20, elapsed)


def report(results: list[Result]) -> None:
    first = results[0]
    print(f"\nFile: {first.file_name}")
    print(f"  Shape: {first.shape}, Size: {first.size_mib:.1f} MiB")
    for i, r in enumerate(results, 1):
        print(f"  Run {i}: {r.elapsed_sec:.3f} sec, {r.throughput_mibps:.1f} MiB/s")
    if len(results) > 1:
        print(
            f"  Average: {statistics.mean(r.elapsed_sec for r in results):.3f} sec, "
            f"{statistics.mean(r.throughput_mibps for r in results):.1f} MiB/s"
        )


def summary(results: list[Result]) -> None:
    throughputs = [r.throughput_mibps for r in results]
    total = sum(r.size_mib for r in results)
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files tested: {len({r.file_name for r in results})}")
    print(f"Total data transferred: {total:.1f} MiB ({total / 1024:.2f} GiB)")
    print(f"Average throughput: {statistics.mean(throughputs):.1f} MiB/s")
    print(f"Median throughput: {statistics.median(throughputs):.1f} MiB/s")
    print(f"Min throughput: {min(throughputs):.1f} MiB/s")
    print(f"Max throughput: {max(throughputs):.1f} MiB/s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="localhost:50051")
    parser.add_argument("--sizes", nargs="+", default=["10M", "50M", "100M"])
    parser.add_argument("--dtypes", nargs="+", default=["float32"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--data-dir",
        default="benchmarks/data",
        help="directory of the files, as the server sees it (default: benchmarks/data)",
    )
    parser.add_argument("--dataset", default="/array")
    parser.add_argument("--v1", type=Path, metavar="PATH", help="measure the v1 checkout at PATH")
    args = parser.parse_args()

    if args.v1 is not None:
        sys.path.insert(0, str(args.v1.resolve()))
    from aex.client import Client

    paths = sorted(
        f"{args.data_dir}/benchmark_{size.upper()}_{dtype}.npy"
        for size in args.sizes
        for dtype in args.dtypes
    )

    print("=" * 80)
    print(f"AEX Transfer Performance Benchmark ({'v1' if args.v1 else 'v2'})")
    print("=" * 80)
    print(f"Server: {args.host}")
    print(f"Files to test: {len(paths)}")
    print(f"Runs per file: {args.runs}")
    settings = {k: v for k, v in sorted(os.environ.items()) if k.startswith("AEX_")}
    print(f"Client settings: {settings or 'defaults'}")

    all_results: list[Result] = []
    client = Client(args.host)
    try:
        for path in paths:
            results = []
            for run in range(args.runs):
                try:
                    results.append(run_once(client, path, args.dataset))
                except Exception as e:
                    print(f"\n  {Path(path).name}, run {run + 1}: {e}")
                    if run == 0:
                        break
            if results:
                report(results)
                all_results.extend(results)
        if all_results:
            summary(all_results)
        else:
            print("\nNo successful benchmark runs completed.")
    finally:
        client.close()
    return 0 if all_results else 1


if __name__ == "__main__":
    sys.exit(main())
