#!/bin/bash
#
# poolbench: is read-ahead worth it against the best the server can do today?
#
# `fadvise` is one thread, one buffer, and the next pieces hinted to the kernel,
# which is what the server does now. It is compared with what it did before
# (`serial`, and `pair` for the reader thread that used to be an option) and
# with the two together, over memory-resident and cold data, contiguous and
# strided.
#
# The two buy different things — a second core, and read depth — so compare at
# equal thread counts too: `pair-fadvise` at 4 connections is 8 threads, and so
# is `fadvise` at 8.
#
# Usage: benchmarks/mdx2/fadvise-sweep.sh [rounds]

set -euo pipefail

SERVER=${SERVER:-aex2-eval1}
CLIENT=${CLIENT:-aex2-eval2}
SINK=${SINK:-192.168.101.235:50399}
ROUNDS=${1:-5}
MEM=/mnt/aexram/mem.npy
DISK=${DISK:-/home/mdxuser/disk/disk.npy}
BYTES=$((4 << 30))
# A quarter-dense selection whose fragments are too far apart to be joined,
# which is what reaches the reads after `selection.rs` has coalesced.
STRIDED="--frag 16384 --gap 49152"

run() {  # run <file> <streams> <mode> [extra...]
    local file=$1 streams=$2 mode=$3
    shift 3
    ssh "$SERVER" "bash -lc 'cd aex2 && target/release/poolbench send $file \
        --to $SINK --bytes $BYTES --streams $streams --mode $mode --depth 8 $*'"
}

ssh "$CLIENT" 'pkill -x poolbench || true; mkdir -p ~/logs; cd aex2;
    setsid nohup target/release/poolbench sink > ~/logs/fadvise-sink.log 2>&1 < /dev/null &'
sleep 1
trap 'ssh "$CLIENT" "pkill -x poolbench || true"' EXIT

for _ in $(seq "$ROUNDS"); do
    for s in 1 4 8 16; do
        for m in serial pair fadvise pair-fadvise; do
            run $MEM "$s" "$m"
            run $MEM "$s" "$m" $STRIDED
            run $DISK "$s" "$m" --cold
            run $DISK "$s" "$m" $STRIDED --cold
        done
    done
done
