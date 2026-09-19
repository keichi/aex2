#!/bin/bash
#
# poolbench: one connection, which is the case a client cannot widen.
#
# `pair` is the only thing that wins there today, because it puts the read and
# the send on two cores. This asks whether read-ahead on one thread matches it,
# and what the two together do.
#
# Usage: benchmarks/mdx2/one-conn-sweep.sh [rounds]

set -euo pipefail

SERVER=${SERVER:-aex2-eval1}
CLIENT=${CLIENT:-aex2-eval2}
SINK=${SINK:-192.168.101.235:50399}
ROUNDS=${1:-7}
MEM=/mnt/aexram/mem.npy
DISK=${DISK:-/home/mdxuser/disk/disk.npy}
BYTES=$((4 << 30))
STRIDED="--frag 16384 --gap 49152"

run() {  # run <file> <mode> [extra...]
    local file=$1 mode=$2
    shift 2
    ssh "$SERVER" "bash -lc 'cd aex2 && target/release/poolbench send $file \
        --to $SINK --bytes $BYTES --streams 1 --mode $mode --buffers 3 $*'"
}

ssh "$CLIENT" 'pkill -x poolbench || true; mkdir -p ~/logs; cd aex2;
    setsid nohup target/release/poolbench sink > ~/logs/one-conn-sink.log 2>&1 < /dev/null &'
sleep 1
trap 'ssh "$CLIENT" "pkill -x poolbench || true"' EXIT

for _ in $(seq "$ROUNDS"); do
    for m in serial pair fadvise pair-fadvise; do
        run $MEM "$m" --depth 8
        run $MEM "$m" --depth 8 $STRIDED
        run $DISK "$m" --depth 8 --cold
        run $DISK "$m" --depth 8 $STRIDED --cold
    done
    # How deep read-ahead has to go on the side where it pays.
    for d in 4 16 32; do
        run $DISK fadvise --depth "$d" --cold
        run $DISK fadvise --depth "$d" $STRIDED --cold
    done
done
