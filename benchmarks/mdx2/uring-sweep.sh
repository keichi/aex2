#!/bin/bash
#
# poolbench: is io_uring worth it? Send from the server VM to a sink on the
# client VM, alternating the conditions within each round, as the link's
# bandwidth moves during the day.
#
# Usage: benchmarks/mdx2/uring-sweep.sh [rounds]

set -euo pipefail

SERVER=${SERVER:-aex2-eval1}
CLIENT=${CLIENT:-aex2-eval2}
SINK=${SINK:-192.168.101.235:50399}
ROUNDS=${1:-5}
MEM=/mnt/aexram/mem.npy
DISK=${DISK:-/home/mdxuser/disk/disk.npy}
BYTES=$((4 << 30))

run() {  # run <file> <streams> <mode> [extra...]
    local file=$1 streams=$2 mode=$3
    shift 3
    ssh "$SERVER" "bash -lc 'cd aex2 && target/release/poolbench send $file \
        --to $SINK --bytes $BYTES --streams $streams --mode $mode $*'"
}

ssh "$CLIENT" 'pkill -x poolbench || true; mkdir -p ~/logs; cd aex2;
    setsid nohup target/release/poolbench sink > ~/logs/uring-sink.log 2>&1 < /dev/null &'
sleep 1
trap 'ssh "$CLIENT" "pkill -x poolbench || true"' EXIT

for _ in $(seq "$ROUNDS"); do
    for s in 1 4 16; do
        run $MEM "$s" serial
        run $MEM "$s" pair --buffers 3
        for d in 1 2 4 8; do
            run $MEM "$s" uring --depth "$d"
        done
        run $MEM "$s" uring-copy --depth 4
        run $MEM "$s" uring-zc --depth 4
    done
    for s in 1 4; do
        run $DISK "$s" serial --cold
        run $DISK "$s" pair --buffers 3 --cold
        run $DISK "$s" uring --depth 4 --cold
        run $DISK "$s" uring-zc --depth 4 --cold
    done
done
