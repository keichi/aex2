#!/bin/bash
#
# poolbench: is io_uring worth it? Send from the server VM to a sink on the
# client VM, alternating the conditions within each round, as the link's
# bandwidth moves during the day.
#
# Usage: benchmarks/mdx2/uring-sweep.sh [rounds] [mem|disk|all]

set -euo pipefail

SERVER=${SERVER:-aex2-eval1}
CLIENT=${CLIENT:-aex2-eval2}
SINK=${SINK:-192.168.101.235:50399}
ROUNDS=${1:-5}
WHICH=${2:-all}
MEM=/mnt/aexram/mem.npy
DISK=${DISK:-/home/mdxuser/disk/disk.npy}
BYTES=$((4 << 30))

run() {  # run <file> <streams> <mode> [extra...]
    local file=$1 streams=$2 mode=$3
    shift 3
    ssh "$SERVER" "bash -lc 'cd aex2 && target/release/poolbench send $file \
        --to $SINK --bytes $BYTES --streams $streams --mode $mode $*'"
}

# Memory-resident: what the send path costs when the reading is free.
mem_round() {
    for s in 1 4 8 16; do
        run $MEM "$s" serial
        run $MEM "$s" pair --buffers 3
        for d in 1 2 4 8; do
            run $MEM "$s" uring --depth "$d"
        done
        run $MEM "$s" uring-copy --depth 4
        run $MEM "$s" uring-zc --depth 4
    done
}

# Cold disk: the case the overlap is for. The depth is swept as far as the
# connection count, to see whether read-ahead can stand in for connections.
disk_round() {
    for s in 1 4 8 16; do
        run $DISK "$s" serial --cold
        run $DISK "$s" pair --buffers 3 --cold
        for d in 4 8 16; do
            run $DISK "$s" fadvise --depth "$d" --cold
            run $DISK "$s" uring --depth "$d" --cold
        done
        run $DISK "$s" uring-zc --depth 8 --cold
    done
}

ssh "$CLIENT" 'pkill -x poolbench || true; mkdir -p ~/logs; cd aex2;
    setsid nohup target/release/poolbench sink > ~/logs/uring-sink.log 2>&1 < /dev/null &'
sleep 1
trap 'ssh "$CLIENT" "pkill -x poolbench || true"' EXIT

for _ in $(seq "$ROUNDS"); do
    [ "$WHICH" = disk ] || mem_round
    [ "$WHICH" = mem ] || disk_round
done
