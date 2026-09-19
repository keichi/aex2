#!/bin/bash
#
# Sweep one tuning knob at a time to settle what the shipping defaults should be.
#
# A full grid of streams x chunk x credit x round trip is a day of link time for
# a question each axis answers on its own: the M4 tables already showed the axes
# barely interact once credit is deep enough to keep a connection busy. So each
# knob is swept with the others left at what the code ships, at three round
# trips, because the answer at 0 ms is not the answer at 50 ms.
#
# Socket buffer ceilings are raised throughout. With the kernel defaults every
# point above about 5 ms reads the same capped number and the sweep measures the
# sysctl instead of the knob.
#
# Each pass visits every point once and the passes are repeated, because the
# shared link drifts over the day; take the median across passes.
#
# Usage: benchmarks/mdx2/param-sweep.sh [reps]
#   The server must be up: aex.toml (50191).

set -euo pipefail

REPS=${1:-5}
RTTS=(0 10 50)
. "$(dirname "$0")/netem.sh"

# What the code ships today, and so the point every axis crosses.
BASE_STREAMS=4 BASE_CHUNK=4194304 BASE_CREDIT=4 BASE_RCVBUF=0
STREAMS=(1 2 4 8 16)
CHUNKS=(262144 1048576 4194304 16777216)
CREDITS=(1 2 4 8 16 64)
RCVBUFS=(0 4194304 16777216 67108864 268435456)

BYTES=$((4 << 30))
PORT_DEFAULT=50191    # aex.toml

restore() {
    clear_rtt
    set_buffers default
}
trap restore EXIT

# run <port> <label> <aexbench flags...>
run() {
    local port=$1 label=$2
    shift 2
    flush_metrics
    ssh "$CLIENT" "bash -lc 'cd aex2 && target/release/aexbench \
        http://$SERVER_IP:$port mem.npy --bytes $BYTES --reps 1 --prefault \
        $* --label \"$label\"'"
}

set_buffers tuned
for r in $(seq "$REPS"); do
    for rtt in "${RTTS[@]}"; do
        set_rtt "$rtt"
        p="rtt+$rtt"
        for s in "${STREAMS[@]}"; do
            run $PORT_DEFAULT "$p streams" --streams "$s" --chunk $BASE_CHUNK --credit $BASE_CREDIT
        done
        for c in "${CHUNKS[@]}"; do
            run $PORT_DEFAULT "$p chunk" --streams $BASE_STREAMS --chunk "$c" --credit $BASE_CREDIT
        done
        for k in "${CREDITS[@]}"; do
            run $PORT_DEFAULT "$p credit" --streams $BASE_STREAMS --chunk $BASE_CHUNK --credit "$k"
        done
        for b in "${RCVBUFS[@]}"; do
            run $PORT_DEFAULT "$p rcvbuf" --streams $BASE_STREAMS --chunk $BASE_CHUNK \
                --credit $BASE_CREDIT --rcvbuf "$b"
        done    done
done
