#!/bin/bash
#
# Where SZ3 starts to pay, across RTT, capped bandwidth and connections (=
# compressing cores); see docs/benchmark-sz.md. The fixture is wave.npy: a counting
# ramp compresses unboundedly. Passes repeat; take the median.
#
# Usage: benchmarks/mdx2/sz-sweep.sh [reps]
#   The server must be up with the sz feature: aex.toml (50191).
#   The fixture must exist: mknpy /mnt/aexram/wave.npy 268435456 \
#       --row-elements 2048 --field wave

set -euo pipefail

REPS=${1:-5}
RTTS=(0 10 50)
. "$(dirname "$0")/netem.sh"

# 0 is the exact transfer: the number every bounded one has to beat.
BOUNDS=(0 0.0001 0.001 0.01 0.1 1.0)
# Capped bandwidths, at one round trip a wide-area link plausibly has.
RATES=(1gbit 2gbit 5gbit 10gbit)
RATE_RTT=10
# Connections, which is how many cores compress. The VMs have 16.
STREAMS=(4 8 16 24)

BYTES=$((1 << 30))
PORT=50191    # aex.toml

trap restore EXIT

# run <label> <aexbench flags...>
run() {
    local label=$1
    shift
    flush_metrics
    ssh "$CLIENT" "bash -lc 'cd aex2 && target/release/aexbench \
        http://$SERVER_IP:$PORT wave.npy --bytes $BYTES --reps 1 --prefault \
        --pattern none $* --label \"$label\"'"
}

set_buffers tuned
for r in $(seq "$REPS"); do
    for rtt in "${RTTS[@]}"; do
        set_rtt "$rtt"
        for eps in "${BOUNDS[@]}"; do
            run "rtt+$rtt abs_error=$eps" --abs-error "$eps"
        done
    done
    for rate in "${RATES[@]}"; do
        set_rtt_rate "$RATE_RTT" "$rate"
        for eps in "${BOUNDS[@]}"; do
            run "rate=$rate abs_error=$eps" --abs-error "$eps"
        done
    done
    # Uncapped: what this is after is the ceiling the server's cores set, and a
    # capped link would hide it.
    clear_rtt
    for s in "${STREAMS[@]}"; do
        for eps in "${BOUNDS[@]}"; do
            run "streams=$s abs_error=$eps" --streams "$s" --abs-error "$eps"
        done
    done
done
