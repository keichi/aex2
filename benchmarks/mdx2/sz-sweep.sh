#!/bin/bash
#
# What error-bounded compression costs and what it buys, against the round trip.
#
# The question is not whether SZ3 compresses -- it does -- but where the trade
# turns over. Compressing costs CPU on both sides and gives up the receiver's
# zero copy; it pays only once the link is slower than what the codec can feed
# it. So every point is measured at several round trips, and the exact transfer
# is measured alongside as the thing to beat.
#
# The fixture is `wave.npy`: a smooth 2-d float32 field with noise in the low
# bits. A counting ramp would compress unboundedly and say nothing about real
# data, so it is not used here.
#
# Each pass visits every point once and the passes are repeated, because the
# shared link drifts over the day; take the median across passes.
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

BYTES=$((1 << 30))
PORT=50191    # aex.toml

restore() {
    clear_rtt
    set_buffers default
}
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
done
