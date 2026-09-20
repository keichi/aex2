#!/bin/bash
#
# What error-bounded compression costs and what it buys, against the round trip.
#
# The question is not whether SZ3 compresses -- it does -- but where the trade
# turns over. Compressing costs CPU on both sides and gives up the receiver's
# zero copy; it pays only once the link is slower than what the codec can feed
# it. The exact transfer is measured alongside every point as the thing to beat.
#
# Three axes, because the first one on its own gave the wrong impression. Round
# trip alone never turns the trade over here: this link runs at 26 to 133
# Gbit/s, so even at 50 ms an exact transfer still beats a compressed one. It
# is *bandwidth* that decides, so the second sweep caps it.
#
# The third is connections, which for a compressed transfer is not a network
# knob at all: SZ3 runs single-threaded inside each connection thread, so it is
# the only thing that decides how many cores compress. The shipping default of
# 8 was chosen for lossless transfers, where the link is the limit; it leaves
# half of a 16-core server idle here and halves the bandwidth at which
# compression starts to pay.
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
# Capped bandwidths, at one round trip a wide-area link plausibly has.
RATES=(1gbit 2gbit 5gbit 10gbit)
RATE_RTT=10
# Connections, which is how many cores compress. The VMs have 16.
STREAMS=(4 8 16 24)

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
