#!/bin/bash
#
# SZ3 and ZFP in the same pass, since the VMs drift ~25 % between days.
# ZFP rounds its bound down to a power of two, so its ratios are lower bounds.
# See docs/benchmark-zfp.md.
#
# Usage: benchmarks/mdx2/zfp-sweep.sh [reps]
#   The server must be up with both codecs: aex.toml (50191).
#   The fixture must exist: mknpy /mnt/aexram/wave.npy 268435456 \
#       --row-elements 2048 --field wave

set -euo pipefail

REPS=${1:-5}
. "$(dirname "$0")/netem.sh"

# The bounds sz-sweep.sh used, so an SZ3 column can be sanity checked against
# docs/benchmark-sz.md even though it is not compared with it.
BOUNDS=(0.0001 0.001 0.01 0.1 1.0)
CODECS=(sz zfp)
# Connections, which is how many cores compress. 8 is the shipping default,
# 16 is the VMs' core count.
STREAMS=(8 16)
# One narrow link, the wide-area rate the conclusion is about.
CAP=1gbit
CAP_RTT=10
# The round trip the credit droop shows up at, at one bound.
DROOP_RTT=50
DROOP_BOUND=0.001

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

# Every bound on every codec at one setting, and the exact transfer once --
# it is the same number whichever codec is not being used.
block() {
    local what=$1
    shift
    run "$what codec=none abs_error=0" "$@"
    for eps in "${BOUNDS[@]}"; do
        for codec in "${CODECS[@]}"; do
            run "$what codec=$codec abs_error=$eps" \
                --codec "$codec" --abs-error "$eps" "$@"
        done
    done
}

# Both codecs return what they were asked for, on the fixture whose content
# aexbench can check. The sweep itself runs --pattern none, so this is the
# only place the bound is verified rather than assumed.
for codec in "${CODECS[@]}"; do
    ssh "$CLIENT" "bash -lc 'cd aex2 && target/release/aexbench \
        http://$SERVER_IP:$PORT mem.npy --bytes $((1 << 26)) --reps 1 \
        --codec $codec --abs-error 0.001 --label \"check $codec\"'"
done

set_buffers tuned
for r in $(seq "$REPS"); do
    # netem off first: a cap left from the last rep would hide the cores' ceiling (see netem.sh).
    clear_rtt
    for s in "${STREAMS[@]}"; do
        block "streams=$s" --streams "$s"
    done
    # The narrow link: does each codec hold its constant, and by how much does
    # it beat the exact transfer where a wide-area line would put it.
    set_rtt_rate "$CAP_RTT" "$CAP"
    block "rate=$CAP" --streams 8
    # One round trip, to see whether the credit droop differs between codecs.
    # It should track the compression ratio and nothing else.
    clear_rtt
    set_rtt "$DROOP_RTT"
    for codec in "${CODECS[@]}"; do
        run "rtt=$DROOP_RTT codec=$codec abs_error=$DROOP_BOUND" \
            --codec "$codec" --abs-error "$DROOP_BOUND" --streams 8
    done
done
