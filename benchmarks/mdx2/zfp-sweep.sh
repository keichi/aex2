#!/bin/bash
#
# SZ3 and ZFP side by side: which error-bounded codec wins where.
#
# sz-sweep.sh already settled the shape of the trade -- a compressed transfer
# runs at a CPU-bound constant that moves with neither round trip nor
# bandwidth, so the link speed at which compression starts to pay *is* that
# constant -- and this sweep does not re-derive it. It measures each codec's
# constant, and checks once on a narrow link that the constant really is one.
#
# Both codecs are measured in the same pass, alternately, because the numbers
# in docs/benchmark-sz.md were taken on another day and this VM pair drifts
# about 25 % over one (eval-mdx2.md). An SZ3 column here is a fresh
# measurement, not a copy.
#
# Dropped from sz-sweep.sh, with reasons:
#   - the RTT axis. Its 17 % droop comes from credit being counted in logical
#     bytes, which is the client's accounting and not the codec's. One point
#     at the end checks that it is the same droop for both.
#   - three of the four bandwidth caps. The compressed row was flat across all
#     four, so one cap tests that claim for ZFP and three more only repeat it.
#   - streams 4 and 24. 4 only confirmed linearity and 24 was already flat.
#
# Note when reading ratios: zfp honours a bound by rounding it down to a power
# of two, so at 1e-4, 1e-3, 1e-2 and 1e-1 it is compressing to a tighter bound
# than SZ3 is. Every ZFP ratio here is therefore a lower bound on what it would
# do at the same effective tolerance.
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
    # netem off first: these two blocks are after the ceiling the server's
    # cores set, and a cap left over from the last rep would hide it. `replace`
    # keeps whatever the previous call did not mention, so this is a teardown
    # and not a tidy-up (see netem.sh).
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
