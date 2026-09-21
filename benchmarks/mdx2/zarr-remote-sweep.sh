#!/bin/bash
#
# zarr-python reading a store over HTTP against AEX2 reading the same store,
# from the same client, over the same link, at a sweep of distances.
#
# The comparison recorded so far had zarr-python reading the store on the
# machine that holds it, which leaves open the obvious objection that the store
# could simply be served over HTTP. Here both readers are on the client and
# both cross the link; nginx serves the very same directories the AEX2 server
# has as its roots.
#
# What differs is where the decompression happens, and therefore what travels:
# AEX2 decompresses on the server and sends raw bytes, HTTP sends the
# compressed chunks and the client decompresses. So the noisy stores (35 %) are
# the close comparison and the counting ones (3 %) are where HTTP moves a
# thirtieth of the bytes. COUNTERS=1 records both sides of that per run.
#
# Socket buffers are tuned throughout, as in the other delay sweeps, and the
# server is the 64 MiB decode cache one so that every chunk is decompressed
# rather than served from the previous rep.
#
# Usage: [RTTS="0 100"] [STORES="mem.zarr"] [COUNTERS=1] zarr-remote-sweep.sh [reps]

set -euo pipefail

REPS=${1:-3}
. "$(dirname "$0")/netem.sh"
. "$(dirname "$0")/measure.sh"

read -r -a RTTS <<< "${RTTS:-0 5 25 100}"
read -r -a STORES <<< "${STORES:-mem-noisy.zarr mem.zarr}"
# zarr-python's async concurrency, swept where it can matter: one process has
# ten requests in flight by default, sixteen processes already have a hundred
# and sixty. Not sweeping it would be measuring a default rather than a reader.
read -r -a CONCURRENCY <<< "${CONCURRENCY:-10 64 256}"
VIA=${VIA:-obstore}
HTTP=http://$SERVER_IP:8080
AEX=http://$SERVER_IP:50391
PY=".venv/bin/python -u benchmarks/mdx2/zarr-procs.py"
ELEMENTS=$((1 << 30))

restore() {
    clear_rtt
    set_buffers default
}
trap restore EXIT

set_buffers tuned
for _ in $(seq "$REPS"); do
    for rtt in "${RTTS[@]}"; do
        set_rtt "$rtt"
        for store in "${STORES[@]}"; do
            case $store in *noisy*) p=none ;; *) p=counting-f32 ;; esac
            tag="rtt+$rtt $store"
            flush_metrics
            for c in "${CONCURRENCY[@]}"; do
                run "$tag zarr p1 c$c" \
                    "$PY $HTTP/$store $ELEMENTS --via $VIA --concurrency $c --procs 1 --reps 1 \
                        --label \"$tag zarr p1 c$c\""
            done
            # The whole machine, which is what a user has to reach for to get
            # past one interpreter's ceiling.
            run "$tag zarr p16" \
                "$PY $HTTP/$store $ELEMENTS --via $VIA --procs 16 --reps 1 --label \"$tag zarr p16\""
            # The same two points for AEX2, expressed as connections.
            for s in 1 16; do
                run "$tag aex s$s" \
                    "$PY $AEX/$store $ELEMENTS --backend aex --streams $s --procs 1 --reps 1 \
                        --label \"$tag aex s$s\""
            done
            # The Rust client, as the ceiling the Python one is read against.
            for s in 1 16; do
                run "$tag aexbench s$s" \
                    "target/release/aexbench $AEX $store --bytes \$(($ELEMENTS * 4)) \
                        --streams $s --reps 1 --prefault --pattern $p \
                        --label \"$tag aexbench s$s\""
            done
        done
    done
done
