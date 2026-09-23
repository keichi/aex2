#!/bin/bash
#
# What AEX2 can trade for the bytes it puts on the wire, against zarr-python
# reading the same store over HTTP.
#
# The remote comparison found the one place zarr-python wins: a store that
# compresses well, read from far enough away that the link decides, because
# AEX2 decompresses on the server and sends raw bytes. AEX2 has two answers
# that comparison never asked for -- deflate on the wire, which is exact, and
# SZ3, which is not -- and this measures what each costs and buys.
#
# The axis is bandwidth rather than round trip. A compressed transfer is
# limited by the cores that compress it, not by the link, so it pays only once
# the link is slower than those cores can feed. Round trip is swept too,
# uncapped, because zarr-python's thousand GETs answer to it and a compressed
# transfer does not.
#
# The fixture is wave.zarr: a smooth two-dimensional field with noise in the
# low bits. The counting stores compress unboundedly under an error bound and
# would say nothing about real data.
#
# Usage: [RTTS="0 100"] [RATES="1gbit"] [STORE=... ELEMENTS=...] zarr-lossy-sweep.sh [reps]
#   A cold disk-resident round: HTTP=http://SERVER:8080/disk COLD=/home/mdxuser/disk
#   Both sides must be built with the sz feature, and the fixture must exist:
#   mkzarr.py /mnt/aexram/wave.zarr 268435456 plain wave

set -euo pipefail

REPS=${1:-3}
. "$(dirname "$0")/netem.sh"
. "$(dirname "$0")/measure.sh"

read -r -a RTTS <<< "${RTTS:-0 25 100}"
read -r -a RATES <<< "${RATES:-10gbit 2gbit 1gbit}"
# The round trip the capped points are measured at: a wide-area link has one.
RATE_RTT=${RATE_RTT:-10}
STORE=${STORE:-wave.zarr}
COUNTERS=${COUNTERS:-1}
# Overridable so that the disk-resident stores, which nginx serves under
# /disk/, can be swept by the same script.
HTTP=${HTTP:-http://$SERVER_IP:8080}
AEX=http://$SERVER_IP:50391
PY=".venv/bin/python -u benchmarks/mdx2/read-procs.py"
ELEMENTS=${ELEMENTS:-$((1 << 28))}    # 1 GiB, as the SZ3 measurement used
# Exact, exact-but-smaller, and three bounds. The field runs to about 300, so
# 1.0 is coarse and 1e-3 is not.
IFS="|" read -r -a QUALITIES <<< "${QUALITIES:-|--codec gzip|--abs-error 0.001|--abs-error 0.01|--abs-error 1.0}"

trap restore EXIT

point() {
    local tag=$1 q name
    COLD_STORE=$STORE
    flush_metrics
    run "$tag zarr p16" \
        "$PY $HTTP/$STORE $ELEMENTS --via obstore --procs 16 --reps 1 --label \"$tag zarr p16\""
    for q in "${QUALITIES[@]}"; do
        name=${q:-exact}
        run "$tag aex $name" \
            "$PY $AEX/$STORE $ELEMENTS --backend aex --streams 16 --procs 1 --reps 1 $q \
                --label \"$tag aex $name\""
    done
}

set_buffers tuned
for _ in $(seq "$REPS"); do
    for rtt in "${RTTS[@]}"; do
        set_rtt "$rtt"
        point "rtt+$rtt rate=none"
    done
    for rate in "${RATES[@]}"; do
        set_rtt_rate "$RATE_RTT" "$rate"
        point "rtt+$RATE_RTT rate=$rate"
    done
done
