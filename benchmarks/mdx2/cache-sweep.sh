#!/bin/bash
#
# Sweep added RTT and time reads the client cache answers against transfers.
#
# A hit never touches the network, so what the sweep shows is how far apart
# the two grow with distance. Each pass visits every delay once, because the
# shared link drifts over the day.
#
# Usage: benchmarks/mdx2/cache-sweep.sh [passes] [reps]   (the aex.toml server must be up)
#   KERNEL=tuned turns off slow start after idle and raises the socket buffer
#   ceilings first, so the transfers measure AEX2 rather than those defaults.

set -euo pipefail

PASSES=${1:-3}
REPS=${2:-20}
RTTS=(0 5 25 100)
. "$(dirname "$0")/netem.sh"

trap restore EXIT
if [ "${KERNEL:-default}" = tuned ]; then
    set_buffers tuned
    set_idle_restart off
fi

for p in $(seq "$PASSES"); do
    for rtt in "${RTTS[@]}"; do
        set_rtt "$rtt"
        ssh "$CLIENT" "bash -lc 'cd aex2 && .venv/bin/python benchmarks/mdx2/cachebench.py \
            $SERVER_IP:50191 wave.npy --reps $REPS --label ${KERNEL:-default}/rtt+$rtt'"
    done
done
