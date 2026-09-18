#!/bin/bash
#
# Sweep added RTT and time small selections taken one by one against gathered.
#
# What gather removes is round trips, so the reading only means something on a
# link with some. Each pass visits every delay once, because the shared link
# drifts over the day.
#
# Usage: benchmarks/mdx2/gather-sweep.sh [reps]   (the aex.toml server must be up)

set -euo pipefail

REPS=${1:-3}
RTTS=(0 1 5 25 100)
. "$(dirname "$0")/netem.sh"

trap clear_rtt EXIT

for r in $(seq "$REPS"); do
    for rtt in "${RTTS[@]}"; do
        set_rtt "$rtt"
        ssh "$CLIENT" "bash -lc 'cd aex2 && target/release/latbench \
            http://$SERVER_IP:50191 mem.npy --reps 200 --sizes 4096 \
            --gather 2,8,32,64 --gather-reps 20'" | sed "s/^/rtt+$rtt /"
    done
done
