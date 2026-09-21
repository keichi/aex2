#!/bin/bash
#
# Start (or stop) the HSDS servers the comparison reads from, on aex2-eval1.
#
# A standalone HSDS runs one service node, and every byte a client reads is
# copied by that one Python process; it stops at about 300 MiB/s however many
# data nodes are behind it. The deployments the HDF Group ships for clusters
# put several service nodes behind a load balancer, so several servers are
# started here instead, on consecutive ports, and the readers are spread over
# them (read-procs.py --endpoints).
#
# The data node ports are hardcoded to 6101+ in this HSDS, so a second server
# would take the first one's ports. hsds_app.py is patched on the VM to read
# HSDS_DN_PORT instead, which is what keeps them apart.
#
# Usage: hsds.sh [servers] [data nodes each]   (hsds.sh stop to stop them)

set -euo pipefail

VENV=${VENV:-$HOME/hsds-venv}
ROOT=${ROOT:-/mnt/aexram}
BUCKET=${BUCKET:-hsds}
PORT=${PORT:-5101}

if [ "${1:-}" = stop ]; then
    pkill -f "$VENV/bin/hsds" || true
    exit 0
fi

servers=${1:-4}
nodes=${2:-4}
for i in $(seq 0 $((servers - 1))); do
    # Far enough apart that one server's data nodes cannot reach the next.
    HSDS_DN_PORT=$((6101 + i * 100)) nohup "$VENV/bin/hsds" \
        --root_dir "$ROOT" --bucket_name "$BUCKET" --port $((PORT + i)) --count "$nodes" \
        --hs_username test --hs_password test \
        --logfile "/tmp/hsds$i.log" --loglevel WARNING > "/tmp/hsds$i.out" 2>&1 < /dev/null &
done
sleep 15
for i in $(seq 0 $((servers - 1))); do
    HS_ENDPOINT=http://localhost:$((PORT + i)) HS_USERNAME=test HS_PASSWORD=test \
        HS_BUCKET=$BUCKET "$VENV/bin/hsinfo" | sed -n '2p;7p' | paste -sd' ' -
done
