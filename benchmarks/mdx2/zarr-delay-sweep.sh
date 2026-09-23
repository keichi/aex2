#!/bin/bash
#
# Sweep added RTT and time the same 4 GiB from three stores: uncompressed
# `.npy`, and Zarr compressed well and badly.
#
# What this asks is not what the codec sweeps ask. Zarr's zstd is storage-side
# only -- the server decompresses and sends raw bytes -- so all three put the
# same 4 GiB on the wire and differ only in what the server pays to produce
# them. The question is how far away a client has to be before that stops
# mattering.
#
# Socket buffers are tuned throughout: the default ceilings cap a stream below
# the BDP from about 5 ms upwards, and benchmark-delay.md already measured
# that. The decode cache is the 64 MiB one, so every chunk is decompressed
# rather than served from a previous rep.
#
# Usage: benchmarks/mdx2/zarr-delay-sweep.sh [reps]

set -euo pipefail

REPS=${1:-3}
RTTS=(0 5 25 100)
STORES=(mem.npy mem-noisy.zarr mem.zarr)
. "$(dirname "$0")/netem.sh"

trap restore EXIT

set_buffers tuned
for r in $(seq "$REPS"); do
    for rtt in "${RTTS[@]}"; do
        set_rtt "$rtt"
        for s in 1 16; do
            for store in "${STORES[@]}"; do
                case $store in *noisy*) p=none ;; *) p=counting-f32 ;; esac
                flush_metrics
                ssh "$CLIENT" "bash -lc 'cd aex2 && target/release/aexbench \
                    http://$SERVER_IP:50391 $store --bytes \$((4<<30)) \
                    --streams $s --reps 1 --prefault --pattern $p \
                    --label \"rtt+$rtt s$s $store\"'"
            done
        done
    done
done
