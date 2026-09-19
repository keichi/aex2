#!/bin/bash
#
# poolbench: how should a strided selection's reads be issued?
#
# A gather reads fragments with gaps between them, because `selection.rs` can
# only join runs closer than SPAN_GAP_BYTES. This sweeps the fragment size and
# the gap against the ways of getting more than one read in flight. Cold disk,
# which is where the reads cost anything.
#
# Usage: benchmarks/mdx2/gather-io-sweep.sh [rounds]

set -euo pipefail

SERVER=${SERVER:-aex2-eval1}
CLIENT=${CLIENT:-aex2-eval2}
SINK=${SINK:-192.168.101.235:50399}
ROUNDS=${1:-5}
DISK=${DISK:-/home/mdxuser/disk/disk.npy}
STREAMS=${STREAMS:-4}

# frag:gap:file-span. The last two read a quarter of the file, because at those
# fragment sizes a whole pass is thousands of times more reads.
PATTERNS="0:0:$((4 << 30)) 65536:65536:$((4 << 30)) 16384:49152:$((4 << 30)) \
    4096:61440:$((1 << 30)) 4096:4096:$((1 << 30))"

ssh "$CLIENT" 'pkill -x poolbench || true; mkdir -p ~/logs; cd aex2;
    setsid nohup target/release/poolbench sink > ~/logs/gather-sink.log 2>&1 < /dev/null &'
sleep 1
trap 'ssh "$CLIENT" "pkill -x poolbench || true"' EXIT

for _ in $(seq "$ROUNDS"); do
    for p in $PATTERNS; do
        IFS=: read -r frag gap bytes <<< "$p"
        for m in serial fadvise uring uring-zc; do
            ssh "$SERVER" "bash -lc 'cd aex2 && target/release/poolbench send $DISK \
                --to $SINK --bytes $bytes --streams $STREAMS --mode $m --depth 8 \
                --frag $frag --gap $gap --cold'"
        done
    done
done
