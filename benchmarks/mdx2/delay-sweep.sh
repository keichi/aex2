#!/bin/bash
#
# Sweep added RTT between the VMs with netem and time AEX2 and iPerf3 at each.
#
# iPerf3 runs twice: for 10 s, the link's capacity, and for the same 4 GiB as
# AEX2, which pays the same slow start. AEX2 prefaults its buffer, since
# iPerf3 reuses one and never pays the page faults.
#
# Each pass visits every condition once, because the shared link drifts over
# the day. The delay itself is set by netem.sh.
#
# Usage: benchmarks/mdx2/delay-sweep.sh [reps]   (the aex.toml server must be up)

set -euo pipefail

REPS=${1:-3}
RTTS=(0 1 5 10 25 50 100)
. "$(dirname "$0")/netem.sh"
DEFAULT_RMEM="4096 131072 6291456" DEFAULT_WMEM="4096 16384 4194304"
# Holds a single stream's BDP at about 20 Gbit/s and 100 ms.
TUNED_MAX=$((256 << 20))

set_buffers() {
    local rmem=$DEFAULT_RMEM wmem=$DEFAULT_WMEM
    [ "$1" = tuned ] && rmem="4096 131072 $TUNED_MAX" wmem="4096 16384 $TUNED_MAX"
    for h in $SERVER $CLIENT; do
        ssh "$h" "sudo sysctl -q -w net.ipv4.tcp_rmem='$rmem' net.ipv4.tcp_wmem='$wmem'"
    done
}

# Cached ssthresh from the previous run would skip slow start.
flush_metrics() {
    for h in $SERVER $CLIENT; do ssh "$h" 'sudo ip tcp_metrics flush all' 2>/dev/null || true; done
}

restore() {
    clear_rtt
    for h in $SERVER $CLIENT; do ssh "$h" 'pkill -x iperf3' || true; done
    set_buffers default
}
trap restore EXIT

ssh $SERVER 'iperf3 -s -D'
for r in $(seq "$REPS"); do
    for rtt in "${RTTS[@]}"; do
        set_rtt "$rtt"
        for buf in default tuned; do
            set_buffers $buf
            for s in 1 16; do
                label="rtt+$rtt $buf s$s"
                flush_metrics
                ssh $CLIENT "bash -lc 'cd aex2 && target/release/aexbench \
                    http://$SERVER_IP:50191 mem.npy --bytes \$((4<<30)) \
                    --streams $s --reps 1 --prefault --label \"aex $label\"'"
                flush_metrics
                gbit=$(ssh $CLIENT "iperf3 -c $SERVER_IP -t 10 -Z -R -P $s -J" |
                    jq '.end.sum_received.bits_per_second / 1e9')
                printf 'iperf %-40s %.2f Gbit/s\n' "$label" "$gbit"
                flush_metrics
                # Intervals time the end to 0.1 s, the finest iperf3 allows; else whole seconds.
                gbit=$(ssh $CLIENT "iperf3 -c $SERVER_IP -n 4G -i 0.1 -Z -R -P $s -J" |
                    jq '.end.sum_received.bits_per_second / 1e9')
                printf 'iperf4g %-38s %.2f Gbit/s\n' "$label" "$gbit"
            done
        done
    done
done
