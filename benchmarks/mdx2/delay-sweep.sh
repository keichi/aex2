#!/bin/bash
#
# Sweep added RTT between the VMs with netem and time AEX2 and iPerf3 at each.
#
# Half the delay goes on each side, so ACKs are late too, and only traffic to
# the peer is delayed, so ssh stays fast. Each pass visits every condition once,
# because the shared link drifts over the day.
#
# Usage: benchmarks/mdx2/delay-sweep.sh [reps]   (the aex.toml server must be up)

set -euo pipefail

REPS=${1:-3}
RTTS=(0 1 5 10 25 50 100)
SERVER=aex2-eval1 SERVER_IP=192.168.100.207
CLIENT=aex2-eval2 CLIENT_IP=192.168.101.235
DEFAULT_RMEM="4096 131072 6291456" DEFAULT_WMEM="4096 16384 4194304"
# Holds a single stream's BDP at about 20 Gbit/s and 100 ms.
TUNED_MAX=$((256 << 20))

set_rtt() {
    local half
    half=$(echo "$1 / 2" | bc -l)
    for pair in "$SERVER $CLIENT_IP" "$CLIENT $SERVER_IP"; do
        set -- $pair
        # A 20 Gbit/s flow queues about 180k packets at 100 ms; the default 1000 drops.
        ssh "$1" "sudo tc qdisc replace dev enp3s0 root handle 1: prio bands 4 &&
            sudo tc qdisc replace dev enp3s0 parent 1:4 handle 40: netem delay ${half}ms limit 1000000 &&
            sudo tc filter replace dev enp3s0 parent 1: protocol ip prio 1 u32 match ip dst $2/32 flowid 1:4"
    done
}

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
    for h in $SERVER $CLIENT; do ssh "$h" 'sudo tc qdisc del dev enp3s0 root 2>/dev/null; pkill -x iperf3' || true; done
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
                    --streams $s --reps 1 --label \"aex $label\"'"
                flush_metrics
                gbit=$(ssh $CLIENT "iperf3 -c $SERVER_IP -t 10 -Z -R -P $s -J" |
                    jq '.end.sum_received.bits_per_second / 1e9')
                printf 'iperf %-40s %.2f Gbit/s\n' "$label" "$gbit"
            done
        done
    done
done
