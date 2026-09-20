# Added RTT between the VMs, shared by the sweeps that need it.
#
# Half the delay goes on each side, so ACKs are late too, and only traffic to
# the peer is delayed, so ssh stays fast. Source this, then call set_rtt; the
# caller's EXIT trap has to call clear_rtt, or the VMs keep the delay.

SERVER=${SERVER:-aex2-eval1} SERVER_IP=${SERVER_IP:-192.168.100.207}
CLIENT=${CLIENT:-aex2-eval2} CLIENT_IP=${CLIENT_IP:-192.168.101.235}

set_rtt() {
    set_rtt_rate "$1" ""
}

# Added RTT, and optionally a bandwidth cap (a tc rate such as `1gbit`).
#
# The cap goes on both sides at its full value rather than half: two shapers of
# the same rate in series still pass that rate, and the link is only ever
# measured in one direction anyway.
#
# Wanted because delay alone cannot answer what compression is for. This link
# runs at 26 to 133 Gbit/s, so even at 50 ms of round trip an exact transfer
# beats a compressed one; the trade only turns over on a narrow link.
set_rtt_rate() {
    local half rate=$2
    half=$(echo "$1 / 2" | bc -l)
    for pair in "$SERVER $CLIENT_IP" "$CLIENT $SERVER_IP"; do
        set -- $pair
        # A 20 Gbit/s flow queues about 180k packets at 100 ms; the default 1000 drops.
        ssh "$1" "sudo tc qdisc replace dev enp3s0 root handle 1: prio bands 4 &&
            sudo tc qdisc replace dev enp3s0 parent 1:4 handle 40: netem delay ${half}ms ${rate:+rate $rate} limit 1000000 &&
            sudo tc filter replace dev enp3s0 parent 1: protocol ip prio 1 u32 match ip dst $2/32 flowid 1:4"
    done
}

clear_rtt() {
    for h in $SERVER $CLIENT; do
        ssh "$h" 'sudo tc qdisc del dev enp3s0 root 2>/dev/null' || true
    done
}

DEFAULT_RMEM="4096 131072 6291456" DEFAULT_WMEM="4096 16384 4194304"
# Holds a single stream's BDP at about 20 Gbit/s and 100 ms.
TUNED_MAX=$((256 << 20))

# Kernel socket buffer ceilings: `default` or `tuned`. The default ones cap a
# single stream below the BDP from about 5 ms of round trip upwards, so a sweep
# that means to measure anything else has to raise them first.
# `net.core.*mem_max` only caps a buffer size asked for by setsockopt, which
# autotuning never does, so it moves with the ceilings rather than separately.
DEFAULT_CORE_MAX=212992

set_buffers() {
    local rmem=$DEFAULT_RMEM wmem=$DEFAULT_WMEM core=$DEFAULT_CORE_MAX
    [ "$1" = tuned ] && rmem="4096 131072 $TUNED_MAX" wmem="4096 16384 $TUNED_MAX" core=$TUNED_MAX
    for h in $SERVER $CLIENT; do
        ssh "$h" "sudo sysctl -q -w net.ipv4.tcp_rmem='$rmem' net.ipv4.tcp_wmem='$wmem' \
            net.core.rmem_max=$core net.core.wmem_max=$core"
    done
}

# Cached ssthresh from the previous run would skip slow start.
flush_metrics() {
    for h in $SERVER $CLIENT; do ssh "$h" 'sudo ip tcp_metrics flush all' 2>/dev/null || true; done
}
