# Added RTT between the VMs, shared by the sweeps that need it.
#
# Half the delay goes on each side, so ACKs are late too, and only traffic to
# the peer is delayed, so ssh stays fast. Source this, then call set_rtt; the
# caller's EXIT trap has to call clear_rtt, or the VMs keep the delay.

SERVER=${SERVER:-aex2-eval1} SERVER_IP=${SERVER_IP:-192.168.100.207}
CLIENT=${CLIENT:-aex2-eval2} CLIENT_IP=${CLIENT_IP:-192.168.101.235}

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

clear_rtt() {
    for h in $SERVER $CLIENT; do
        ssh "$h" 'sudo tc qdisc del dev enp3s0 root 2>/dev/null' || true
    done
}
