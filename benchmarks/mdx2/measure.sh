# One measurement, and what it cost the server. Shared by the sweeps that
# compare readers rather than settings, because for those the bytes on the
# wire and the CPU behind them are half the answer.
#
# Source this after netem.sh: it uses $SERVER and $CLIENT.
#
# COLD names the directory the stores are in, on the server, and must be a path
# that means the same there: /home/mdxuser/disk, not ~/disk, which the local
# shell would expand to the wrong home.

# Bytes the server sent and CPU seconds it spent, as one pair to difference.
# Both readers are served by the same machine, so one pair covers aex-server
# decompressing, nginx reading files, and HSDS's nodes alike.
counters() {
    ssh "$SERVER" bash -s <<'EOF'
grep enp3s0 /proc/net/dev | tr ':' ' ' | awk '{printf "%s ", $10}'
for p in $(pgrep -x aex-server) $(pgrep -x nginx) $(pgrep -f hsds); do cat "/proc/$p/stat"; done |
    awk '{s += $14 + $15} END {print s}'
EOF
}

# Evict $COLD_STORE's pages on the server, so the run reads the disk the store
# lives on. Both readers meet the same files, so this is even-handed: it is the
# stored bytes that are read, not the delivered ones.
cold() {
    [ -n "${COLD:-}" ] || return 0
    ssh "$SERVER" "cd aex2 && target/release/dropcache \$(find $COLD/$COLD_STORE -type f)" \
        > /dev/null
}

# run <label> <command to run on the client>. COUNTERS=1 adds the server's side.
run() {
    local label=$1 before after
    shift
    cold
    [ -n "${COUNTERS:-}" ] && before=$(counters)
    ssh "$CLIENT" "bash -lc 'cd aex2 && $*'"
    if [ -n "${COUNTERS:-}" ]; then
        after=$(counters)
        echo "$before $after" |
            awk -v l="$label" '{printf "%s  wire %6.0f MiB  server cpu %5.2f s\n",
                l, ($3 - $1) / 1048576, ($4 - $2) / 100}'
    fi
}
