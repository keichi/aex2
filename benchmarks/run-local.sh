#!/bin/bash
#
# Local transfer benchmark: AEX2 against what the machine can do at all.
#
# Three numbers matter and this measures all three. iperf3 on loopback says how
# fast bytes can cross a socket; pread says how fast they can leave storage; and
# AEX2 says how much of that a real transfer keeps. On loopback the first two
# are the same memory bus, so the interesting reading is the CPU per gibibyte
# rather than the throughput.
#
# Data is measured in two places. A RAM disk holds the resident case, where
# every read is a memory copy. A file half again the size of RAM holds the
# disk-resident case: it cannot be cached, and reading its tail evicts its head,
# which is how a cold read is arranged without privileges to drop the cache.
#
# macOS only, for the RAM disk and the eviction trick. On Linux, mount a tmpfs
# and write 3 to /proc/sys/vm/drop_caches instead.
#
# Usage:
#   benchmarks/run-local.sh setup      # build, make the RAM disk and fixtures
#   benchmarks/run-local.sh run        # measure
#   benchmarks/run-local.sh teardown   # remove both

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${AEX_BENCH_WORK:-${TMPDIR:-/tmp}/aex-bench}
MEM_MOUNT=${AEX_BENCH_MEM:-/tmp/aexram}
CONTROL_PORT=${AEX_BENCH_CONTROL_PORT:-50151}
DATA_PORT=${AEX_BENCH_DATA_PORT:-50152}
IPERF_PORT=${AEX_BENCH_IPERF_PORT:-5301}

# 4 GiB per transfer: long enough to time, small enough to keep the client's
# buffer from competing with the page cache for the machine.
XFER_BYTES=$((4 * 1024 * 1024 * 1024))
MEM_ELEMENTS=$((XFER_BYTES / 4))
RAM_BYTES=$(sysctl -n hw.memsize)
# Half again the size of RAM, so that nothing can hold it and reading the tail
# evicts the head.
DISK_ELEMENTS=$((RAM_BYTES * 3 / 2 / 4))
# Everything past the first 8 GiB, which is more than RAM and so evicts it.
BUST_SKIP_MIB=8192

BIN=$ROOT/target/release
CHUNKS=${AEX_BENCH_CHUNKS:-262144 1048576 2097152 4194304 16777216}

setup() {
    cargo build --release --manifest-path "$ROOT/Cargo.toml" -p aex-server -p aex-bench

    mkdir -p "$WORK/disk" "$MEM_MOUNT"
    # Compare resolved paths: /tmp is a symlink on macOS, and mount reports the
    # target, so matching on the spelling would mount a second disk over the
    # first and quietly wire down twice the memory.
    if ! mounted_device >/dev/null; then
        # Twice the fixture, so the filesystem has somewhere to put it.
        local sectors=$((MEM_ELEMENTS * 4 * 2 / 512))
        local dev
        dev=$(hdiutil attach -nomount "ram://$sectors" | tr -d ' \t')
        newfs_hfs -v aexbench "$dev" >/dev/null
        mount -t hfs "$dev" "$MEM_MOUNT"
        echo "ram disk $dev on $MEM_MOUNT"
    fi

    [ -f "$MEM_MOUNT/mem.npy" ] || "$BIN/mknpy" "$MEM_MOUNT/mem.npy" "$MEM_ELEMENTS"
    [ -f "$WORK/disk/disk.npy" ] || "$BIN/mknpy" "$WORK/disk/disk.npy" "$DISK_ELEMENTS"
    df -h "$MEM_MOUNT" "$WORK" | tail -2
}

teardown() {
    stop_server || true
    rm -rf "$WORK"
    local dev
    if dev=$(mounted_device); then
        umount "$MEM_MOUNT" && hdiutil detach "$dev"
    fi
}

# The device backing the RAM disk, if one is mounted there.
mounted_device() {
    local resolved
    resolved=$(cd "$MEM_MOUNT" 2>/dev/null && pwd -P) || return 1
    mount | awk -v mount="$resolved" '$3 == mount { print $1; found = 1 }
                                     END { exit !found }'
}

start_server() {
    "$BIN/aex-server" --control-addr "127.0.0.1:$CONTROL_PORT" \
        --data-addr "127.0.0.1:$DATA_PORT" \
        --root "$MEM_MOUNT" --root "$WORK/disk" >"$WORK/server.log" 2>&1 &
    echo $! >"$WORK/server.pid"
    sleep 1.5
}

stop_server() {
    [ -f "$WORK/server.pid" ] || return 0
    kill "$(cat "$WORK/server.pid")" 2>/dev/null || true
    rm -f "$WORK/server.pid"
}

# Cumulative CPU seconds of the running server.
server_cpu() {
    ps -p "$(cat "$WORK/server.pid")" -o time= | tr -d ' ' |
        awk -F: '{ print ($1 * 60) + $2 }'
}

# Evict the head of the disk fixture by reading everything after it.
bust_cache() {
    dd if="$WORK/disk/disk.npy" of=/dev/null bs=8m skip=$((BUST_SKIP_MIB / 8)) 2>/dev/null
}

url() { echo "http://127.0.0.1:$CONTROL_PORT"; }

run() {
    echo "### environment"
    echo "$(sysctl -n hw.model), $(sysctl -n machdep.cpu.brand_string), \
$(sysctl -n hw.ncpu) cores, $((RAM_BYTES / 1024 / 1024 / 1024)) GiB RAM"
    sw_vers | tr '\n' ' '
    echo
    "$BIN/aex-server" --version
    echo

    echo "### baseline: loopback TCP (iperf3)"
    iperf3 -s -p "$IPERF_PORT" --logfile "$WORK/iperf-server.log" &
    local iperf_pid=$!
    sleep 1
    for block in 128K 256K 1M; do
        printf '%-44s' "iperf3 -l $block"
        iperf3 -c 127.0.0.1 -p "$IPERF_PORT" -t 6 -l "$block" -f m |
            awk '/receiver/ { printf "%14.0f MiB/s (%5.1f Gbit/s)\n", $7 / 8.388608, $7 / 1000 }'
    done
    kill $iperf_pid 2>/dev/null || true
    echo

    echo "### baseline: pread, data resident in memory (RAM disk)"
    for chunk in $CHUNKS; do
        "$BIN/preadbench" "$MEM_MOUNT/mem.npy" --bytes "$XFER_BYTES" --chunk "$chunk" \
            --reps 3 --label memory
    done
    echo

    echo "### baseline: pread, data on disk (page cache evicted before each run)"
    for chunk in $CHUNKS; do
        bust_cache
        "$BIN/preadbench" "$WORK/disk/disk.npy" --bytes "$XFER_BYTES" --chunk "$chunk" \
            --reps 1 --label disk-cold
    done
    echo

    start_server
    trap stop_server EXIT

    echo "### AEX2, data resident in memory (RAM disk)"
    local before after
    before=$(server_cpu)
    for chunk in $CHUNKS; do
        "$BIN/aexbench" "$(url)" mem.npy --bytes "$XFER_BYTES" --chunk "$chunk" \
            --reps 5 --label memory
    done
    after=$(server_cpu)
    echo "server cpu over the sweep: $(echo "$after $before" | awk '{ printf "%.1f", $1 - $2 }') s"
    echo

    echo "### AEX2, data on disk (page cache evicted before each run)"
    for chunk in $CHUNKS; do
        bust_cache
        "$BIN/aexbench" "$(url)" disk.npy --bytes "$XFER_BYTES" --chunk "$chunk" \
            --reps 1 --label disk-cold
    done
    echo

    echo "### AEX2, one connection each, several clients at once (memory)"
    # One connection per client, so this is what parallel streams would have to
    # beat once one client can open more than one.
    for clients in 1 2 4 6 8; do
        local pids=()
        for i in $(seq "$clients"); do
            "$BIN/aexbench" "$(url)" mem.npy --bytes $((256 * 1024 * 1024)) \
                --chunk 1048576 --reps 20 --label "memory[$i/$clients]" &
            pids+=($!)
        done
        # Named, because a bare wait would also wait for the server.
        wait "${pids[@]}"
        echo
    done

    echo "### AEX2, latency of a small read (memory)"
    "$BIN/latbench" "$(url)" mem.npy

    stop_server
    trap - EXIT
}

case "${1:-run}" in
    setup) setup ;;
    run) run ;;
    teardown) teardown ;;
    *) echo "usage: $0 {setup|run|teardown}" >&2; exit 2 ;;
esac
