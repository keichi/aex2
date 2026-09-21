#!/bin/bash
#
# The ways of reading one HDF5 file from another machine, against each other,
# at a sweep of distances.
#
# All three readers cross the same link and read the same bytes of the same
# file: nginx serves /mnt/aexram, HSDS serves hard links into it (its chunks
# stay in the file, see hslink.py), and the AEX2 server has it as a root.
#
# What differs is where libhdf5 runs. h5py runs it on the client over a file
# object that turns seeks into range requests, so the client decompresses and
# pays a round trip per read. HSDS and AEX2 both read and decompress on the
# server and send the values, which is the same shape of answer through very
# different machinery.
#
# h5py's file object is swept over its cache settings for the same reason the
# Zarr sweep sweeps concurrency: a default is not a reader.
#
# Usage: [RTTS="0 100"] [FILES="mem.h5"] [ENDPOINTS=8] [COUNTERS=1] h5-remote-sweep.sh [reps]
#        (start that many HSDS servers first: hsds.sh $ENDPOINTS 2)

set -euo pipefail

REPS=${1:-3}
. "$(dirname "$0")/netem.sh"
. "$(dirname "$0")/measure.sh"

read -r -a RTTS <<< "${RTTS:-0 5 25 100}"
read -r -a FILES <<< "${FILES:-mem-gzip-noisy.h5 mem.h5}"
read -r -a CACHES <<< "${CACHES:-none blockcache background}"
read -r -a PROCS <<< "${PROCS:-1 16}"
# HSDS servers to spread the readers over; one of them stops at about 400 MiB/s.
ENDPOINTS=${ENDPOINTS:-8}
# Rows per h5pyd read. HSDS builds a whole selection in memory before it
# answers, so a quarter of a gigabyte at a time from sixteen readers gets it
# killed by the kernel; in 16 MiB pieces it is both alive and faster.
PIECE=${PIECE:-4194304}
HTTP=http://$SERVER_IP:8080
HSDS=http://$SERVER_IP:5101/home/test
AEX=http://$SERVER_IP:50391
# h5pyd has no config file here, so the credentials travel with the command.
HS="HS_USERNAME=test HS_PASSWORD=test HS_BUCKET=hsds"
PY=".venv/bin/python -u benchmarks/mdx2/read-procs.py"
ELEMENTS=$((1 << 30))

restore() {
    clear_rtt
    set_buffers default
}
trap restore EXIT

set_buffers tuned
for _ in $(seq "$REPS"); do
    for rtt in "${RTTS[@]}"; do
        set_rtt "$rtt"
        for file in "${FILES[@]}"; do
            case $file in *noisy*) p=none ;; *) p=counting-f32 ;; esac
            tag="rtt+$rtt $file"
            flush_metrics
            for procs in "${PROCS[@]}"; do
                for cache in "${CACHES[@]}"; do
                    run "$tag h5py p$procs $cache" \
                        "$PY $HTTP/$file $ELEMENTS --backend h5py --cache $cache \
                            --procs $procs --reps 1 --label \"$tag h5py p$procs $cache\""
                done
                run "$tag h5pyd p$procs" \
                    "$HS $PY $HSDS/$file $ELEMENTS --backend h5pyd --endpoints $ENDPOINTS \
                        --piece $PIECE --procs $procs --reps 1 --label \"$tag h5pyd p$procs\""
            done
            # AEX2 spends its parallelism on connections rather than processes.
            for s in 1 16; do
                run "$tag aex s$s" \
                    "$PY $AEX/$file $ELEMENTS --backend aex --streams $s --procs 1 --reps 1 \
                        --label \"$tag aex s$s\""
            done
            run "$tag aexbench s16" \
                "target/release/aexbench $AEX $file --bytes \$(($ELEMENTS * 4)) \
                    --streams 16 --reps 1 --prefault --pattern $p \
                    --label \"$tag aexbench s16\""
        done
    done
done
