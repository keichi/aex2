#!/bin/bash
#
# Copy the working tree to the evaluation VMs and build it there.
#
# rsync rather than git, so uncommitted changes can be tried on Linux before CI.
# Ignored files never cross: a macOS .so or target/ on a VM breaks the build or
# the import. REVISION records what was copied, so a number traces to a commit.
#
# Usage: benchmarks/mdx2/sync.sh [host...]    (default: aex2-eval1 aex2-eval2)
#   FEATURES=... overrides what is built, for a sweep that needs more than the
#   usual set (the codec features, say, which most sweeps have no use for).

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -eq 0 ] && set -- aex2-eval1 aex2-eval2
FEATURES=${FEATURES:-aex-server/hdf5}
# The extension module has the codec features under its own names, and without
# them the Python client cannot decode what a bounded transfer sends.
PY_FEATURES=$(printf '%s' "$FEATURES" | tr ',' '\n' | sed -n 's|^aex-client/||p' | paste -sd, -)

# Untracked files are copied too, and git describe --dirty would miss them.
rev=$(git -C "$ROOT" rev-parse --short HEAD)
[ -n "$(git -C "$ROOT" status --porcelain)" ] && rev+=-dirty
for h in "$@"; do
    # Not --filter=':- .gitignore': macOS openrsync then deletes .venv.
    rsync -az --delete --exclude=.git --exclude=REVISION \
        --exclude-from="$ROOT/.gitignore" "$ROOT/" "${h}:aex2/"
    echo "$rev" | ssh "$h" 'cat > aex2/REVISION'
done

# A login shell, because .bashrc stops before it puts cargo on PATH.
# The extension is rebuilt too, so Python never imports a stale one.
# The server gets the hdf5 feature by default; the VMs have libhdf5 in
# /usr/local.
#
# LIBCLANG_PATH/BINDGEN_EXTRA_CLANG_ARGS (sz, zfp): the VMs have llvm-18's runtime
# libclang but no clang headers, so bindgen borrows gcc's.
build="cd aex2 &&
    export LIBCLANG_PATH=/usr/lib/llvm-18/lib &&
    export BINDGEN_EXTRA_CLANG_ARGS=\"-isystem \$(ls -d /usr/lib/gcc/*/*/include | tail -1)\" &&
    cargo build --release --features $FEATURES &&
    if [ -d .venv ]; then . .venv/bin/activate &&
        maturin develop --release -q ${PY_FEATURES:+--features $PY_FEATURES}; fi"
pids=()
for h in "$@"; do
    ssh "$h" "bash -lc '$build'" 2>&1 | sed "s/^/[$h] /" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "synced $rev to $* (features: $FEATURES${PY_FEATURES:+, extension: $PY_FEATURES})"
