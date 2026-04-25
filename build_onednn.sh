#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ONEDNN_DIR="$WORKSPACE_DIR/oneDNN"

source "$SCRIPT_DIR/activate_env.sh"

if [ ! -d "$ONEDNN_DIR/.git" ]; then
  echo "Source not found, cloning..."
  git clone https://github.com/uxlfoundation/oneDNN.git "$ONEDNN_DIR"
fi

cd "$ONEDNN_DIR"
OLD_HEAD=$(git rev-parse HEAD)
git fetch --all
git checkout main && git pull
NEW_HEAD=$(git rev-parse HEAD)

BUILD_DIR="$ONEDNN_DIR/build"

if [[ "$OLD_HEAD" != "$NEW_HEAD" || ! -f "$BUILD_DIR/src/libdnnl.so" ]]; then
  echo "Code updated ($OLD_HEAD -> $NEW_HEAD) or build missing, rebuilding..."

  export CCACHE_BASEDIR="$ONEDNN_DIR"
  export CCACHE_NOHASHDIR=1
  export CCACHE_COMPILERCHECK=content

  cmake -S "$ONEDNN_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DDNNL_CPU_RUNTIME=OMP \
    -DDNNL_GPU_RUNTIME=SYCL \
    -DDNNL_GPU_VENDOR=INTEL \
    -DDNNL_ENABLE_PRIMITIVE_GPU_ISA=XE2

  cmake --build "$BUILD_DIR" -j64
else
  echo "Code is up to date, skipping build."
fi
