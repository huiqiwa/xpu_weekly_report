#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCL_TLA_DIR="$(cd "$SCRIPT_DIR/../sycl-tla" && pwd)"
BUILD_DIR="$SYCL_TLA_DIR/build"

cd "$SYCL_TLA_DIR"
OLD_HEAD=$(git rev-parse HEAD)
git pull
NEW_HEAD=$(git rev-parse HEAD)

if [[ "$OLD_HEAD" != "$NEW_HEAD" || ! -d "$BUILD_DIR" ]]; then
  echo "Code updated ($OLD_HEAD -> $NEW_HEAD) or build dir missing, rebuilding..."
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"

  cmake .. \
    -G Ninja \
    -DCUTLASS_ENABLE_SYCL=ON \
    -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

  ninja -j32 $(ninja -t targets all | grep -oE '^(06_|00_)[^:]+')
else
  echo "Code is up to date, skipping build."
fi
