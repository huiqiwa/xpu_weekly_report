#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SYCL_TLA_DIR="$WORKSPACE_DIR/libraries.ai.cutlass.internal"
SYCL_TLA_REPO="https://github.com/intel-innersource/libraries.ai.cutlass.internal.git"
SYCL_TLA_BRANCH="master_next"
BUILD_DIR="$SYCL_TLA_DIR/build"

GITHUB_TOKEN_FILE="$SCRIPT_DIR/.github_token"
if [ -f "$GITHUB_TOKEN_FILE" ]; then
  GITHUB_TOKEN=$(cat "$GITHUB_TOKEN_FILE")
  SYCL_TLA_REPO="https://${GITHUB_TOKEN}@github.com/intel-innersource/libraries.ai.cutlass.internal.git"
fi

# Clone if not present, else fetch and checkout correct branch
if [ ! -d "$SYCL_TLA_DIR/.git" ]; then
  echo "Source not found, cloning..."
  git clone --branch "$SYCL_TLA_BRANCH" "$SYCL_TLA_REPO" "$SYCL_TLA_DIR"
else
  cd "$SYCL_TLA_DIR"
  git remote set-url origin "$SYCL_TLA_REPO"
  git fetch origin
  git checkout "$SYCL_TLA_BRANCH"
  git reset --hard "origin/$SYCL_TLA_BRANCH"
  cd -
fi

cd "$SYCL_TLA_DIR"
OLD_HEAD=$(git rev-parse HEAD)
git pull origin "$SYCL_TLA_BRANCH"
NEW_HEAD=$(git rev-parse HEAD)

if [[ "$OLD_HEAD" != "$NEW_HEAD" || ! -d "$BUILD_DIR" ]]; then
  echo "Code updated ($OLD_HEAD -> $NEW_HEAD) or build dir missing, rebuilding..."
  
  export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file" 
  export IGC_VISAOptions="-perfmodel"
  export IGC_VectorAliasBBThreshold=10000
  export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"

  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"

  export CCACHE_BASEDIR="$SYCL_TLA_DIR"
  export CCACHE_NOHASHDIR=1
  export CCACHE_COMPILERCHECK=content

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
