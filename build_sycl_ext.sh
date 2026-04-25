#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCL_EXT_DIR="$SCRIPT_DIR/../xpu-perf/micro_perf/backends/INTEL/ops/sycl_ext"

source "$SCRIPT_DIR/activate_env.sh"

# Check if rebuild is needed: no .so files or any .cpp newer than oldest .so
NEED_BUILD=0
OLDEST_SO=$(find "$SYCL_EXT_DIR" -maxdepth 1 -name '*.so' -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2)
if [[ -z "$OLDEST_SO" ]]; then
  NEED_BUILD=1
elif [[ -n $(find "$SYCL_EXT_DIR" -maxdepth 1 -name '*.cpp' -newer "$OLDEST_SO" 2>/dev/null) ]]; then
  NEED_BUILD=1
fi

if [[ "$NEED_BUILD" -eq 0 ]]; then
  echo "SYCL KV cache extensions are up to date, skipping."
  exit 0
fi

# Patch build.sh to use ccache, build, then restore
export CCACHE_BASEDIR="$SYCL_EXT_DIR"
export CCACHE_NOHASHDIR=1
export CCACHE_COMPILERCHECK=content
cd "$SYCL_EXT_DIR"
sed -i 's/^icpx /ccache icpx /' build.sh
bash build.sh
sed -i 's/^ccache icpx /icpx /' build.sh
