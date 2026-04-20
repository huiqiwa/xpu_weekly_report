#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCL_EXT_DIR="$SCRIPT_DIR/../xpu-perf/micro_perf/backends/INTEL/ops/sycl_ext"

# Check if rebuild is needed: .so missing or source files newer than .so
NEED_BUILD=0
if [[ ! -f "$SYCL_EXT_DIR/store_kv_cache_sycl.so" || ! -f "$SYCL_EXT_DIR/dequant_kv_cache_sycl.so" ]]; then
  NEED_BUILD=1
elif [[ -n $(find "$SYCL_EXT_DIR" -name '*.cpp' -newer "$SYCL_EXT_DIR/store_kv_cache_sycl.so" 2>/dev/null) ]]; then
  NEED_BUILD=1
fi

if [[ "$NEED_BUILD" -eq 0 ]]; then
  echo "SYCL KV cache extensions are up to date, skipping."
  exit 0
fi

# Get torch compile paths (must run BEFORE sourcing oneAPI)
export TORCH_INCLUDES=$(python3 -c "
import torch.utils.cpp_extension as ext
for p in ext.include_paths():
    print(f'-I{p}', end=' ')
")
export TORCH_LIBS=$(python3 -c "
import torch.utils.cpp_extension as ext
for p in ext.library_paths():
    print(f'-L{p}', end=' ')
")

# Build
cd "$SYCL_EXT_DIR"
bash build.sh
