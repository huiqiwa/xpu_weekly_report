#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
XPU_PERF_DIR="$WORKSPACE_DIR/xpu-perf"

source "$SCRIPT_DIR/activate_env.sh"

if [ ! -d "$XPU_PERF_DIR/.git" ]; then
  git clone https://github.com/abenmao/xpu-perf.git "$XPU_PERF_DIR"
fi
cd "$XPU_PERF_DIR"
git checkout -- .
git clean -fd
git fetch --all
git checkout intel_gpu_backend
git pull

pip install --ignore-installed blinker -r "$XPU_PERF_DIR/micro_perf/requirements.txt"

# Comment out ipex rms_norm provider to avoid work-group size RuntimeError
sed -i 's/@ProviderRegistry.register_vendor_impl("rms_norm", "ipex")/#@ProviderRegistry.register_vendor_impl("rms_norm", "ipex")/' \
    "${XPU_PERF_DIR}/micro_perf/backends/INTEL/ops/ipex/rms_norm.py"


# Adjust xccl batch size
ADJUST_SCRIPT="${XPU_PERF_DIR}/micro_perf/backends/INTEL/ops/xccl/adjust_batch_size.sh"
bash "$ADJUST_SCRIPT" b60

# Build SYCL extensions
bash "$SCRIPT_DIR/build_sycl_ext.sh"