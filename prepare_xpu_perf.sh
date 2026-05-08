#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
XPU_PERF_DIR="$WORKSPACE_DIR/xpu-perf"
CONDA_ENV="${1:-xpu-perf-test}"

if [[ "$CONDA_ENV" == "ipex" ]]; then
  source "$SCRIPT_DIR/activate_ipex_env.sh"
else
  source "$SCRIPT_DIR/activate_env.sh"
fi

if [ ! -d "$XPU_PERF_DIR/.git" ]; then
  git clone https://github.com/abenmao/xpu-perf.git "$XPU_PERF_DIR"
fi
cd "$XPU_PERF_DIR"
git checkout -- .
git clean -fd
git fetch --all
OLD_HEAD=$(git rev-parse HEAD)
git checkout intel_gpu_backend
git pull
NEW_HEAD=$(git rev-parse HEAD)

REQS_FILE="$XPU_PERF_DIR/micro_perf/requirements.txt"
MISSING=$(pip install --dry-run -r "$REQS_FILE" 2>&1 | grep "^Would install" || true)
if [[ "$OLD_HEAD" != "$NEW_HEAD" || -n "$MISSING" ]]; then
  echo "Code updated or dependencies missing: $MISSING. Installing requirements..."
  pip install --ignore-installed blinker -r "$REQS_FILE"
else
  echo "Code and dependencies are up to date, skipping pip install."
fi

# # Comment out ipex rms_norm provider to avoid work-group size RuntimeError
# sed -i 's/@ProviderRegistry.register_vendor_impl("rms_norm", "ipex")/#@ProviderRegistry.register_vendor_impl("rms_norm", "ipex")/' \
#     "${XPU_PERF_DIR}/micro_perf/backends/INTEL/ops/ipex/rms_norm.py"


# # Adjust xccl batch size
# ADJUST_SCRIPT="${XPU_PERF_DIR}/micro_perf/backends/INTEL/ops/xccl/adjust_batch_size.sh"
# bash "$ADJUST_SCRIPT" b60

# Build SYCL extensions
bash "$SCRIPT_DIR/build_sycl_ext.sh"