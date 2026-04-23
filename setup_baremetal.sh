sudo apt update && sudo apt install -y --no-install-recommends ccache intel-ocloc 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_env.sh"

pip install torch==2.11.0+xpu pyyaml --extra-index-url https://download.pytorch.org/whl/xpu

pip install numpy

# Prepare xpu-perf
bash "$SCRIPT_DIR/prepare_xpu_perf.sh"

# Build all components
bash "$SCRIPT_DIR/build_vllm_xpu_kernels.sh"
bash "$SCRIPT_DIR/build_auto_round.sh"
bash "$SCRIPT_DIR/build_sycl_tla.sh"
bash "$SCRIPT_DIR/build_ipex.sh"

