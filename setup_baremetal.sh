BANNER="============================================================"
STEP=0
TOTAL=11
step_banner() { STEP=$((STEP + 1)); echo -e "\n$BANNER\n  [$STEP/$TOTAL] $1\n$BANNER"; }

step_banner "Installing system packages"
APT="apt"
if [ "$(id -u)" -ne 0 ]; then APT="sudo apt"; fi
$APT update && $APT install -y --no-install-recommends ccache intel-ocloc 

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step_banner "Activating environment"
source "$SCRIPT_DIR/activate_env.sh"

step_banner "Installing PyTorch"
pip install torch==2.12.0+xpu pyyaml --extra-index-url https://download.pytorch.org/whl/xpu
pip uninstall oneccl oneccl-devel -y

step_banner "Installing numpy"
pip install numpy

step_banner "Installing ninja"
pip install ninja

step_banner "Building vllm-xpu-kernels"
bash "$SCRIPT_DIR/build_vllm_xpu_kernels.sh"

step_banner "Building auto-round"
bash "$SCRIPT_DIR/build_auto_round.sh"

step_banner "Building sycl-tla"
bash "$SCRIPT_DIR/build_sycl_tla.sh"

step_banner "Building oneDNN"
bash "$SCRIPT_DIR/build_onednn.sh"

# step_banner "Building IPEX"
# bash "$SCRIPT_DIR/build_ipex.sh"

step_banner "Building oneCCL and replacing libccl.so"
bash "$SCRIPT_DIR/build_oneccl.sh"

step_banner "Preparing xpu-perf"
bash "$SCRIPT_DIR/prepare_xpu_perf.sh"

