BANNER="============================================================"
step_banner() { echo -e "\n$BANNER\n  [$1] $2\n$BANNER"; }

step_banner "1/10" "Installing system packages"
APT="apt"
if [ "$(id -u)" -ne 0 ]; then APT="sudo apt"; fi
$APT update && $APT install -y --no-install-recommends ccache intel-ocloc 

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step_banner "2/10" "Activating environment"
source "$SCRIPT_DIR/activate_env.sh"

step_banner "3/10" "Installing PyTorch"
pip install torch==2.11.0+xpu pyyaml --extra-index-url https://download.pytorch.org/whl/xpu

step_banner "4/10" "Installing numpy"
pip install numpy

step_banner "5/10" "Preparing xpu-perf"
bash "$SCRIPT_DIR/prepare_xpu_perf.sh"

step_banner "6/10" "Building vllm-xpu-kernels"
bash "$SCRIPT_DIR/build_vllm_xpu_kernels.sh"

step_banner "7/10" "Building auto-round"
bash "$SCRIPT_DIR/build_auto_round.sh"

step_banner "8/10" "Building sycl-tla"
bash "$SCRIPT_DIR/build_sycl_tla.sh"

step_banner "9/10" "Building oneDNN"
bash "$SCRIPT_DIR/build_onednn.sh"

step_banner "10/10" "Building IPEX"
bash "$SCRIPT_DIR/build_ipex.sh"

