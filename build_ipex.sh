#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IPEX_DIR="$WORKSPACE_DIR/intel-extension-for-pytorch"

# Use a separate conda env for IPEX
CONDA_ENV="ipex"
eval "$(conda shell.bash hook)"
if ! conda env list | grep -qw "$CONDA_ENV"; then
  echo "Creating conda environment '$CONDA_ENV'..."
  conda create -y -n "$CONDA_ENV" python=3.12
fi
conda activate "$CONDA_ENV"

source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/latest/env/vars.sh --force

pip install torch==2.11.0+xpu --extra-index-url https://download.pytorch.org/whl/xpu

if [ ! -d "$IPEX_DIR/.git" ]; then
  echo "Source not found, cloning..."
  git clone https://github.com/abenmao/intel-extension-for-pytorch "$IPEX_DIR"
fi

cd "$IPEX_DIR"
OLD_HEAD=$(git rev-parse HEAD)
git fetch --all
git checkout quant_matmul_v2.10
git submodule sync
git submodule update --init --recursive
NEW_HEAD=$(git rev-parse HEAD)

if [[ "$OLD_HEAD" != "$NEW_HEAD" || ! -d "$IPEX_DIR/build" ]]; then
  echo "Code updated ($OLD_HEAD -> $NEW_HEAD) or build dir missing, rebuilding..."

  pip install -r requirements.txt

  CMAKE_C_COMPILER_LAUNCHER=ccache \
  CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  MAX_JOBS=16 \
  TORCH_XPU_ARCH_LIST="bmg" \
  CXXFLAGS="-w" \
  pip install -v . --no-build-isolation 2>&1 | \
  grep -aiE '\[.*%\]|error[: ]|FAILED|fatal[: ]|Building wheel|Successfully|running |creating |copying .*->'
else
  echo "Code is up to date, skipping build."
fi
