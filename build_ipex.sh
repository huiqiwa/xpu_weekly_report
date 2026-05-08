#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IPEX_DIR="$WORKSPACE_DIR/intel-extension-for-pytorch"

# Use a separate conda env for IPEX
CONDA_ENV="ipex"
if ! command -v conda &>/dev/null; then
  eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
fi
eval "$(conda shell.bash hook)"
if ! conda env list | grep -E "^${CONDA_ENV}\s" &>/dev/null; then
  echo "Creating conda environment '$CONDA_ENV'..."
  conda create -y -n "$CONDA_ENV" python=3.12 pip
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
git fetch --all
git checkout quant_matmul_v2.10
git submodule sync
git submodule update --init --recursive

# Run import check from a temp dir to avoid importing from source tree
if ! python -c "import os; os.chdir('/tmp'); import intel_extension_for_pytorch" 2>&1; then
  echo "[build_ipex] IPEX import failed, rebuilding..."

  # Clean stale CMake cache to avoid path mismatch (e.g. host vs container)
  find "$IPEX_DIR/build" -name CMakeCache.txt -delete 2>/dev/null

  export CCACHE_BASEDIR="$IPEX_DIR"
  export CCACHE_NOHASHDIR=1
  export CCACHE_COMPILERCHECK=content

  pip install -r requirements.txt

  CMAKE_C_COMPILER_LAUNCHER=ccache \
  CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  MAX_JOBS=64 \
  TORCH_XPU_ARCH_LIST="bmg" \
  BUILD_WITH_CPU=OFF \
  CXXFLAGS="-w" \
  pip install -v . --no-build-isolation 2>&1 | \
  grep -aiE '\[.*%\]|error[: ]|FAILED|fatal[: ]|Building wheel|Successfully|running |creating |copying .*->'
else
  echo "Code is up to date, skipping build."
fi
