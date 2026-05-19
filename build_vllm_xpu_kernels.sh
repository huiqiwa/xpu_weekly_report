#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_XPU_KERNELS_DIR="$WORKSPACE_DIR/vllm-xpu-kernels"

source "$SCRIPT_DIR/activate_env.sh"

if [ ! -d "$VLLM_XPU_KERNELS_DIR/.git" ]; then
  echo "Source not found, cloning..."
  git clone https://github.com/vllm-project/vllm-xpu-kernels.git "$VLLM_XPU_KERNELS_DIR"
fi

cd "$VLLM_XPU_KERNELS_DIR"
git checkout -- .
OLD_HEAD=$(git rev-parse HEAD)
git fetch --all
git checkout main && git pull
NEW_HEAD=$(git rev-parse HEAD)

if [[ "$OLD_HEAD" != "$NEW_HEAD" ]] || ! python -c "import os; os.chdir('/tmp'); import vllm_xpu_kernels" 2>/dev/null; then
  echo "Code updated ($OLD_HEAD -> $NEW_HEAD) or build dir missing, rebuilding..."

  rm -rf .deps build vllm_xpu_kernels.egg-info _version.py

  sed -i '/^torch==/d' requirements.txt
  pip install -r requirements.txt

  export CCACHE_BASEDIR="$VLLM_XPU_KERNELS_DIR"
  export CCACHE_NOHASHDIR=1
  export CCACHE_COMPILERCHECK=content

  CMAKE_C_COMPILER_LAUNCHER=ccache \
  CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  MAX_JOBS=16 \
  VLLM_XPU_AOT_DEVICES="bmg" \
  VLLM_XPU_XE2_AOT_DEVICES="bmg" \
  pip install --no-build-isolation -v .
else
  echo "Code is up to date, skipping build."
fi
