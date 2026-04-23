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
OLD_HEAD=$(git rev-parse HEAD)
git fetch --all
git checkout main && git pull
NEW_HEAD=$(git rev-parse HEAD)

if [[ "$OLD_HEAD" != "$NEW_HEAD" || ! -d "$VLLM_XPU_KERNELS_DIR/build" ]]; then
  echo "Code updated ($OLD_HEAD -> $NEW_HEAD) or build dir missing, rebuilding..."

  sed -i '/^torch==/d' requirements.txt
  pip install -r requirements.txt

  CMAKE_C_COMPILER_LAUNCHER=ccache \
  CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  MAX_JOBS=16 \
  VLLM_XPU_AOT_DEVICES="bmg" \
  VLLM_XPU_XE2_AOT_DEVICES="bmg" \
  pip install --no-build-isolation -v .
else
  echo "Code is up to date, skipping build."
fi
