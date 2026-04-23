#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AUTO_ROUND_DIR="$WORKSPACE_DIR/frameworks.ai.lpot.auto-round"
ARK_DIR="$AUTO_ROUND_DIR/auto_round_extension/ark"

source "$SCRIPT_DIR/activate_env.sh"

if [ ! -d "$AUTO_ROUND_DIR/.git" ]; then
  echo "Source not found, cloning..."
  git clone https://github.com/intel-innersource/frameworks.ai.lpot.auto-round.git "$AUTO_ROUND_DIR"
fi

cd "$AUTO_ROUND_DIR"
OLD_HEAD=$(git rev-parse HEAD)
git pull
NEW_HEAD=$(git rev-parse HEAD)

if [[ "$OLD_HEAD" != "$NEW_HEAD" || ! -d "$ARK_DIR/build" ]]; then
  echo "Code updated ($OLD_HEAD -> $NEW_HEAD) or build dir missing, rebuilding..."

  cd "$ARK_DIR"
  CMAKE_C_COMPILER_LAUNCHER=ccache \
  CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  ONEAPI_VERSION=$(readlink /opt/intel/oneapi/compiler/latest) \
  pip install --no-build-isolation -e .

  cp build/auto_round_kernel_cpu.cpython-312-x86_64-linux-gnu.so ./auto_round_kernel/
  cp xbuild/auto_round_kernel_xpu.cpython-312-x86_64-linux-gnu.so ./auto_round_kernel/
else
  echo "Code is up to date, skipping build."
fi
