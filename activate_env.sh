#!/bin/bash
# Source this file to activate the xpu-perf-test conda environment:
#   source "$(dirname "${BASH_SOURCE[0]}")/activate_env.sh"

# Ensure conda is on PATH (non-interactive shells don't load .bashrc)
if ! command -v conda &>/dev/null; then
  if [ -f /opt/miniforge3/bin/conda ]; then
    eval "$(/opt/miniforge3/bin/conda shell.bash hook)"
  elif [ -f "$HOME/miniforge3/bin/conda" ]; then
    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
  fi
fi

CONDA_ENV="xpu-perf-test"
if ! conda env list | grep -E "^${CONDA_ENV}\s" &>/dev/null; then
  echo "Creating conda environment '$CONDA_ENV'..."
  conda create -y -n "$CONDA_ENV" python=3.12 pip
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/latest/env/vars.sh --force

export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
