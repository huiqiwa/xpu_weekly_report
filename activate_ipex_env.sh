#!/bin/bash
# Source this file to activate the ipex conda environment:
#   source "$(dirname "${BASH_SOURCE[0]}")/activate_ipex_env.sh"

# Ensure conda is on PATH (non-interactive shells don't load .bashrc)
if ! command -v conda &>/dev/null; then
  eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
fi

CONDA_ENV="ipex"
if ! conda env list | grep -qw "$CONDA_ENV"; then
  echo "Creating conda environment '$CONDA_ENV'..."
  conda create -y -n "$CONDA_ENV" python=3.12
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/latest/env/vars.sh --force
