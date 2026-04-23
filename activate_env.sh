#!/bin/bash
# Source this file to activate the conda environment:
#   source "$(dirname "${BASH_SOURCE[0]}")/activate_env.sh"

CONDA_ENV="xpu-perf-test"
if ! conda env list | grep -qw "$CONDA_ENV"; then
  echo "Creating conda environment '$CONDA_ENV'..."
  conda create -y -n "$CONDA_ENV" python=3.12
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/latest/env/vars.sh --force
