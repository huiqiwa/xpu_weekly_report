#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

docker run -it --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --privileged \
    -e http_proxy=http://proxy-ir.intel.com:911 \
    -e https_proxy=http://proxy-ir.intel.com:912 \
    -e no_proxy=localhost,127.0.0.1 \
    -e HTTP_PROXY=http://proxy-ir.intel.com:911 \
    -e HTTPS_PROXY=http://proxy-ir.intel.com:912 \
    -e NO_PROXY=localhost,127.0.0.1 \
    -e TZ=Asia/Shanghai \
    -v "$PARENT_DIR":/workspace \
    -w /workspace/xpu_weekly_report \
    --name yupengzh-nv-perf \
    nvcr.io/nvidia/pytorch:26.04-py3 \
    bash -c "pip install flashinfer-python[cu13] flashinfer-cubin && exec bash"
