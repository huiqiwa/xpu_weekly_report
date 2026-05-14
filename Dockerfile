FROM intel/oneapi:2026.0.0-devel-ubuntu24.04

ENV http_proxy=http://proxy-ir.intel.com:911 \
    https_proxy=http://proxy-ir.intel.com:912 \
    no_proxy=localhost,127.0.0.1 \
    HTTP_PROXY=http://proxy-ir.intel.com:911 \
    HTTPS_PROXY=http://proxy-ir.intel.com:912 \
    NO_PROXY=localhost,127.0.0.1 \
    PATH="/opt/intel/oneapi/compiler/latest/bin/compiler:${PATH}" \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/xpu \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache

RUN apt update && \
    apt install -y --no-install-recommends \
        python3-pip python-is-python3 ninja-build \
        ccache intel-ocloc python3-dev

RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -bc -p /opt/miniforge3 && \
    rm -f Miniforge3-$(uname)-$(uname -m).sh

SHELL ["/bin/bash", "-l", "-c"]

WORKDIR /workspace

RUN git config --global --add safe.directory '*'
