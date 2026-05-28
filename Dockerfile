FROM intel/oneapi:2025.3.1-0-devel-ubuntu24.04

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
        ccache intel-ocloc python3-dev \
        curl wget git vim

# Install UMD (compute runtime + level-zero)
RUN mkdir neo && \
    cd neo && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.24.8/intel-igc-core-2_2.24.8+20344_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.24.8/intel-igc-opencl-2_2.24.8+20344_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/intel-ocloc_25.48.36300.8-0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/intel-opencl-icd_25.48.36300.8-0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/libigdgmm12_22.8.2_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/libze-intel-gpu1_25.48.36300.8-0_amd64.deb && \
    wget https://github.com/oneapi-src/level-zero/releases/download/v1.26.0/level-zero_1.26.0+u24.04_amd64.deb && \
    dpkg -i *.deb || apt-get install -f -y && \
    cd .. && \
    rm -rf neo

# Install miniforge / conda
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -bc -p /opt/miniforge3 && \
    rm -f Miniforge3-$(uname)-$(uname -m).sh

ENV LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"

SHELL ["/bin/bash", "-l", "-c"]

WORKDIR /workspace

RUN git config --global --add safe.directory '*'

RUN echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc && \
    echo "source /opt/intel/oneapi/ccl/latest/env/vars.sh --force" >> /root/.bashrc

CMD ["/bin/bash", "-i"]