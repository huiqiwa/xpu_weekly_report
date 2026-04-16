# syntax=docker/dockerfile:1
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
        ccache intel-ocloc python3-dev && \
    pip3 install torch==2.11.0+xpu pyyaml

WORKDIR /workspace

SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/git-src \
    source /opt/intel/oneapi/setvars.sh --force && \
    source /opt/intel/oneapi/ccl/latest/env/vars.sh --force && \
    if [ ! -d /root/.cache/git-src/vllm-xpu-kernels/.git ]; then \
        git clone https://github.com/vllm-project/vllm-xpu-kernels.git /root/.cache/git-src/vllm-xpu-kernels; \
    fi && \
    cd /root/.cache/git-src/vllm-xpu-kernels && \
    git fetch --all && \
    git checkout "$(git branch -r --sort=-version:refname --list 'origin/release/*' | head -1 | xargs)" && \
    cd /workspace && \
    cp -a /root/.cache/git-src/vllm-xpu-kernels . && \
    cd vllm-xpu-kernels && \
    sed -i '/^torch==/d' requirements.txt && \
    pip install -r requirements.txt && \
    MAX_JOBS=64 \
    VLLM_XPU_AOT_DEVICES="bmg" \
    VLLM_XPU_XE2_AOT_DEVICES="bmg" \
    pip install --no-build-isolation -v -e .

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/git-src \
    --mount=type=secret,id=github_token \
    source /opt/intel/oneapi/setvars.sh --force && \
    if [ ! -d /root/.cache/git-src/auto-round/.git ]; then \
        git clone https://$(cat /run/secrets/github_token)@github.com/intel-innersource/frameworks.ai.lpot.auto-round.git /root/.cache/git-src/auto-round; \
    fi && \
    cd /root/.cache/git-src/auto-round && \
    git remote set-url origin https://$(cat /run/secrets/github_token)@github.com/intel-innersource/frameworks.ai.lpot.auto-round.git && \
    git fetch --all && git pull && \
    cd /workspace && \
    cp -a /root/.cache/git-src/auto-round . && \
    cd auto-round/auto_round_extension/ark && \
    ONEAPI_VERSION=$(readlink /opt/intel/oneapi/compiler/latest) \
    pip install -e . && \
    cp build/auto_round_kernel_cpu.cpython-312-x86_64-linux-gnu.so ./auto_round_kernel/ && \
    cp xbuild/auto_round_kernel_xpu.cpython-312-x86_64-linux-gnu.so ./auto_round_kernel/

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/git-src \
    source /opt/intel/oneapi/setvars.sh --force && \
    if [ ! -d /root/.cache/git-src/sycl-tla/.git ]; then \
        git clone https://github.com/intel/sycl-tla.git /root/.cache/git-src/sycl-tla; \
    fi && \
    cd /root/.cache/git-src/sycl-tla && \
    git fetch --all && git pull && \
    cd /workspace && \
    cp -a /root/.cache/git-src/sycl-tla . && \
    cd sycl-tla && \
    mkdir build && cd build && \
    cmake .. \
      -G Ninja \
      -DCUTLASS_ENABLE_SYCL=ON \
      -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_C_COMPILER=icx && \
    ninja -j32 $(ninja -t targets all | grep -oE '^(06_|00_)[^:]+')

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/git-src \
    source /opt/intel/oneapi/setvars.sh --force && \
    source /opt/intel/oneapi/ccl/latest/env/vars.sh --force && \
    if [ ! -d /root/.cache/git-src/intel-extension-for-pytorch/.git ]; then \
        git clone https://github.com/abenmao/intel-extension-for-pytorch /root/.cache/git-src/intel-extension-for-pytorch; \
    fi && \
    cd /root/.cache/git-src/intel-extension-for-pytorch && \
    git fetch --all && \
    git checkout quant_matmul_v2.10 && \
    git submodule sync && \
    git submodule update --init --recursive && \
    cd /workspace && \
    cp -a /root/.cache/git-src/intel-extension-for-pytorch . && \
    cd intel-extension-for-pytorch && \
    MAX_JOBS=64 \
    TORCH_XPU_ARCH_LIST="bmg" \
    CXXFLAGS="-w" \
    pip install -v . --no-build-isolation 2>&1 | \
    grep -aiE '\[.*%\]|error[: ]|FAILED|fatal[: ]|Building wheel|Successfully|running |creating |copying .*->'


RUN --mount=type=cache,target=/root/.cache/git-src \
    if [ ! -d /root/.cache/git-src/xpu-perf/.git ]; then \
        git clone https://github.com/abenmao/xpu-perf.git /root/.cache/git-src/xpu-perf; \
    fi && \
    cd /root/.cache/git-src/xpu-perf && \
    git fetch --all && git checkout intel_gpu_backend && git pull && \
    cd /workspace && \
    cp -a /root/.cache/git-src/xpu-perf . && \
    pip install --ignore-installed blinker -r xpu-perf/micro_perf/requirements.txt
  

