mkdir build && cd build

cmake .. \
  -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

ninja -j32 $(ninja -t targets all | grep -oE '^(06_|00_)[^:]+')
