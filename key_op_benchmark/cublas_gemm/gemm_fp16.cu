#include "gemm_fp16.h"

void run_gemm_fp16(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C) {
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);

    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    CHECK_CUBLAS(cublasHgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha_h,
                             (const __half*)B, N,
                             (const __half*)A, K,
                             &beta_h,
                             (__half*)C, N));
}
