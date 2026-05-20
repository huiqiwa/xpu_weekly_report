#include "gemm_fp32.h"

void run_gemm_fp32(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C, bool use_tf32) {
    float alpha = 1.0f, beta = 0.0f;

    if (use_tf32) {
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    } else {
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    }

    // Column-major: C(N,M) = B(N,K) * A(K,M)
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             (const float*)B, N,
                             (const float*)A, K,
                             &beta,
                             (float*)C, N));
}
