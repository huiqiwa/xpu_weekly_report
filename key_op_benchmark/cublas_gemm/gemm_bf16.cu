#include "gemm_bf16.h"

void run_gemm_bf16(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C) {
    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Use cublasGemmEx for BF16
    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              B, CUDA_R_16BF, N,
                              A, CUDA_R_16BF, K,
                              &beta,
                              C, CUDA_R_16BF, N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
