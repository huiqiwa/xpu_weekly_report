#include "gemm_int8.h"

void run_gemm_int8(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C) {
    // INT8 GEMM: A(int8) * B(int8) = C(int32)
    int32_t alpha = 1, beta = 0;

    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              B, CUDA_R_8I, N,
                              A, CUDA_R_8I, K,
                              &beta,
                              C, CUDA_R_32I, N,
                              CUBLAS_COMPUTE_32I,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
