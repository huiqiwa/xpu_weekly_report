#include "gemm_fp8.h"

void init_cublaslt_fp8(CublasLtContext& ctx, int M, int N, int K) {
    CHECK_CUBLASLT(cublasLtCreate(&ctx.ltHandle));

    // Allocate workspace
    CHECK_CUDA(cudaMalloc(&ctx.workspace, ctx.workspaceSize));

    // Create matmul descriptor: FP8 compute
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&ctx.matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // For row-major C(M,N) = A(M,K) * B(K,N), use column-major trick:
    // C^T(N,M) = B^T(N,K) * A^T(K,M)
    cublasOperation_t opN = CUBLAS_OP_N;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    // Enable fast accumulation for FP8
    int8_t fastAccu = 1;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccu, sizeof(fastAccu)));

    // "A" in cublasLt = our B: row-major (K,N) → col-major (N,K), layout: rows=N, cols=K, ld=N
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ctx.Adesc, CUDA_R_8F_E4M3, N, K, N));
    // "B" in cublasLt = our A: row-major (M,K) → col-major (K,M), layout: rows=K, cols=M, ld=K
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ctx.Bdesc, CUDA_R_8F_E4M3, K, M, K));
    // "C/D" = our C: row-major (M,N) → col-major (N,M), layout: rows=N, cols=M, ld=N
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ctx.Cdesc, CUDA_R_16BF, N, M, N));
}

void run_gemm_fp8(CublasLtContext& ctx, int M, int N, int K,
                  void* A, void* B, void* C,
                  void* scaleA, void* scaleB) {
    float alpha = 1.0f, beta = 0.0f;

    // scaleA applies to cublasLt "A" which is our B matrix
    // scaleB applies to cublasLt "B" which is our A matrix
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleB, sizeof(scaleB)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleA, sizeof(scaleA)));

    // cublasLt "A" = our B, cublasLt "B" = our A (row-major trick: swap order)
    CHECK_CUBLASLT(cublasLtMatmul(ctx.ltHandle, ctx.matmulDesc,
                                   &alpha,
                                   B, ctx.Adesc,
                                   A, ctx.Bdesc,
                                   &beta,
                                   C, ctx.Cdesc,
                                   C, ctx.Cdesc,
                                   nullptr,  // algo (nullptr = default heuristic)
                                   ctx.workspace, ctx.workspaceSize,
                                   0));  // stream
}
