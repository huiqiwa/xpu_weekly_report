#include "gemm_nvfp4.h"

#if NVFP4_SUPPORTED

void init_cublaslt_nvfp4(CublasLtFP4Context& ctx, int M, int N, int K) {
    CHECK_CUBLASLT(cublasLtCreate(&ctx.ltHandle));
    CHECK_CUDA(cudaMalloc(&ctx.workspace, ctx.workspaceSize));

    const int block_size = 16;

    // Create matmul descriptor
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&ctx.matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t opT = CUBLAS_OP_T;
    cublasOperation_t opN = CUBLAS_OP_N;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    // Set block scaling modes (required for NVFP4)
    cublasLtMatmulMatrixScale_t blockScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t scalarScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &blockScaleMode, sizeof(blockScaleMode)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &blockScaleMode, sizeof(blockScaleMode)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scalarScaleMode, sizeof(scalarScaleMode)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &blockScaleMode, sizeof(blockScaleMode)));

    // Matrix layouts (column-major): D(m,n) = op_T(A)(m,k) * op_N(B)(k,n)
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ctx.Adesc, CUDA_R_4F_E2M1, K, M, K));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ctx.Bdesc, CUDA_R_4F_E2M1, K, N, K));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ctx.Cdesc, CUDA_R_16BF, M, N, M));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&ctx.Ddesc, CUDA_R_4F_E2M1, M, N, M));

    // Allocate A/B block scale factors (UE4M3, one per block_size elements)
    size_t scaleA_size = ((size_t)M * K + block_size - 1) / block_size;
    size_t scaleB_size = ((size_t)N * K + block_size - 1) / block_size;
    CHECK_CUDA(cudaMalloc(&ctx.d_scaleA, scaleA_size));
    CHECK_CUDA(cudaMalloc(&ctx.d_scaleB, scaleB_size));
    CHECK_CUDA(cudaMemset(ctx.d_scaleA, 0x38, scaleA_size));  // ~1.0 in UE4M3
    CHECK_CUDA(cudaMemset(ctx.d_scaleB, 0x38, scaleB_size));

    // Allocate D scale (scalar float) and D output block scale (UE4M3)
    CHECK_CUDA(cudaMalloc(&ctx.d_scaleD, sizeof(float)));
    float one = 1.0f;
    CHECK_CUDA(cudaMemcpy(ctx.d_scaleD, &one, sizeof(float), cudaMemcpyHostToDevice));

    size_t outScaleD_size = ((size_t)M * N + block_size - 1) / block_size;
    CHECK_CUDA(cudaMalloc(&ctx.d_outScaleD, outScaleD_size));
    CHECK_CUDA(cudaMemset(ctx.d_outScaleD, 0x38, outScaleD_size));

    // Allocate D output buffer (FP4, 2 elements per byte)
    size_t size_D = ((size_t)M * N + 1) / 2;
    CHECK_CUDA(cudaMalloc(&ctx.d_D, size_D));
    CHECK_CUDA(cudaMemset(ctx.d_D, 0, size_D));

    // Allocate C buffer (BF16, for beta * C term)
    size_t size_C = (size_t)M * N * sizeof(__nv_bfloat16);
    CHECK_CUDA(cudaMalloc(&ctx.d_C, size_C));
    CHECK_CUDA(cudaMemset(ctx.d_C, 0, size_C));

    // Set scale pointers
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ctx.d_scaleA, sizeof(ctx.d_scaleA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &ctx.d_scaleB, sizeof(ctx.d_scaleB)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &ctx.d_scaleD, sizeof(ctx.d_scaleD)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(ctx.matmulDesc,
        CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &ctx.d_outScaleD, sizeof(ctx.d_outScaleD)));

    // Create preference and find best algorithm via heuristic
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&ctx.preference));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(ctx.preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ctx.workspaceSize, sizeof(ctx.workspaceSize)));

    int returnedResults = 0;
    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(ctx.ltHandle, ctx.matmulDesc,
        ctx.Adesc, ctx.Bdesc, ctx.Cdesc, ctx.Ddesc,
        ctx.preference, 1, &ctx.heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        fprintf(stderr, "Error: no suitable NVFP4 algorithm found\n");
        exit(EXIT_FAILURE);
    }
}

void run_gemm_nvfp4(CublasLtFP4Context& ctx, int M, int N, int K,
                    void* A, void* B, void* /*C_unused*/) {
    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUBLASLT(cublasLtMatmul(ctx.ltHandle, ctx.matmulDesc,
                                   &alpha,
                                   A, ctx.Adesc,
                                   B, ctx.Bdesc,
                                   &beta,
                                   ctx.d_C, ctx.Cdesc,
                                   ctx.d_D, ctx.Ddesc,
                                   &ctx.heuristicResult.algo,
                                   ctx.workspace, ctx.workspaceSize,
                                   0));
}

#endif // NVFP4_SUPPORTED
