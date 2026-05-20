#pragma once
#include "common.h"

// FP8 E4M3 requires cublasLt (not supported by legacy cublas API)

struct CublasLtContext {
    cublasLtHandle_t ltHandle;
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    void* workspace;
    size_t workspaceSize;

    CublasLtContext() : ltHandle(nullptr), matmulDesc(nullptr),
                        Adesc(nullptr), Bdesc(nullptr), Cdesc(nullptr),
                        workspace(nullptr), workspaceSize(32 * 1024 * 1024) {}

    ~CublasLtContext() {
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
        if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
        if (ltHandle) cublasLtDestroy(ltHandle);
        if (workspace) cudaFree(workspace);
    }
};

void init_cublaslt_fp8(CublasLtContext& ctx, int M, int N, int K);
void run_gemm_fp8(CublasLtContext& ctx, int M, int N, int K,
                  void* A, void* B, void* C,
                  void* scaleA, void* scaleB);
