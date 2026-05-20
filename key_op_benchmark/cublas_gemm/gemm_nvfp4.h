#pragma once
#include "common.h"

// NVFP4 (4-bit float) requires Blackwell (SM100+) and CUDA 12.8+

#if NVFP4_SUPPORTED

struct CublasLtFP4Context {
    cublasLtHandle_t ltHandle;
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    void* workspace;
    size_t workspaceSize;
    void* d_scaleA;
    void* d_scaleB;
    void* d_scaleD;
    void* d_outScaleD;
    void* d_D;
    void* d_C;

    CublasLtFP4Context() : ltHandle(nullptr), matmulDesc(nullptr),
                           Adesc(nullptr), Bdesc(nullptr), Cdesc(nullptr), Ddesc(nullptr),
                           preference(nullptr), heuristicResult{},
                           workspace(nullptr), workspaceSize(32 * 1024 * 1024),
                           d_scaleA(nullptr), d_scaleB(nullptr),
                           d_scaleD(nullptr), d_outScaleD(nullptr),
                           d_D(nullptr), d_C(nullptr) {}

    ~CublasLtFP4Context() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
        if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
        if (ltHandle) cublasLtDestroy(ltHandle);
        if (workspace) cudaFree(workspace);
        if (d_scaleA) cudaFree(d_scaleA);
        if (d_scaleB) cudaFree(d_scaleB);
        if (d_scaleD) cudaFree(d_scaleD);
        if (d_outScaleD) cudaFree(d_outScaleD);
        if (d_D) cudaFree(d_D);
        if (d_C) cudaFree(d_C);
    }
};

void init_cublaslt_nvfp4(CublasLtFP4Context& ctx, int M, int N, int K);
void run_gemm_nvfp4(CublasLtFP4Context& ctx, int M, int N, int K,
                    void* A, void* B, void* C_unused);

#endif // NVFP4_SUPPORTED
