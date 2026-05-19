/**
 * cuBLAS GEMM benchmark
 * Usage: ./cublas_gemm [gpu_id] [dtype] [M] [K] [N] [iters] [warmup]
 *   dtype: fp32, tf32, fp16, bf16, int8, fp8, nvfp4
 *
 * Compile:
 *   nvcc -O3 -o cublas_gemm cublas_gemm.cu -lcublas -lcublasLt
 *
 * Notes:
 *   - fp8 (E4M3) requires SM >= 89 (Ada/Hopper) and CUDA >= 11.8
 *   - nvfp4 requires SM >= 100 (Blackwell) and CUDA >= 12.8
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define CHECK_CUBLAS(call)                                                 \
    do {                                                                    \
        cublasStatus_t status = (call);                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,       \
                    __LINE__, (int)status);                                 \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define CHECK_CUBLASLT(call)                                               \
    do {                                                                    \
        cublasStatus_t status = (call);                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            fprintf(stderr, "cuBLASLt error at %s:%d: %d\n", __FILE__,     \
                    __LINE__, (int)status);                                 \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

enum DType { FP32, TF32, FP16, BF16, INT8, FP8, NVFP4 };

// NVFP4 requires CUDA 12.8+
#define NVFP4_SUPPORTED (CUDART_VERSION >= 12080)

struct Config {
    int gpu_id = 0;
    DType dtype = BF16;
    int M = 8192;
    int K = 8192;
    int N = 8192;
    int iters = 10;
    int warmup = 2;
};

DType parse_dtype(const char* s) {
    if (strcmp(s, "fp32") == 0 || strcmp(s, "float32") == 0) return FP32;
    if (strcmp(s, "tf32") == 0 || strcmp(s, "tfloat32") == 0) return TF32;
    if (strcmp(s, "fp16") == 0 || strcmp(s, "float16") == 0) return FP16;
    if (strcmp(s, "bf16") == 0 || strcmp(s, "bfloat16") == 0) return BF16;
    if (strcmp(s, "int8") == 0) return INT8;
    if (strcmp(s, "fp8") == 0 || strcmp(s, "fp8_e4m3") == 0) return FP8;
    if (strcmp(s, "nvfp4") == 0 || strcmp(s, "fp4") == 0) {
#if NVFP4_SUPPORTED
        return NVFP4;
#else
        fprintf(stderr, "Error: nvfp4 requires CUDA >= 12.8 (current: %d)\n", CUDART_VERSION);
        exit(EXIT_FAILURE);
#endif
    }
    fprintf(stderr, "Unsupported dtype: %s (supported: fp32, tf32, fp16, bf16, int8, fp8, nvfp4)\n", s);
    exit(EXIT_FAILURE);
}

const char* dtype_name(DType d) {
    switch (d) {
        case FP32: return "fp32";
        case TF32: return "tf32";
        case FP16: return "fp16";
        case BF16: return "bf16";
        case INT8: return "int8";
        case FP8: return "fp8_e4m3";
        case NVFP4: return "nvfp4";
        default: return "unknown";
    }
}

void run_gemm_fp32(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C, bool use_tf32) {
    // cuBLAS uses column-major, so we compute C^T = B^T * A^T
    // which gives us row-major C = A * B
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

void run_gemm_int8(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C) {
    // INT8 GEMM: A(int8) * B(int8) = C(int32)
    // Using cublasGemmEx with CUDA_R_8I inputs and CUDA_R_32I output
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

// ============ FP8 GEMM using cublasLt ============
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

void init_cublaslt_fp8(CublasLtContext& ctx, int M, int N, int K) {
    CHECK_CUBLASLT(cublasLtCreate(&ctx.ltHandle));

    // Allocate workspace
    CHECK_CUDA(cudaMalloc(&ctx.workspace, ctx.workspaceSize));

    // Create matmul descriptor: FP8 compute
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&ctx.matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // For row-major C(M,N) = A(M,K) * B(K,N), use column-major trick:
    // C^T(N,M) = B^T(N,K) * A^T(K,M)
    // cublasLt: D(m,n) = op(first)(m,k) * op(second)(k,n)
    // m=N, n=M, k=K, transa=N, transb=N
    // first matrix = B data (row-major K×N = col-major N×K → layout N,K,ld=N)
    // second matrix = A data (row-major M×K = col-major K×M → layout K,M,ld=K)
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

// ============ NVFP4 GEMM using cublasLt ============
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
    // Scale factors for block-scaled FP4
    void* d_scaleA;       // UE4M3 block scale, one per 16 elements of A
    void* d_scaleB;       // UE4M3 block scale, one per 16 elements of B
    void* d_scaleD;       // float scalar scale for D input
    void* d_outScaleD;    // UE4M3 block scale for D output, one per 16 elements
    void* d_D;            // FP4 output buffer
    void* d_C;            // BF16 bias buffer (for C, used with beta)

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

void init_cublaslt_nvfp4(CublasLtFP4Context& ctx, int M, int N, int K) {
    CHECK_CUBLASLT(cublasLtCreate(&ctx.ltHandle));
    CHECK_CUDA(cudaMalloc(&ctx.workspace, ctx.workspaceSize));

    // NVFP4: uses block-scaling with block size 16
    // Each element is 4-bit, packed 2 per byte
    // Scale factors are UE4M3 (unsigned FP8), one per 16 elements
    const int block_size = 16;

    // Create matmul descriptor
    // D(fp4) = alpha * op_T(A(fp4)) * op_N(B(fp4)) + beta * C(bf16)
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
    // m=M, n=N, k=K
    // A storage: (k, m) = (K, M), ld=K (transa=T)
    // B storage: (k, n) = (K, N), ld=K (transb=N)
    // C: (m, n) = (M, N), ld=M  [BF16]
    // D: (m, n) = (M, N), ld=M  [FP4]
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

int main(int argc, char** argv) {
    Config cfg;

    if (argc > 1) cfg.gpu_id = atoi(argv[1]);
    if (argc > 2) cfg.dtype = parse_dtype(argv[2]);
    if (argc > 3) cfg.M = atoi(argv[3]);
    if (argc > 4) cfg.K = atoi(argv[4]);
    if (argc > 5) cfg.N = atoi(argv[5]);
    if (argc > 6) cfg.iters = atoi(argv[6]);
    if (argc > 7) cfg.warmup = atoi(argv[7]);

    CHECK_CUDA(cudaSetDevice(cfg.gpu_id));

    // Print GPU info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, cfg.gpu_id));
    printf("GPU %d: %s\n", cfg.gpu_id, prop.name);
    printf("GEMM: [%d x %d] x [%d x %d], dtype=%s, iters=%d, warmup=%d\n\n",
           cfg.M, cfg.K, cfg.K, cfg.N, dtype_name(cfg.dtype), cfg.iters, cfg.warmup);

    // Determine element size and allocation sizes
    size_t size_A, size_B, size_C;

    switch (cfg.dtype) {
        case FP32: case TF32:
            size_A = (size_t)cfg.M * cfg.K * sizeof(float);
            size_B = (size_t)cfg.K * cfg.N * sizeof(float);
            size_C = (size_t)cfg.M * cfg.N * sizeof(float);
            break;
        case FP16:
            size_A = (size_t)cfg.M * cfg.K * sizeof(__half);
            size_B = (size_t)cfg.K * cfg.N * sizeof(__half);
            size_C = (size_t)cfg.M * cfg.N * sizeof(__half);
            break;
        case BF16:
            size_A = (size_t)cfg.M * cfg.K * sizeof(__nv_bfloat16);
            size_B = (size_t)cfg.K * cfg.N * sizeof(__nv_bfloat16);
            size_C = (size_t)cfg.M * cfg.N * sizeof(__nv_bfloat16);
            break;
        case INT8:
            size_A = (size_t)cfg.M * cfg.K * sizeof(int8_t);
            size_B = (size_t)cfg.K * cfg.N * sizeof(int8_t);
            size_C = (size_t)cfg.M * cfg.N * sizeof(int32_t);
            break;
        case FP8:
            size_A = (size_t)cfg.M * cfg.K;  // 1 byte per element
            size_B = (size_t)cfg.K * cfg.N;
            size_C = (size_t)cfg.M * cfg.N * sizeof(__nv_bfloat16);  // output BF16
            break;
        case NVFP4:
#if NVFP4_SUPPORTED
            // 4-bit: 2 elements per byte
            size_A = ((size_t)cfg.M * cfg.K + 1) / 2;
            size_B = ((size_t)cfg.N * cfg.K + 1) / 2;
            size_C = (size_t)cfg.M * cfg.N * sizeof(__nv_bfloat16);  // output BF16
#endif
            break;
    }

    // Allocate device memory
    void *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    // Initialize with random data (just fill with some pattern)
    // For benchmarking, content doesn't matter
    CHECK_CUDA(cudaMemset(d_A, 1, size_A));
    CHECK_CUDA(cudaMemset(d_B, 1, size_B));
    CHECK_CUDA(cudaMemset(d_C, 0, size_C));

    // FP8 scale factors (per-tensor, on device)
    void *d_scaleA = nullptr, *d_scaleB = nullptr;
    if (cfg.dtype == FP8) {
        CHECK_CUDA(cudaMalloc(&d_scaleA, sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_scaleB, sizeof(float)));
        float one = 1.0f;
        CHECK_CUDA(cudaMemcpy(d_scaleA, &one, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_scaleB, &one, sizeof(float), cudaMemcpyHostToDevice));
    }

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Initialize cublasLt contexts if needed
    CublasLtContext fp8Ctx;
#if NVFP4_SUPPORTED
    CublasLtFP4Context fp4Ctx;
#endif
    if (cfg.dtype == FP8) {
        init_cublaslt_fp8(fp8Ctx, cfg.M, cfg.N, cfg.K);
    }
#if NVFP4_SUPPORTED
    else if (cfg.dtype == NVFP4) {
        init_cublaslt_nvfp4(fp4Ctx, cfg.M, cfg.N, cfg.K);
    }
#endif

    // Lambda to dispatch GEMM
    auto dispatch_gemm = [&]() {
        switch (cfg.dtype) {
            case FP32: run_gemm_fp32(handle, cfg.M, cfg.N, cfg.K, d_A, d_B, d_C, false); break;
            case TF32: run_gemm_fp32(handle, cfg.M, cfg.N, cfg.K, d_A, d_B, d_C, true); break;
            case FP16: run_gemm_fp16(handle, cfg.M, cfg.N, cfg.K, d_A, d_B, d_C); break;
            case BF16: run_gemm_bf16(handle, cfg.M, cfg.N, cfg.K, d_A, d_B, d_C); break;
            case INT8: run_gemm_int8(handle, cfg.M, cfg.N, cfg.K, d_A, d_B, d_C); break;
            case FP8: run_gemm_fp8(fp8Ctx, cfg.M, cfg.N, cfg.K, d_A, d_B, d_C, d_scaleA, d_scaleB); break;
#if NVFP4_SUPPORTED
            case NVFP4: run_gemm_nvfp4(fp4Ctx, cfg.M, cfg.N, cfg.K, d_A, d_B, d_C); break;
#endif
            default: break;
        }
    };

    // Warmup
    printf("Warmup (%d iters)...\n", cfg.warmup);
    for (int i = 0; i < cfg.warmup; i++) {
        dispatch_gemm();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark with CUDA events
    printf("Benchmarking (%d iters)...\n", cfg.iters);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < cfg.iters; i++) {
        dispatch_gemm();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / cfg.iters;

    // FLOPS: 2 * M * N * K per GEMM
    double flops_per_op = 2.0 * cfg.M * cfg.N * cfg.K;
    double tflops = (flops_per_op * cfg.iters) / (total_ms / 1000.0) / 1e12;

    printf("\n==================================================\n");
    printf("  cuBLAS GEMM Results: [%d x %d] x [%d x %d] %s\n",
           cfg.M, cfg.K, cfg.K, cfg.N, dtype_name(cfg.dtype));
    printf("==================================================\n");
    printf("  Avg latency:  %.3f ms\n", avg_ms);
    printf("  Avg TFLOPS:   %.2f\n", tflops);
    printf("  FLOPS/op:     %.2e\n", flops_per_op);
    printf("==================================================\n");

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    if (d_scaleA) CHECK_CUDA(cudaFree(d_scaleA));
    if (d_scaleB) CHECK_CUDA(cudaFree(d_scaleB));

    return 0;
}
