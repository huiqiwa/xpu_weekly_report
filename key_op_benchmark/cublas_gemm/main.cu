/**
 * cuBLAS GEMM benchmark
 * Usage: ./cublas_gemm [gpu_id] [dtype] [M] [K] [N] [iters] [warmup]
 *   dtype: fp32, tf32, fp16, bf16, int8, fp8, nvfp4
 *
 * Compile:
 *   nvcc -O3 -o cublas_gemm main.cu gemm_fp32.cu gemm_fp16.cu gemm_bf16.cu gemm_int8.cu gemm_fp8.cu gemm_nvfp4.cu -lcublas -lcublasLt
 *
 * Notes:
 *   - fp8 (E4M3) requires SM >= 89 (Ada/Hopper) and CUDA >= 11.8
 *   - nvfp4 requires SM >= 100 (Blackwell) and CUDA >= 12.8
 */

#include "common.h"
#include "gemm_fp8.h"
#include "gemm_nvfp4.h"

// Forward declarations
void run_gemm_fp32(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C, bool use_tf32);
void run_gemm_fp16(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C);
void run_gemm_bf16(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C);
void run_gemm_int8(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C);

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
