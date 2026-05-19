/**
 * cuBLAS BF16 GEMM benchmark
 * Usage: ./gemm_bf16 [gpu_id] [M] [K] [N] [iters] [warmup]
 */
#include "common.h"

void run_gemm_bf16(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C);

int main(int argc, char** argv) {
    Config cfg;
    cfg.dtype = BF16;

    if (argc > 1) cfg.gpu_id = atoi(argv[1]);
    if (argc > 2) cfg.M = atoi(argv[2]);
    if (argc > 3) cfg.K = atoi(argv[3]);
    if (argc > 4) cfg.N = atoi(argv[4]);
    if (argc > 5) cfg.iters = atoi(argv[5]);
    if (argc > 6) cfg.warmup = atoi(argv[6]);

    CHECK_CUDA(cudaSetDevice(cfg.gpu_id));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, cfg.gpu_id));
    printf("GPU %d: %s\n", cfg.gpu_id, prop.name);
    printf("GEMM: [%d x %d] x [%d x %d], dtype=%s, iters=%d, warmup=%d\n\n",
           cfg.M, cfg.K, cfg.K, cfg.N, dtype_name(cfg.dtype), cfg.iters, cfg.warmup);

    size_t size_A = (size_t)cfg.M * cfg.K * sizeof(__nv_bfloat16);
    size_t size_B = (size_t)cfg.K * cfg.N * sizeof(__nv_bfloat16);
    size_t size_C = (size_t)cfg.M * cfg.N * sizeof(__nv_bfloat16);

    void *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    CHECK_CUDA(cudaMemset(d_A, 1, size_A));
    CHECK_CUDA(cudaMemset(d_B, 1, size_B));
    CHECK_CUDA(cudaMemset(d_C, 0, size_C));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    for (int i = 0; i < cfg.warmup; i++)
        run_gemm_bf16(handle, cfg.M, cfg.N, cfg.K, d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < cfg.iters; i++)
        run_gemm_bf16(handle, cfg.M, cfg.N, cfg.K, d_A, d_B, d_C);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / cfg.iters;
    double flops_per_op = 2.0 * cfg.M * cfg.N * cfg.K;
    double tflops = (flops_per_op * cfg.iters) / (total_ms / 1000.0) / 1e12;

    printf("==================================================\n");
    printf("  cuBLAS GEMM Results: [%d x %d] x [%d x %d] %s\n",
           cfg.M, cfg.K, cfg.K, cfg.N, dtype_name(cfg.dtype));
    printf("==================================================\n");
    printf("  Avg latency:  %.3f ms\n", avg_ms);
    printf("  Avg TFLOPS:   %.2f\n", tflops);
    printf("==================================================\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
