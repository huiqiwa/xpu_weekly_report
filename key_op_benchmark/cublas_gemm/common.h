#pragma once

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

inline DType parse_dtype(const char* s) {
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

inline const char* dtype_name(DType d) {
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
