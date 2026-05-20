#pragma once
#include "common.h"

void run_gemm_fp32(cublasHandle_t handle, int M, int N, int K,
                   void* A, void* B, void* C, bool use_tf32);
