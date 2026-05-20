#!/bin/bash
set -e

NVCC="nvcc -O3"
LIBS="-lcublas -lcublasLt"

# Per-dtype targets
$NVCC -o gemm_fp32 gemm_fp32.cu main_fp32.cu $LIBS
$NVCC -o gemm_fp16 gemm_fp16.cu main_fp16.cu $LIBS
$NVCC -o gemm_bf16 gemm_bf16.cu main_bf16.cu $LIBS
$NVCC -o gemm_int8 gemm_int8.cu main_int8.cu $LIBS
$NVCC -o gemm_fp8  gemm_fp8.cu  main_fp8.cu  $LIBS
$NVCC -o gemm_nvfp4 gemm_nvfp4.cu main_nvfp4.cu $LIBS

echo "Build complete."