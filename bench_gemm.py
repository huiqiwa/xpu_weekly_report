#!/usr/bin/env python3
"""GEMM benchmark kernel. Called by bench_gemm_with_monitor.sh"""
import torch
import sys
import os

# Default (M, K, N, iters, warmup) per dtype — larger matrices for lower precision
default_shapes = {
    'float32':   (4096, 4096, 4096, 2000, 500),
    'tfloat32':  (4096, 8192, 8192, 3000, 500),
    'float16':   (4096, 8192, 8192, 3000, 500),
    'bfloat16':  (4096, 8192, 8192, 3000, 500),
    'int8':      (8192, 8192, 8192, 3000, 500),
    'fp8_e4m3':  (8192, 8192, 8192, 3000, 500),
    'nvfp4':     (8192, 8192, 8192, 3000, 500),
}

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 7
dtype_str = sys.argv[2] if len(sys.argv) > 2 else 'bfloat16'
defaults = default_shapes.get(dtype_str, (4096, 8192, 8192, 3000, 500))
M = int(sys.argv[3]) if len(sys.argv) > 3 else defaults[0]
K = int(sys.argv[4]) if len(sys.argv) > 4 else defaults[1]
N = int(sys.argv[5]) if len(sys.argv) > 5 else defaults[2]
iters = int(sys.argv[6]) if len(sys.argv) > 6 else defaults[3]
warmup = int(sys.argv[7]) if len(sys.argv) > 7 else defaults[4]

dtype_map = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'tfloat32': torch.float32,
    'int8': torch.int8,
    'fp8_e4m3': torch.float8_e4m3fn,
    'nvfp4': torch.float4_e2m1fn_x2,
}

if dtype_str not in dtype_map:
    print(f'Error: unsupported dtype "{dtype_str}". Supported: {", ".join(dtype_map.keys())}')
    sys.exit(1)
dtype = dtype_map[dtype_str]

if dtype_str == 'tfloat32':
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
elif dtype_str == 'float32':
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

device = torch.device(f'cuda:{gpu_id}')
torch.cuda.set_device(device)

print(f'Allocating tensors: A[{M}x{K}] x B[{K}x{N}] ({dtype_str})')
if dtype_str == 'int8':
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
elif dtype_str == 'fp8_e4m3':
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device).to(dtype)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=device).to(dtype)
elif dtype_str == 'nvfp4':
    assert K % 2 == 0, f'K ({K}) must be even for NVFP4 (x2 packing)'
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device).view(dtype=torch.float4_e2m1fn_x2)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device).view(dtype=torch.float4_e2m1fn_x2)
else:
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)

# Select appropriate GEMM function
if dtype_str == 'int8':
    def gemm_fn(x, y):
        return torch._int_mm(x, y)
elif dtype_str == 'fp8_e4m3':
    scale_a = torch.ones(1, dtype=torch.float32, device=device)
    scale_b = torch.ones(1, dtype=torch.float32, device=device)
    def gemm_fn(x, y):
        return torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b,
                                out_dtype=torch.bfloat16, use_fast_accum=True)
elif dtype_str == 'nvfp4':
    # NVFP4: blockwise 1x16 scaling with fp8_e4m3fn scales
    nvfp4_block_size = 16
    n_scale_a = M * (K // 2) * 2 // nvfp4_block_size
    n_scale_b = N * (K // 2) * 2 // nvfp4_block_size
    scale_a = torch.ones(n_scale_a, dtype=torch.float8_e4m3fn, device=device)
    scale_b = torch.ones(n_scale_b, dtype=torch.float8_e4m3fn, device=device)
    def gemm_fn(x, y):
        return torch._scaled_mm(x, y.t(), scale_a=scale_a, scale_b=scale_b,
                                out_dtype=torch.bfloat16)
else:
    def gemm_fn(x, y):
        return torch.matmul(x, y)

# Warmup
print(f'Warmup ({warmup} iters)...')
for _ in range(warmup):
    c = gemm_fn(a, b)
torch.cuda.synchronize(device)

# Benchmark
print(f'Benchmarking ({iters} iters)...')
torch.cuda.synchronize(device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for i in range(iters):
    c = gemm_fn(a, b)
end_event.record()
torch.cuda.synchronize(device)

total_ms = start_event.elapsed_time(end_event)
avg_ms = total_ms / iters

# FLOPS calculation: 2 * M * N * K per GEMM
flops_per_op = 2.0 * M * N * K
tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

print()
print('=' * 50)
print(f'  GEMM Results: [{M} x {K}] x [{K} x {N}] {dtype_str}')
print('=' * 50)
print(f'  Avg latency:  {avg_ms:.3f} ms')
print(f'  Avg TFLOPS:   {tflops:.2f}')
print(f'  FLOPS/op:     {flops_per_op:.2e}')
print('=' * 50)

