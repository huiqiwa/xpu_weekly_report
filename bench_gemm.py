#!/usr/bin/env python3
"""GEMM benchmark kernel. Called by bench_gemm_with_monitor.sh"""
import torch
import sys
import os

gpu_id = int(sys.argv[1])
M, K, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dtype_str = sys.argv[5]
iters = int(sys.argv[6])
warmup = int(sys.argv[7])

dtype_map = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'tfloat32': torch.float32,
}
dtype = dtype_map[dtype_str]

if dtype_str == 'tfloat32':
    torch.set_float32_matmul_precision('high')
elif dtype_str == 'float32':
    torch.set_float32_matmul_precision('highest')

device = torch.device(f'cuda:{gpu_id}')
torch.cuda.set_device(device)

print(f'Allocating tensors: A[{M}x{K}] x B[{K}x{N}] ({dtype_str})')
a = torch.randn(M, K, dtype=dtype, device=device)
b = torch.randn(K, N, dtype=dtype, device=device)

# Warmup
print(f'Warmup ({warmup} iters)...')
for _ in range(warmup):
    c = torch.matmul(a, b)
torch.cuda.synchronize(device)

# Benchmark
print(f'Benchmarking ({iters} iters)...')
torch.cuda.synchronize(device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for i in range(iters):
    c = torch.matmul(a, b)
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

# Theoretical peak (5090D: 680 TC * freq * 128 OPs)
print()
print('  Reference (5090D BF16 theoretical peak):')
print('    @ 2.407 GHz (boost):  209.5 TFLOPS')
print('    @ 3.090 GHz (max):    268.6 TFLOPS')
print(f'  Efficiency vs boost:    {tflops/209.5*100:.1f}%')
print(f'  Efficiency vs max clk:  {tflops/268.6*100:.1f}%')
print()
