#!/usr/bin/env python3
"""FlashInfer GEMM benchmark script — flashinfer-only implementations.

Supported dtypes:
  - bfloat16:  flashinfer mm_bf16
  - fp8_e4m3:  flashinfer bmm_fp8
  - nvfp4:     flashinfer mm_fp4

Usage:
    python bench_flashinfer_gemm.py [gpu_id] [dtype] [M] [K] [N] [iters] [warmup]

Examples:
    python bench_flashinfer_gemm.py 0 bfloat16 8192 8192 8192 10 2
    python bench_flashinfer_gemm.py 0 fp8_e4m3 8192 8192 8192 10 2
    python bench_flashinfer_gemm.py 0 nvfp4 8192 8192 8192 10 2
"""
import torch
import sys

try:
    import flashinfer
except ImportError:
    print("Error: flashinfer is not installed. Install with: pip install flashinfer-python")
    sys.exit(1)

default_shapes = {
    'bfloat16':  (8192, 8192, 8192, 10, 2),
    'fp8_e4m3':  (8192, 8192, 8192, 10, 2),
    'nvfp4':     (8192, 8192, 8192, 10, 2),
}

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dtype_str = sys.argv[2] if len(sys.argv) > 2 else 'bfloat16'
defaults = default_shapes.get(dtype_str, (8192, 8192, 8192, 10, 2))
M = int(sys.argv[3]) if len(sys.argv) > 3 else defaults[0]
K = int(sys.argv[4]) if len(sys.argv) > 4 else defaults[1]
N = int(sys.argv[5]) if len(sys.argv) > 5 else defaults[2]
iters = int(sys.argv[6]) if len(sys.argv) > 6 else defaults[3]
warmup = int(sys.argv[7]) if len(sys.argv) > 7 else defaults[4]

supported_dtypes = ['bfloat16', 'fp8_e4m3', 'nvfp4']

if dtype_str not in supported_dtypes:
    print(f'Error: unsupported dtype "{dtype_str}". Supported: {", ".join(supported_dtypes)}')
    sys.exit(1)

device = torch.device(f'cuda:{gpu_id}')
torch.cuda.set_device(device)

print(f"FlashInfer version: {flashinfer.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(device)}")
print()


def bench_gemm():
    """Run the appropriate GEMM benchmark based on dtype."""
    if dtype_str == 'bfloat16':
        return bench_mm_bf16()
    elif dtype_str == 'fp8_e4m3':
        return bench_bmm_fp8()
    elif dtype_str == 'nvfp4':
        return bench_nvfp4()


def bench_mm_bf16():
    """Benchmark flashinfer mm_bf16 (single GEMM, bf16)."""
    backend_name = "flashinfer.mm_bf16"
    out_dtype = torch.bfloat16

    print(f"=== mm_bf16 Benchmark ===")
    print(f"  M={M}, K={K}, N={N}, dtype={dtype_str}")

    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    print(f"  A: {list(a.shape)}, B: {list(b.shape)}")

    # Warmup
    print(f"  Warmup ({warmup} iters)...")
    for _ in range(warmup):
        out = flashinfer.gemm.mm_bf16(a, b, out_dtype=out_dtype)
    torch.cuda.synchronize(device)

    # Benchmark
    print(f"  Benchmarking ({iters} iters)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        out = flashinfer.gemm.mm_bf16(a, b, out_dtype=out_dtype)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * M * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

    print()
    print("  " + "-" * 46)
    print(f"  mm_bf16 [{M} x {K}] x [{K} x {N}] {dtype_str}")
    print(f"    Backend:      {backend_name}")
    print(f"    Avg latency:  {avg_ms:.3f} ms")
    print(f"    Avg TFLOPS:   {tflops:.2f}")
    print("  " + "-" * 46)
    return avg_ms, tflops


def bench_bmm_fp8():
    """Benchmark flashinfer bmm_fp8 (single FP8 matmul)."""
    backend_name = "flashinfer.bmm_fp8"

    print(f"=== bmm_fp8 Benchmark ===")
    print(f"  M={M}, K={K}, N={N}")

    A = torch.randn(1, M, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    B = torch.randn(1, K, N, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)

    # Per-tensor scales
    A_scale = torch.ones(1, dtype=torch.float32, device=device)
    B_scale = torch.ones(1, dtype=torch.float32, device=device)

    print(f"  A: {list(A.shape)}, B: {list(B.shape)}")

    # Warmup
    print(f"  Warmup ({warmup} iters)...")
    for _ in range(warmup):
        out = flashinfer.gemm.bmm_fp8(A, B, A_scale=A_scale, B_scale=B_scale,
                                       dtype=torch.bfloat16)
    torch.cuda.synchronize(device)

    # Benchmark
    print(f"  Benchmarking ({iters} iters)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        out = flashinfer.gemm.bmm_fp8(A, B, A_scale=A_scale, B_scale=B_scale,
                                       dtype=torch.bfloat16)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * M * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

    print()
    print("  " + "-" * 46)
    print(f"  bmm_fp8 [{M} x {K}] x [{K} x {N}]")
    print(f"    Backend:      {backend_name}")
    print(f"    Avg latency:  {avg_ms:.3f} ms")
    print(f"    Avg TFLOPS:   {tflops:.2f}")
    print("  " + "-" * 46)
    return avg_ms, tflops


def bench_nvfp4():
    """Benchmark flashinfer mm_fp4 for NVFP4."""
    backend_name = "flashinfer.mm_fp4"

    print(f"=== NVFP4 GEMM Benchmark (via {backend_name}) ===")
    print(f"  M={M}, K={K}, N={N}")

    block_size = 16
    assert K % (block_size * 2) == 0, f"K ({K}) must be divisible by {block_size * 2} for NVFP4"

    # Packed FP4: each byte holds 2 fp4 values, so K//2 columns
    # mm_fp4 expects a: [M, K//2] row-major, b: [K//2, N] column-major
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device).t()

    # Block scales: one scale per block_size FP4 elements
    # a_descale: (M, K // block_size), b_descale: (K // block_size, N) column-major
    a_descale = torch.ones(M, K // block_size, dtype=torch.float8_e4m3fn, device=device)
    b_descale = torch.ones(N, K // block_size, dtype=torch.float8_e4m3fn, device=device).t()
    alpha = torch.tensor(1.0, dtype=torch.float32, device=device)

    print(f"  A: {list(a.shape)} (packed uint8), B: {list(b.shape)} (packed uint8)")
    print(f"  a_descale: {list(a_descale.shape)}, b_descale: {list(b_descale.shape)}")

    # Warmup
    print(f"  Warmup ({warmup} iters)...")
    for _ in range(warmup):
        out = flashinfer.gemm.mm_fp4(a, b, a_descale, b_descale, alpha=alpha)
    torch.cuda.synchronize(device)

    # Benchmark
    print(f"  Benchmarking ({iters} iters)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        out = flashinfer.gemm.mm_fp4(a, b, a_descale, b_descale, alpha=alpha)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * M * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

    print()
    print("  " + "-" * 46)
    print(f"  nvfp4 [{M} x {K}] x [{K} x {N}]")
    print(f"    Backend:      {backend_name}")
    print(f"    Avg latency:  {avg_ms:.3f} ms")
    print(f"    Avg TFLOPS:   {tflops:.2f}")
    print("  " + "-" * 46)
    return avg_ms, tflops


if __name__ == '__main__':
    print("=" * 50)
    print("  FlashInfer GEMM Benchmark")
    print("=" * 50)
    print()

    avg_ms, tflops = bench_gemm()

    print()
    print("=" * 50)
    print("  Summary")
    print("=" * 50)
    print(f"  Config: M={M}, K={K}, N={N}, dtype={dtype_str}")
    print(f"  Avg latency:  {avg_ms:.3f} ms")
    print(f"  Avg TFLOPS:   {tflops:.2f}")
    print("=" * 50)
