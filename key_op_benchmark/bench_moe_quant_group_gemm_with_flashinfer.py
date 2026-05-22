#!/usr/bin/env python3
"""FlashInfer MoE Quant Group GEMM benchmark.

Supported dtypes:
  - bfloat16:         flashinfer.grouped_mm_bf16 (cuDNN MOE backend)
  - fp8_per_tensor:   flashinfer.grouped_mm_fp8 (cuDNN MOE backend, per-tensor scale)
  - fp8_per_block:    flashinfer.group_gemm_fp8_nt_groupwise (CUTLASS, per-block 128x128 scale)
  - mxfp4:            flashinfer.group_gemm_mxfp4_nt_groupwise
  - nvfp4:            flashinfer.group_gemm_nvfp4_nt_groupwise

All group GEMM APIs compute: for each expert i,
    y[tokens_of_expert_i] = x[tokens_of_expert_i] @ W[i]^T

Usage:
    python bench_moe_quant_group_gemm_with_flashinfer.py [gpu_id] [dtype] [num_tokens] [hidden_size] [new_hidden_size] [num_experts] [topk] [ep_size] [iters] [warmup]

Examples:
    python bench_moe_quant_group_gemm_with_flashinfer.py 0 bfloat16 4096 7168 2048 64 8 1 10 2
    python bench_moe_quant_group_gemm_with_flashinfer.py 0 fp8_per_tensor 4096 7168 2048 64 8 1 10 2
    python bench_moe_quant_group_gemm_with_flashinfer.py 0 fp8_per_block 4096 7168 2048 64 8 1 10 2
    python bench_moe_quant_group_gemm_with_flashinfer.py 0 mxfp4 4096 7168 2048 64 8 1 10 2
    python bench_moe_quant_group_gemm_with_flashinfer.py 0 nvfp4 4096 7168 2048 64 8 1 10 2
"""
import torch
import sys
import numpy as np

try:
    import flashinfer
except ImportError:
    print("Error: flashinfer is not installed. Install with: pip install flashinfer-python")
    sys.exit(1)

# ---------- CLI args ----------
gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dtype_str = sys.argv[2] if len(sys.argv) > 2 else 'bfloat16'
num_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 10240
hidden_size = int(sys.argv[4]) if len(sys.argv) > 4 else 4096  # K
new_hidden_size = int(sys.argv[5]) if len(sys.argv) > 5 else 3072  # N
num_experts = int(sys.argv[6]) if len(sys.argv) > 6 else 128
topk = int(sys.argv[7]) if len(sys.argv) > 7 else 8
ep_size = int(sys.argv[8]) if len(sys.argv) > 8 else 8
iters = int(sys.argv[9]) if len(sys.argv) > 9 else 10
warmup = int(sys.argv[10]) if len(sys.argv) > 10 else 2

supported_dtypes = ['bfloat16', 'fp8_per_tensor', 'fp8_per_block', 'mxfp4', 'nvfp4']

if dtype_str not in supported_dtypes:
    print(f'Error: unsupported dtype "{dtype_str}". Supported: {", ".join(supported_dtypes)}')
    sys.exit(1)

device = torch.device(f'cuda:{gpu_id}')
torch.cuda.set_device(device)

K = hidden_size
N = new_hidden_size
num_experts_per_rank = num_experts // ep_size

sm_major, sm_minor = torch.cuda.get_device_capability(device)
sm_version = sm_major * 10 + sm_minor

print(f"FlashInfer version: {flashinfer.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(device)} (SM{sm_version})")
print()

total_runs = warmup + iters
MAX_ALLOC_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB limit for tensor pool


def _num_copies(bytes_per_set):
    """Determine how many unique tensor sets to allocate (capped by memory limit)."""
    return max(1, min(total_runs, MAX_ALLOC_BYTES // max(bytes_per_set, 1)))


# ---------- MoE token dispatch simulation ----------
def simulate_moe_dispatch(num_tokens, num_experts, topk, ep_size=1):
    """Simulate MoE routing: assign tokens to experts on a single EP rank.

    With ep_size > 1, only num_experts // ep_size experts are on this rank,
    and only the corresponding fraction of dispatched tokens land here.

    Returns:
        m_indptr: (num_experts_per_rank + 1,) int32 tensor, padded to multiples of 4
        tokens_per_expert: actual token counts per expert on this rank
        tokens_per_expert_padded: padded token counts
        total_dispatch_tokens: total (token, expert) pairs on this rank
        total_padded: total padded rows
    """
    num_experts_per_rank = num_experts // ep_size
    total_dispatch_tokens = num_tokens * topk // ep_size

    # Round-robin uniform distribution (each expert on this rank gets equal share)
    tokens_per_expert = np.full(num_experts_per_rank, total_dispatch_tokens // num_experts_per_rank, dtype=np.int32)
    remainder = total_dispatch_tokens % num_experts_per_rank
    tokens_per_expert[:remainder] += 1

    # Pad each segment to multiple of 4 (required by flashinfer group GEMM kernels)
    tokens_per_expert_padded = ((tokens_per_expert + 3) // 4) * 4

    # Build m_indptr
    m_indptr = np.zeros(num_experts_per_rank + 1, dtype=np.int32)
    m_indptr[1:] = np.cumsum(tokens_per_expert_padded)
    total_padded = int(m_indptr[-1])

    return (
        torch.from_numpy(m_indptr).to(device),
        tokens_per_expert,
        tokens_per_expert_padded,
        total_dispatch_tokens,
        total_padded,
    )


# ---------- Benchmark functions ----------
def bench_bf16():
    """Benchmark MoE group GEMM with bfloat16 using grouped_mm_bf16 (cuDNN MOE)."""
    backend_name = "flashinfer.grouped_mm_bf16 (cuDNN MOE)"
    print(f"=== MoE Group GEMM (BF16) ===")
    print(f"  num_tokens={num_tokens}, K={K}, N={N}, num_experts={num_experts}, topk={topk}, ep_size={ep_size}")

    m_indptr, tokens_per_expert, tokens_per_expert_padded, total_dispatch, total_padded = \
        simulate_moe_dispatch(num_tokens, num_experts, topk, ep_size)

    # Allocate multiple tensor sets to prevent cache hits
    set_bytes = (total_padded * K + num_experts_per_rank * N * K) * 2  # bf16
    n_copies = _num_copies(set_bytes)
    a_t = torch.randn(total_padded, K, dtype=torch.bfloat16, device=device)
    b_t = torch.randn(num_experts_per_rank, N, K, dtype=torch.bfloat16, device=device)
    a_list = [a_t.clone() for _ in range(n_copies)]
    b_list = [b_t.clone() for _ in range(n_copies)]
    del a_t, b_t

    print(f"  a: [{total_padded}, {K}], b: [{num_experts_per_rank}, {N}, {K}]")
    print(f"  total_padded_tokens: {total_padded}, tensor copies: {n_copies}")

    # Warmup
    for i in range(warmup):
        idx = i % n_copies
        out = flashinfer.grouped_mm_bf16(a_list[idx], b_list[idx], m_indptr, out_dtype=torch.bfloat16)
    torch.cuda.synchronize(device)

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(iters):
        idx = (warmup + i) % n_copies
        out = flashinfer.grouped_mm_bf16(a_list[idx], b_list[idx], m_indptr, out_dtype=torch.bfloat16)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12
    # Memory: read a + b (active experts only), write out
    active_experts = int(np.count_nonzero(tokens_per_expert))
    total_bytes = (total_dispatch * K * 2  # a bf16
                   + active_experts * N * K * 2  # b bf16 (active only)
                   + total_dispatch * N * 2)  # out bf16
    mem_bw = total_bytes / (avg_ms / 1000.0) / 1e9  # GB/s

    print_result(backend_name, avg_ms, tflops, mem_bw)
    return avg_ms, tflops, mem_bw


def bench_fp8_per_tensor():
    """Benchmark MoE group GEMM with FP8 per-tensor scaling using grouped_mm_fp8 (cuDNN MOE)."""
    backend_name = "flashinfer.grouped_mm_fp8 (cuDNN MOE)"
    print(f"=== MoE Group GEMM (FP8 per-tensor, cuDNN MOE) ===")
    print(f"  num_tokens={num_tokens}, K={K}, N={N}, num_experts={num_experts}, topk={topk}, ep_size={ep_size}")

    m_indptr, tokens_per_expert, tokens_per_expert_padded, total_dispatch, total_padded = \
        simulate_moe_dispatch(num_tokens, num_experts, topk, ep_size)

    # Allocate multiple tensor sets to prevent cache hits
    set_bytes = (total_padded * K + num_experts_per_rank * N * K) * 1  # fp8
    n_copies = _num_copies(set_bytes)
    a_t = torch.randn(total_padded, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    b_t = torch.randn(num_experts_per_rank, N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    a_list = [a_t.clone() for _ in range(n_copies)]
    b_list = [b_t.clone() for _ in range(n_copies)]
    del a_t, b_t

    print(f"  a: [{total_padded}, {K}], b: [{num_experts_per_rank}, {N}, {K}]")
    print(f"  total_padded_tokens: {total_padded}, tensor copies: {n_copies}")

    for i in range(warmup):
        idx = i % n_copies
        out = flashinfer.grouped_mm_fp8(a_list[idx], b_list[idx], m_indptr, out_dtype=torch.bfloat16)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iters):
        idx = (warmup + i) % n_copies
        out = flashinfer.grouped_mm_fp8(a_list[idx], b_list[idx], m_indptr, out_dtype=torch.bfloat16)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12
    # Memory: read a + b (active experts only), write out
    active_experts = int(np.count_nonzero(tokens_per_expert))
    total_bytes = (total_dispatch * K * 1  # a fp8
                   + active_experts * N * K * 1  # b fp8 (active only)
                   + total_dispatch * N * 2)  # out bf16
    mem_bw = total_bytes / (avg_ms / 1000.0) / 1e9

    print_result(backend_name, avg_ms, tflops, mem_bw)
    return avg_ms, tflops, mem_bw


def bench_fp8_per_block():
    """Benchmark MoE group GEMM with FP8 per-block (128x128) using gemm_fp8_nt_groupwise per expert."""
    backend_name = "flashinfer.gemm_fp8_nt_groupwise (per-block 128x128, per-expert loop)"
    block_size = 128
    print(f"=== MoE Group GEMM (FP8 per-block, gemm_fp8_nt_groupwise unrolled) ===")
    print(f"  num_tokens={num_tokens}, K={K}, N={N}, num_experts={num_experts}, topk={topk}, ep_size={ep_size}")

    m_indptr, tokens_per_expert, tokens_per_expert_padded, total_dispatch, total_padded = \
        simulate_moe_dispatch(num_tokens, num_experts, topk, ep_size)

    m_indptr_cpu = m_indptr.cpu().numpy()

    # Per-expert a_scale sizes
    a_scale_sizes = []
    for i in range(num_experts_per_rank):
        m_i = int(m_indptr_cpu[i + 1] - m_indptr_cpu[i])
        a_scale_sizes.append((m_i, K // block_size))
    b_scale = torch.ones(N // block_size, K // block_size, dtype=torch.float32, device=device)

    # Allocate multiple tensor sets to prevent cache hits
    a_scale_bytes = sum(r * c * 4 for r, c in a_scale_sizes)
    set_bytes = (total_padded * K * 1  # a fp8
                 + num_experts_per_rank * N * K * 1  # b fp8
                 + total_padded * N * 2  # out bf16
                 + a_scale_bytes)  # a_scales float32
    n_copies = _num_copies(set_bytes)

    a_t = torch.randn(total_padded, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    b_t = torch.randn(num_experts_per_rank, N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    out_t = torch.empty(total_padded, N, dtype=torch.bfloat16, device=device)
    a_scales_t = [torch.ones(r, c, dtype=torch.float32, device=device) for r, c in a_scale_sizes]

    a_list = [a_t.clone() for _ in range(n_copies)]
    b_list = [b_t.clone() for _ in range(n_copies)]
    out_list = [out_t.clone() for _ in range(n_copies)]
    a_scales_list = [[s.clone() for s in a_scales_t] for _ in range(n_copies)]
    del a_t, b_t, out_t, a_scales_t

    print(f"  a: [{total_padded}, {K}], b: [{num_experts_per_rank}, {N}, {K}]")
    print(f"  a_scale: [{num_experts_per_rank} x (m_i, {K // block_size})], b_scale: {list(b_scale.shape)}")
    print(f"  total_padded_tokens: {total_padded}, tensor copies: {n_copies}")

    def run_once(copy_idx):
        ac = a_list[copy_idx]
        bc = b_list[copy_idx]
        oc = out_list[copy_idx]
        sc = a_scales_list[copy_idx]
        for i in range(num_experts_per_rank):
            start = int(m_indptr_cpu[i])
            end = int(m_indptr_cpu[i + 1])
            flashinfer.gemm.gemm_fp8_nt_groupwise(
                ac[start:end], bc[i], sc[i], b_scale,
                scale_granularity_mnk=(1, block_size, block_size),
                scale_major_mode='K', out=oc[start:end], out_dtype=torch.bfloat16)

    for i in range(warmup):
        run_once(i % n_copies)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iters):
        run_once((warmup + i) % n_copies)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12
    # Memory: read a + b + scales (active experts only), write out
    active_experts = int(np.count_nonzero(tokens_per_expert))
    total_a_scale_bytes = sum(r * c * 4 for r, c in a_scale_sizes[:active_experts])
    total_bytes = (total_dispatch * K * 1  # a fp8
                   + active_experts * N * K * 1  # b fp8 (active only)
                   + total_a_scale_bytes
                   + active_experts * b_scale.numel() * b_scale.element_size()
                   + total_dispatch * N * 2)  # out
    mem_bw = total_bytes / (avg_ms / 1000.0) / 1e9

    print_result(backend_name, avg_ms, tflops, mem_bw)
    return avg_ms, tflops, mem_bw


def bench_mxfp4():
    """Benchmark MoE group GEMM with MXFP4."""
    backend_name = "flashinfer.group_gemm_mxfp4_nt_groupwise"
    block_size = 32  # MXFP4 uses groups of 32 for scales
    print(f"=== MoE Group GEMM (MXFP4) ===")
    print(f"  num_tokens={num_tokens}, K={K}, N={N}, num_experts={num_experts}, topk={topk}, ep_size={ep_size}")

    assert K % 32 == 0, f"K ({K}) must be divisible by 32 for MXFP4"

    m_indptr, tokens_per_expert, tokens_per_expert_padded, total_dispatch, total_padded = \
        simulate_moe_dispatch(num_tokens, num_experts, topk, ep_size)

    # a_scale needs alignment padding for CUTLASS group GEMM kernel
    alignment = 128
    m_indptr_cpu = m_indptr.cpu().numpy()
    last_group = num_experts_per_rank - 1
    last_sf_m_offset = int((m_indptr_cpu[last_group] + last_group * (alignment - 1)) // alignment * alignment)
    last_sf_rows = int(((tokens_per_expert_padded[last_group] + alignment - 1) // alignment) * alignment)
    total_sf_rows = last_sf_m_offset + last_sf_rows

    sf_k = K // block_size
    N_padded = ((N + 127) // 128) * 128

    # Allocate multiple tensor sets to prevent cache hits
    set_bytes = (total_padded * K * 1  # a fp8
                 + num_experts_per_rank * N * (K // 2) * 1  # b uint8
                 + total_sf_rows * sf_k * 1  # a_scale uint8
                 + num_experts_per_rank * N_padded * sf_k * 1)  # b_scale uint8
    n_copies = _num_copies(set_bytes)

    a_t = torch.randn(total_padded, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    b_t = torch.randint(0, 256, (num_experts_per_rank, N, K // 2), dtype=torch.uint8, device=device)
    # MXFP4 scales use E8M0 format: biased exponent, 127 = 2^(127-127) = 1.0
    a_scale_t = torch.full((total_sf_rows, sf_k), 127, dtype=torch.uint8, device=device)
    b_scale_t = torch.full((num_experts_per_rank, N_padded, sf_k), 127, dtype=torch.uint8, device=device)

    a_list = [a_t.clone() for _ in range(n_copies)]
    b_list = [b_t.clone() for _ in range(n_copies)]
    a_scale_list = [a_scale_t.clone() for _ in range(n_copies)]
    b_scale_list = [b_scale_t.clone() for _ in range(n_copies)]
    del a_t, b_t, a_scale_t, b_scale_t

    print(f"  a: [{total_padded}, {K}], b: [{num_experts_per_rank}, {N}, {K // 2}]")
    print(f"  a_scale: [{total_sf_rows}, {sf_k}], b_scale: [{num_experts_per_rank}, {N_padded}, {sf_k}]")
    print(f"  total_padded_tokens: {total_padded}, tensor copies: {n_copies}")

    for i in range(warmup):
        idx = i % n_copies
        out = flashinfer.gemm.group_gemm_mxfp4_nt_groupwise(
            a_list[idx], b_list[idx], a_scale_list[idx], b_scale_list[idx], m_indptr,
            tile_n=128, out_dtype=torch.bfloat16)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iters):
        idx = (warmup + i) % n_copies
        out = flashinfer.gemm.group_gemm_mxfp4_nt_groupwise(
            a_list[idx], b_list[idx], a_scale_list[idx], b_scale_list[idx], m_indptr,
            tile_n=128, out_dtype=torch.bfloat16)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12
    # Memory: read a + b + scales (active experts only), write out
    active_experts = int(np.count_nonzero(tokens_per_expert))
    total_bytes = (total_dispatch * K * 1  # a fp8
                   + active_experts * N * (K // 2) * 1  # b uint8 packed (active only)
                   + total_sf_rows * sf_k * 1  # a_scale
                   + active_experts * N_padded * sf_k * 1  # b_scale (active only)
                   + total_dispatch * N * 2)  # out
    mem_bw = total_bytes / (avg_ms / 1000.0) / 1e9

    print_result(backend_name, avg_ms, tflops, mem_bw)
    return avg_ms, tflops, mem_bw


def bench_nvfp4():
    """Benchmark MoE group GEMM with NVFP4."""
    backend_name = "flashinfer.group_gemm_nvfp4_nt_groupwise"
    block_size = 16  # NVFP4 uses groups of 16 for scales
    print(f"=== MoE Group GEMM (NVFP4) ===")
    print(f"  num_tokens={num_tokens}, K={K}, N={N}, num_experts={num_experts}, topk={topk}, ep_size={ep_size}")

    assert K % 32 == 0, f"K ({K}) must be divisible by 32 for NVFP4"

    m_indptr, tokens_per_expert, tokens_per_expert_padded, total_dispatch, total_padded = \
        simulate_moe_dispatch(num_tokens, num_experts, topk, ep_size)

    # a_scale needs special alignment padding for NVFP4 kernel
    alignment = 128
    m_indptr_cpu = m_indptr.cpu().numpy()
    last_group = num_experts_per_rank - 1
    last_sf_m_offset = int((m_indptr_cpu[last_group] + last_group * (alignment - 1)) // alignment * alignment)
    last_sf_rows = int(((tokens_per_expert_padded[last_group] + alignment - 1) // alignment) * alignment)
    total_sf_rows = last_sf_m_offset + last_sf_rows

    sf_k = K // block_size
    N_padded = ((N + 127) // 128) * 128
    # alpha: per-expert scale (shared, tiny)
    alpha = torch.ones(num_experts_per_rank, dtype=torch.float32, device=device)

    # Allocate multiple tensor sets to prevent cache hits
    set_bytes = (total_padded * (K // 2) * 1  # a uint8
                 + num_experts_per_rank * N * (K // 2) * 1  # b uint8
                 + total_sf_rows * sf_k * 1  # a_scale uint8
                 + num_experts_per_rank * N_padded * sf_k * 1)  # b_scale uint8
    n_copies = _num_copies(set_bytes)

    a_t = torch.randint(0, 256, (total_padded, K // 2), dtype=torch.uint8, device=device)
    b_t = torch.randint(0, 256, (num_experts_per_rank, N, K // 2), dtype=torch.uint8, device=device)
    # NVFP4 scales use E4M3 format: 0x3C = float8_e4m3fn(1.0)
    a_scale_t = torch.full((total_sf_rows, sf_k), 0x3C, dtype=torch.uint8, device=device)
    b_scale_t = torch.full((num_experts_per_rank, N_padded, sf_k), 0x3C, dtype=torch.uint8, device=device)

    a_list = [a_t.clone() for _ in range(n_copies)]
    b_list = [b_t.clone() for _ in range(n_copies)]
    a_scale_list = [a_scale_t.clone() for _ in range(n_copies)]
    b_scale_list = [b_scale_t.clone() for _ in range(n_copies)]
    del a_t, b_t, a_scale_t, b_scale_t

    print(f"  a: [{total_padded}, {K // 2}], b: [{num_experts_per_rank}, {N}, {K // 2}]")
    print(f"  a_scale: [{total_sf_rows}, {sf_k}], b_scale: [{num_experts_per_rank}, {N_padded}, {sf_k}]")
    print(f"  total_padded_tokens: {total_padded}, tensor copies: {n_copies}")

    for i in range(warmup):
        idx = i % n_copies
        out = flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            a_list[idx], b_list[idx], a_scale_list[idx], b_scale_list[idx], m_indptr,
            alpha=alpha, tile_n=128, out_dtype=torch.bfloat16)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iters):
        idx = (warmup + i) % n_copies
        out = flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            a_list[idx], b_list[idx], a_scale_list[idx], b_scale_list[idx], m_indptr,
            alpha=alpha, tile_n=128, out_dtype=torch.bfloat16)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12
    # Memory: read a + b + scales + alpha (active experts only), write out
    active_experts = int(np.count_nonzero(tokens_per_expert))
    total_bytes = (total_dispatch * (K // 2) * 1  # a uint8 packed
                   + active_experts * N * (K // 2) * 1  # b uint8 packed (active only)
                   + total_sf_rows * sf_k * 1  # a_scale
                   + active_experts * N_padded * sf_k * 1  # b_scale (active only)
                   + active_experts * alpha.element_size()  # alpha
                   + total_dispatch * N * 2)  # out
    mem_bw = total_bytes / (avg_ms / 1000.0) / 1e9

    print_result(backend_name, avg_ms, tflops, mem_bw)
    return avg_ms, tflops, mem_bw


# ---------- Helpers ----------
def print_result(backend_name, avg_ms, tflops, mem_bw):
    print()
    print("  " + "-" * 56)
    print(f"  MoE Group GEMM [{num_tokens}*{topk} tokens, {num_experts} experts, ep_size={ep_size}]")
    print(f"    Shape:        [{K}] x [{N}] per expert (NT layout)")
    print(f"    Backend:      {backend_name}")
    print(f"    Avg latency:  {avg_ms:.3f} ms")
    print(f"    Avg TFLOPS:   {tflops:.2f}")
    print(f"    Mem BW:       {mem_bw:.2f} GB/s")
    print("  " + "-" * 56)


# ---------- Main ----------
if __name__ == '__main__':
    print("=" * 60)
    print("  FlashInfer MoE Quant Group GEMM Benchmark")
    print("=" * 60)
    print()
    print(f"  Config: num_tokens={num_tokens}, K={K}, N={N}")
    print(f"          num_experts={num_experts}, topk={topk}, ep_size={ep_size}")
    print(f"          dtype={dtype_str}, iters={iters}, warmup={warmup}")
    print()

    dispatch = {
        'bfloat16': bench_bf16,
        'fp8_per_tensor': bench_fp8_per_tensor,
        'fp8_per_block': bench_fp8_per_block,
        'mxfp4': bench_mxfp4,
        'nvfp4': bench_nvfp4,
    }

    avg_ms, tflops, mem_bw = dispatch[dtype_str]()

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  dtype={dtype_str}, num_tokens={num_tokens}, K={K}, N={N}")
    print(f"  num_experts={num_experts}, topk={topk}")
    print(f"  Avg latency:  {avg_ms:.3f} ms")
    print(f"  Avg TFLOPS:   {tflops:.2f}")
    print(f"  Mem BW:       {mem_bw:.2f} GB/s")
    print("=" * 60)
