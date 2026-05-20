#!/usr/bin/env python3
"""FlashInfer MoE Quant Group GEMM benchmark.

Supported dtypes:
  - bfloat16:       flashinfer SegmentGEMMWrapper (native group GEMM)
  - fp8_per_block:  flashinfer group_gemm_fp8_nt_groupwise (SM100/103/110)
                    or per-expert gemm_fp8_nt_blockscaled fallback (SM120/121)
  - fp8_per_tensor: flashinfer group_gemm_fp8_nt_groupwise with full-tensor scale
                    or per-expert bmm_fp8 fallback (SM120/121)
  - mxfp4:          flashinfer group_gemm_mxfp4_nt_groupwise
  - nvfp4:          flashinfer group_gemm_nvfp4_nt_groupwise

All group GEMM APIs compute: for each expert i,
    y[tokens_of_expert_i] = x[tokens_of_expert_i] @ W[i]^T

Usage:
    python bench_moe_quant_group_gemm_with_flashinfer.py [gpu_id] [dtype] [num_tokens] [hidden_size] [new_hidden_size] [num_experts] [topk] [iters] [warmup]

Examples:
    python bench_moe_quant_group_gemm_with_flashinfer.py 0 bfloat16 4096 7168 2048 64 8 10 2
    python bench_moe_quant_group_gemm_with_flashinfer.py 0 fp8_per_block 4096 7168 2048 64 8 10 2
    python bench_moe_quant_group_gemm_with_flashinfer.py 0 fp8_per_tensor 4096 7168 2048 64 8 10 2
    python bench_moe_quant_group_gemm_with_flashinfer.py 0 mxfp4 4096 7168 2048 64 8 10 2
    python bench_moe_quant_group_gemm_with_flashinfer.py 0 nvfp4 4096 7168 2048 64 8 10 2
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
num_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
hidden_size = int(sys.argv[4]) if len(sys.argv) > 4 else 7168  # K
new_hidden_size = int(sys.argv[5]) if len(sys.argv) > 5 else 2048  # N
num_experts = int(sys.argv[6]) if len(sys.argv) > 6 else 64
topk = int(sys.argv[7]) if len(sys.argv) > 7 else 8
iters = int(sys.argv[8]) if len(sys.argv) > 8 else 10
warmup = int(sys.argv[9]) if len(sys.argv) > 9 else 2

supported_dtypes = ['bfloat16', 'fp8_per_block', 'fp8_per_tensor', 'mxfp4', 'nvfp4']

if dtype_str not in supported_dtypes:
    print(f'Error: unsupported dtype "{dtype_str}". Supported: {", ".join(supported_dtypes)}')
    sys.exit(1)

device = torch.device(f'cuda:{gpu_id}')
torch.cuda.set_device(device)

K = hidden_size
N = new_hidden_size

sm_major, sm_minor = torch.cuda.get_device_capability(device)
sm_version = sm_major * 10 + sm_minor

print(f"FlashInfer version: {flashinfer.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(device)} (SM{sm_version})")
print()


# ---------- MoE token dispatch simulation ----------
def simulate_moe_dispatch(num_tokens, num_experts, topk):
    """Simulate MoE routing: assign tokens to experts.

    Returns:
        m_indptr: (num_experts + 1,) int32 tensor, padded to multiples of 4
        tokens_per_expert: actual token counts
        total_dispatch_tokens: total (token, expert) pairs
        total_padded: total padded rows
    """
    total_dispatch_tokens = num_tokens * topk

    # Randomly assign tokens to experts (uniform distribution)
    expert_ids = np.random.randint(0, num_experts, size=total_dispatch_tokens)
    tokens_per_expert = np.bincount(expert_ids, minlength=num_experts).astype(np.int32)

    # Pad each segment to multiple of 4 (required by flashinfer group GEMM kernels)
    tokens_per_expert_padded = ((tokens_per_expert + 3) // 4) * 4

    # Build m_indptr
    m_indptr = np.zeros(num_experts + 1, dtype=np.int32)
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
    """Benchmark MoE group GEMM with bfloat16 using SegmentGEMMWrapper."""
    backend_name = "flashinfer.SegmentGEMMWrapper"
    print(f"=== MoE Group GEMM (BF16) ===")
    print(f"  num_tokens={num_tokens}, K={K}, N={N}, num_experts={num_experts}, topk={topk}")

    m_indptr, tokens_per_expert, tokens_per_expert_padded, total_dispatch, total_padded = \
        simulate_moe_dispatch(num_tokens, num_experts, topk)

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    segment_gemm = flashinfer.gemm.SegmentGEMMWrapper(workspace_buffer)

    x = torch.randn(total_padded, K, dtype=torch.bfloat16, device=device)
    weights = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device=device)
    weight_indices = torch.arange(num_experts, dtype=torch.int64, device=device)

    print(f"  x: {list(x.shape)}, weights: {list(weights.shape)}")
    print(f"  total_padded_tokens: {total_padded}")

    # Warmup
    for _ in range(warmup):
        out = segment_gemm.run(x, weights, batch_size=num_experts,
                               weight_column_major=False,
                               weight_indices=weight_indices,
                               seg_indptr=m_indptr.to(torch.int64))
    torch.cuda.synchronize(device)

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        out = segment_gemm.run(x, weights, batch_size=num_experts,
                               weight_column_major=False,
                               weight_indices=weight_indices,
                               seg_indptr=m_indptr.to(torch.int64))
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

    print_result(backend_name, avg_ms, tflops)
    return avg_ms, tflops


def bench_fp8_per_block():
    """Benchmark MoE group GEMM with FP8 per-block scaling (128x128)."""
    block_size = 128
    print(f"=== MoE Group GEMM (FP8 per-block, scale_granularity=128x128) ===")
    print(f"  num_tokens={num_tokens}, K={K}, N={N}, num_experts={num_experts}, topk={topk}")

    m_indptr, tokens_per_expert, tokens_per_expert_padded, total_dispatch, total_padded = \
        simulate_moe_dispatch(num_tokens, num_experts, topk)

    # Check SM120/121 limitation
    if sm_version in (120, 121) and num_experts > 1:
        backend_name = "flashinfer.gemm_fp8_nt_blockscaled (per-expert loop, SM120 fallback)"
        print(f"  NOTE: group_gemm_fp8 has correctness issues for num_groups>1 on SM{sm_version}")
        print(f"        Using per-expert loop with gemm_fp8_nt_blockscaled")
        return _bench_fp8_per_block_loop(m_indptr, tokens_per_expert_padded,
                                          total_dispatch, total_padded, backend_name)
    else:
        backend_name = "flashinfer.group_gemm_fp8_nt_groupwise (per-block 128x128)"
        return _bench_fp8_per_block_group(m_indptr, tokens_per_expert_padded,
                                           total_dispatch, total_padded, backend_name)


def _bench_fp8_per_block_group(m_indptr, tokens_per_expert_padded, total_dispatch, total_padded, backend_name):
    """Native group GEMM path for FP8 per-block (SM100/103/110)."""
    block_size = 128
    a = torch.randn(total_padded, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    a_scale = torch.ones(total_padded, K // block_size, dtype=torch.float32, device=device)
    b_scale = torch.ones(num_experts, N // block_size, K // block_size, dtype=torch.float32, device=device)

    print(f"  a: {list(a.shape)}, b: {list(b.shape)}")
    print(f"  total_padded_tokens: {total_padded}")

    for _ in range(warmup):
        out = flashinfer.gemm.group_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr,
            scale_granularity_mnk=(1, block_size, block_size),
            scale_major_mode='K', out_dtype=torch.bfloat16)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        out = flashinfer.gemm.group_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr,
            scale_granularity_mnk=(1, block_size, block_size),
            scale_major_mode='K', out_dtype=torch.bfloat16)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

    print_result(backend_name, avg_ms, tflops)
    return avg_ms, tflops


def _bench_fp8_per_block_loop(m_indptr, tokens_per_expert_padded, total_dispatch, total_padded, backend_name):
    """Per-expert loop fallback for FP8 per-block on SM120/121."""
    block_size = 128
    a_full = torch.randn(total_padded, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    a_scale_full = torch.ones(total_padded, K // block_size, dtype=torch.float32, device=device)
    b_scale = torch.ones(num_experts, N // block_size, K // block_size, dtype=torch.float32, device=device)
    out_full = torch.empty(total_padded, N, dtype=torch.bfloat16, device=device)

    m_indptr_cpu = m_indptr.cpu().numpy()

    print(f"  a: {list(a_full.shape)}, b: {list(b.shape)}")
    print(f"  total_padded_tokens: {total_padded}")

    def run_loop():
        for i in range(num_experts):
            start_row = int(m_indptr_cpu[i])
            end_row = int(m_indptr_cpu[i + 1])
            if start_row == end_row:
                continue
            a_i = a_full[start_row:end_row]
            a_scale_i = a_scale_full[start_row:end_row]
            out_i = out_full[start_row:end_row]
            flashinfer.gemm.gemm_fp8_nt_blockscaled(
                a_i, b[i], a_scale_i, b_scale[i],
                scale_major_mode='K', out=out_i, out_dtype=torch.bfloat16)

    for _ in range(warmup):
        run_loop()
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        run_loop()
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

    print_result(backend_name, avg_ms, tflops)
    return avg_ms, tflops


def bench_fp8_per_tensor():
    """Benchmark MoE group GEMM with FP8 per-tensor scaling."""
    print(f"=== MoE Group GEMM (FP8 per-tensor) ===")
    print(f"  num_tokens={num_tokens}, K={K}, N={N}, num_experts={num_experts}, topk={topk}")

    m_indptr, tokens_per_expert, tokens_per_expert_padded, total_dispatch, total_padded = \
        simulate_moe_dispatch(num_tokens, num_experts, topk)

    if sm_version in (120, 121) and num_experts > 1:
        backend_name = "flashinfer.bmm_fp8 (per-expert loop, SM120 fallback)"
        print(f"  NOTE: group_gemm_fp8 has correctness issues for num_groups>1 on SM{sm_version}")
        print(f"        Using per-expert loop with bmm_fp8")
        return _bench_fp8_per_tensor_loop(m_indptr, tokens_per_expert_padded,
                                           total_dispatch, total_padded, backend_name)
    else:
        backend_name = "flashinfer.group_gemm_fp8_nt_groupwise (per-tensor)"
        return _bench_fp8_per_tensor_group(m_indptr, tokens_per_expert_padded,
                                            total_dispatch, total_padded, backend_name)


def _bench_fp8_per_tensor_group(m_indptr, tokens_per_expert_padded, total_dispatch, total_padded, backend_name):
    """Native group GEMM path for FP8 per-tensor (SM100/103/110)."""
    a = torch.randn(total_padded, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    a_scale = torch.ones(total_padded, 1, dtype=torch.float32, device=device)
    b_scale = torch.ones(num_experts, 1, 1, dtype=torch.float32, device=device)

    print(f"  a: {list(a.shape)}, b: {list(b.shape)}")
    print(f"  total_padded_tokens: {total_padded}")

    for _ in range(warmup):
        out = flashinfer.gemm.group_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr,
            scale_granularity_mnk=(1, N, K),
            scale_major_mode='K', out_dtype=torch.bfloat16)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        out = flashinfer.gemm.group_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr,
            scale_granularity_mnk=(1, N, K),
            scale_major_mode='K', out_dtype=torch.bfloat16)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

    print_result(backend_name, avg_ms, tflops)
    return avg_ms, tflops


def _bench_fp8_per_tensor_loop(m_indptr, tokens_per_expert_padded, total_dispatch, total_padded, backend_name):
    """Per-expert loop fallback for FP8 per-tensor on SM120/121."""
    a_full = torch.randn(total_padded, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    # Per-tensor scale: one scalar per expert
    a_scale = torch.ones(num_experts, dtype=torch.float32, device=device)
    b_scale = torch.ones(num_experts, dtype=torch.float32, device=device)
    out_full = torch.empty(total_padded, N, dtype=torch.bfloat16, device=device)

    m_indptr_cpu = m_indptr.cpu().numpy()

    print(f"  a: {list(a_full.shape)}, b: {list(b.shape)}")
    print(f"  total_padded_tokens: {total_padded}")

    def run_loop():
        for i in range(num_experts):
            start_row = int(m_indptr_cpu[i])
            end_row = int(m_indptr_cpu[i + 1])
            if start_row == end_row:
                continue
            rows = end_row - start_row
            # bmm_fp8 requires 3D: (1, M, K) x (1, K, N)
            a_i = a_full[start_row:end_row].unsqueeze(0)  # (1, rows, K)
            b_i = b[i].t().unsqueeze(0)  # (N, K) -> (K, N) -> (1, K, N)
            out_i = flashinfer.gemm.bmm_fp8(
                a_i, b_i,
                A_scale=a_scale[i:i+1], B_scale=b_scale[i:i+1],
                dtype=torch.bfloat16)
            out_full[start_row:end_row] = out_i.squeeze(0)

    for _ in range(warmup):
        run_loop()
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        run_loop()
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

    print_result(backend_name, avg_ms, tflops)
    return avg_ms, tflops


def bench_mxfp4():
    """Benchmark MoE group GEMM with MXFP4."""
    backend_name = "flashinfer.group_gemm_mxfp4_nt_groupwise"
    block_size = 32  # MXFP4 uses groups of 32 for scales
    print(f"=== MoE Group GEMM (MXFP4) ===")
    print(f"  num_tokens={num_tokens}, K={K}, N={N}, num_experts={num_experts}, topk={topk}")

    assert K % 32 == 0, f"K ({K}) must be divisible by 32 for MXFP4"

    m_indptr, tokens_per_expert, tokens_per_expert_padded, total_dispatch, total_padded = \
        simulate_moe_dispatch(num_tokens, num_experts, topk)

    # a: (total_padded, K) fp8 (MXFP4 activation is fp8 input)
    a = torch.randn(total_padded, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    # b: (num_experts, N, K // 2) uint8 packed fp4
    b = torch.randint(0, 256, (num_experts, N, K // 2), dtype=torch.uint8, device=device)

    # a_scale: (total_padded, K // 32) uint8 (MX block scale, 1 scale per 32 elements)
    a_scale = torch.ones(total_padded, K // block_size, dtype=torch.uint8, device=device)
    # b_scale: (num_experts, N_padded, K // 32) uint8
    N_padded = ((N + 127) // 128) * 128
    b_scale = torch.ones(num_experts, N_padded, K // block_size, dtype=torch.uint8, device=device)

    print(f"  a: {list(a.shape)}, b: {list(b.shape)}")
    print(f"  a_scale: {list(a_scale.shape)}, b_scale: {list(b_scale.shape)}")
    print(f"  total_padded_tokens: {total_padded}")

    for _ in range(warmup):
        out = flashinfer.gemm.group_gemm_mxfp4_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr,
            tile_n=128, out_dtype=torch.bfloat16)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        out = flashinfer.gemm.group_gemm_mxfp4_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr,
            tile_n=128, out_dtype=torch.bfloat16)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

    print_result(backend_name, avg_ms, tflops)
    return avg_ms, tflops


def bench_nvfp4():
    """Benchmark MoE group GEMM with NVFP4."""
    backend_name = "flashinfer.group_gemm_nvfp4_nt_groupwise"
    block_size = 16  # NVFP4 uses groups of 16 for scales
    print(f"=== MoE Group GEMM (NVFP4) ===")
    print(f"  num_tokens={num_tokens}, K={K}, N={N}, num_experts={num_experts}, topk={topk}")

    assert K % 32 == 0, f"K ({K}) must be divisible by 32 for NVFP4"

    m_indptr, tokens_per_expert, tokens_per_expert_padded, total_dispatch, total_padded = \
        simulate_moe_dispatch(num_tokens, num_experts, topk)

    # a: (total_padded, K // 2) uint8 packed nvfp4
    a = torch.randint(0, 256, (total_padded, K // 2), dtype=torch.uint8, device=device)
    # b: (num_experts, N, K // 2) uint8 packed nvfp4
    b = torch.randint(0, 256, (num_experts, N, K // 2), dtype=torch.uint8, device=device)

    # a_scale: (total_padded, K // 16) uint8
    a_scale = torch.ones(total_padded, K // block_size, dtype=torch.uint8, device=device)
    # b_scale: (num_experts, N_padded, K // 16) uint8
    N_padded = ((N + 127) // 128) * 128
    b_scale = torch.ones(num_experts, N_padded, K // block_size, dtype=torch.uint8, device=device)
    # alpha: per-expert scale
    alpha = torch.ones(num_experts, dtype=torch.float32, device=device)

    print(f"  a: {list(a.shape)}, b: {list(b.shape)}")
    print(f"  a_scale: {list(a_scale.shape)}, b_scale: {list(b_scale.shape)}")
    print(f"  total_padded_tokens: {total_padded}")

    for _ in range(warmup):
        out = flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr,
            alpha=alpha, tile_n=128, out_dtype=torch.bfloat16)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        out = flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            a, b, a_scale, b_scale, m_indptr,
            alpha=alpha, tile_n=128, out_dtype=torch.bfloat16)
    end_event.record()
    torch.cuda.synchronize(device)

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters
    flops_per_op = 2.0 * total_dispatch * N * K
    tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

    print_result(backend_name, avg_ms, tflops)
    return avg_ms, tflops


# ---------- Helpers ----------
def print_result(backend_name, avg_ms, tflops):
    print()
    print("  " + "-" * 56)
    print(f"  MoE Group GEMM [{num_tokens}*{topk} tokens, {num_experts} experts]")
    print(f"    Shape:        [{K}] x [{N}] per expert (NT layout)")
    print(f"    Backend:      {backend_name}")
    print(f"    Avg latency:  {avg_ms:.3f} ms")
    print(f"    Avg TFLOPS:   {tflops:.2f}")
    print("  " + "-" * 56)


# ---------- Main ----------
if __name__ == '__main__':
    print("=" * 60)
    print("  FlashInfer MoE Quant Group GEMM Benchmark")
    print("=" * 60)
    print()
    print(f"  Config: num_tokens={num_tokens}, K={K}, N={N}")
    print(f"          num_experts={num_experts}, topk={topk}")
    print(f"          dtype={dtype_str}, iters={iters}, warmup={warmup}")
    print()

    dispatch = {
        'bfloat16': bench_bf16,
        'fp8_per_block': bench_fp8_per_block,
        'fp8_per_tensor': bench_fp8_per_tensor,
        'mxfp4': bench_mxfp4,
        'nvfp4': bench_nvfp4,
    }

    avg_ms, tflops = dispatch[dtype_str]()

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  dtype={dtype_str}, num_tokens={num_tokens}, K={K}, N={N}")
    print(f"  num_experts={num_experts}, topk={topk}")
    print(f"  Avg latency:  {avg_ms:.3f} ms")
    print(f"  Avg TFLOPS:   {tflops:.2f}")
    print("=" * 60)
