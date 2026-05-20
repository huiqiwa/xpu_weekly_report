#!/usr/bin/env python3
"""Flash Attention 2 benchmark kernel. Called by bench_fa2_with_monitor.sh"""
import torch
import sys

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
except ImportError:
    print('Error: flash_attn not installed. Install with: pip install flash-attn')
    sys.exit(1)

# Default (batch, seq_len, num_heads, num_kv_heads, head_dim, iters, warmup) per mode
default_configs = {
    'prefill': {
        'float16':  (2, 32768, 128, 8, 64, 50, 10),
        'bfloat16': (2, 32768, 128, 8, 64, 50, 10),
    },
    'decode': {
        'float16':  (32, 1, 32, 8, 128, 3000, 500),
        'bfloat16': (32, 1, 32, 8, 128, 3000, 500),
    },
}

# --- Parse arguments ---
# Usage: bench_fa2.py [GPU_ID] [MODE] [DTYPE] [BATCH] [SEQ_LEN] [NUM_HEADS] [NUM_KV_HEADS] [HEAD_DIM] [ITERS] [WARMUP]
gpu_id       = int(sys.argv[1]) if len(sys.argv) > 1 else 7
mode         = sys.argv[2] if len(sys.argv) > 2 else 'prefill'       # prefill / decode
dtype_str    = sys.argv[3] if len(sys.argv) > 3 else 'bfloat16'

if mode not in default_configs:
    print(f'Error: unsupported mode "{mode}". Supported: {", ".join(default_configs.keys())}')
    sys.exit(1)
if dtype_str not in default_configs[mode]:
    print(f'Error: unsupported dtype "{dtype_str}" for mode "{mode}". Supported: {", ".join(default_configs[mode].keys())}')
    sys.exit(1)

defaults     = default_configs[mode][dtype_str]
batch        = int(sys.argv[4]) if len(sys.argv) > 4 else defaults[0]
seq_len      = int(sys.argv[5]) if len(sys.argv) > 5 else defaults[1]
num_heads    = int(sys.argv[6]) if len(sys.argv) > 6 else defaults[2]
num_kv_heads = int(sys.argv[7]) if len(sys.argv) > 7 else defaults[3]
head_dim     = int(sys.argv[8]) if len(sys.argv) > 8 else defaults[4]
iters        = int(sys.argv[9]) if len(sys.argv) > 9 else defaults[5]
warmup       = int(sys.argv[10]) if len(sys.argv) > 10 else defaults[6]

dtype_map = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}
dtype = dtype_map[dtype_str]

device = torch.device(f'cuda:{gpu_id}')
torch.cuda.set_device(device)

if mode == 'prefill':
    # Prefill: full sequence attention
    # q/k/v shape: [batch, seq_len, num_heads/num_kv_heads, head_dim]
    print(f'Mode: prefill (causal)')
    print(f'Allocating tensors: Q[{batch}x{seq_len}x{num_heads}x{head_dim}]  '
          f'K/V[{batch}x{seq_len}x{num_kv_heads}x{head_dim}] ({dtype_str})')

    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    def attn_fn():
        return flash_attn_func(q, k, v, causal=True)

    # FLOPS for causal attention (lower triangle only):
    #   QK^T: 2 * batch * num_heads * seq_len^2/2 * head_dim  (causal halves it)
    #   PV:   2 * batch * num_heads * seq_len^2/2 * head_dim
    #   Total: 2 * batch * num_heads * seq_len^2 * head_dim
    flops_per_op = 2.0 * batch * num_heads * seq_len * seq_len * head_dim

elif mode == 'decode':
    # Decode with KV cache (no paged, simple linear cache)
    # q shape: [batch, 1, num_heads, head_dim]
    # k_cache/v_cache shape: [batch, max_cache_len, num_kv_heads, head_dim]
    max_cache_len = 4096  # simulated KV cache length
    print(f'Mode: decode (cache_len={max_cache_len})')
    print(f'Allocating tensors: Q[{batch}x{seq_len}x{num_heads}x{head_dim}]  '
          f'KV_cache[{batch}x{max_cache_len}x{num_kv_heads}x{head_dim}] ({dtype_str})')

    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(batch, max_cache_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn(batch, max_cache_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    cache_seqlens = torch.full((batch,), max_cache_len, dtype=torch.int32, device=device)

    def attn_fn():
        return flash_attn_with_kvcache(q, k_cache, v_cache,
                                       cache_seqlens=cache_seqlens,
                                       causal=True)

    # FLOPS for decode: each query token attends to max_cache_len keys
    #   QK^T: 2 * batch * num_heads * seq_len * max_cache_len * head_dim
    #   PV:   2 * batch * num_heads * seq_len * max_cache_len * head_dim
    #   Total: 4 * batch * num_heads * seq_len * max_cache_len * head_dim
    flops_per_op = 4.0 * batch * num_heads * seq_len * max_cache_len * head_dim

# Warmup
print(f'Warmup ({warmup} iters)...')
for _ in range(warmup):
    out = attn_fn()
torch.cuda.synchronize(device)

# Benchmark
print(f'Benchmarking ({iters} iters)...')
torch.cuda.synchronize(device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(iters):
    out = attn_fn()
end_event.record()
torch.cuda.synchronize(device)

total_ms = start_event.elapsed_time(end_event)
avg_ms = total_ms / iters
tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12

print()
print('=' * 60)
if mode == 'prefill':
    print(f'  FA2 Prefill Results (causal)')
    print(f'  B={batch}, S={seq_len}, H={num_heads}, KVH={num_kv_heads}, D={head_dim}  {dtype_str}')
else:
    print(f'  FA2 Decode Results (cache_len={max_cache_len})')
    print(f'  B={batch}, Q=1, H={num_heads}, KVH={num_kv_heads}, D={head_dim}  {dtype_str}')
print('=' * 60)
print(f'  Avg latency:  {avg_ms:.3f} ms')
print(f'  Avg TFLOPS:   {tflops:.2f}')
print(f'  FLOPS/op:     {flops_per_op:.2e}')
print('=' * 60)
