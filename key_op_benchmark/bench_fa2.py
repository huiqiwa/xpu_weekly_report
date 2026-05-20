#!/usr/bin/env python3
"""Flash Attention 2 benchmark kernel. Called by fa_micro_test.py or bench_fa2_with_monitor.sh"""
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
# Usage: bench_fa2.py [GPU_ID] [MODE] [DTYPE] [BATCH] [SEQ_LEN] [NUM_HEADS] [NUM_KV_HEADS] [HEAD_DIM] [ITERS] [WARMUP] [CACHE_LEN]
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
cache_len    = int(sys.argv[11]) if len(sys.argv) > 11 else 0

dtype_map = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}
dtype = dtype_map[dtype_str]
elem_size = 2  # bytes per element for fp16/bf16

device = torch.device(f'cuda:{gpu_id}')
torch.cuda.set_device(device)

total_runs = warmup + iters
MAX_ALLOC_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB limit for tensor pool


def _num_copies(bytes_per_set):
    """Determine how many unique tensor sets to allocate (capped by 1GB)."""
    n = max(1, min(total_runs, MAX_ALLOC_BYTES // max(bytes_per_set, 1)))
    return n


if mode == 'prefill' and cache_len == 0:
    # bytes per set: Q + K + V
    set_bytes = (batch * seq_len * num_heads * head_dim
                 + batch * seq_len * num_kv_heads * head_dim * 2) * elem_size
    n_copies = _num_copies(set_bytes)

    print(f'Mode: prefill (causal, cache_len=0)')
    print(f'Allocating {n_copies} sets (of {total_runs} runs): '
          f'Q[{batch}x{seq_len}x{num_heads}x{head_dim}]  '
          f'K/V[{batch}x{seq_len}x{num_kv_heads}x{head_dim}] ({dtype_str})')

    # Create template then clone
    q_t = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k_t = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_t = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    qs = [q_t.clone() for _ in range(n_copies)]
    ks = [k_t.clone() for _ in range(n_copies)]
    vs = [v_t.clone() for _ in range(n_copies)]
    del q_t, k_t, v_t

    def attn_fn(i):
        idx = i % n_copies
        return flash_attn_func(qs[idx], ks[idx], vs[idx], causal=True)

    flops_per_op = 2.0 * batch * num_heads * seq_len * seq_len * head_dim

    # Memory: read Q + K + V, write O
    bytes_per_op = (batch * seq_len * num_heads * head_dim * 2
                    + batch * seq_len * num_kv_heads * head_dim * 2) * elem_size

elif mode == 'prefill' and cache_len > 0:
    total_len = cache_len + seq_len
    # bytes per set: Q + K_new + V_new + K_cache + V_cache
    set_bytes = (batch * seq_len * num_heads * head_dim
                 + batch * seq_len * num_kv_heads * head_dim * 2
                 + batch * total_len * num_kv_heads * head_dim * 2) * elem_size
    n_copies = _num_copies(set_bytes)

    print(f'Mode: prefill (causal, cache_len={cache_len}, q_len={seq_len})')
    print(f'Allocating {n_copies} sets (of {total_runs} runs): '
          f'Q[{batch}x{seq_len}x{num_heads}x{head_dim}]  '
          f'K/V_new[{batch}x{seq_len}x{num_kv_heads}x{head_dim}]  '
          f'KV_cache[{batch}x{total_len}x{num_kv_heads}x{head_dim}] ({dtype_str})')

    q_t = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    kn_t = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    vn_t = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    kc_t = torch.randn(batch, total_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    vc_t = torch.randn(batch, total_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    qs = [q_t.clone() for _ in range(n_copies)]
    k_news = [kn_t.clone() for _ in range(n_copies)]
    v_news = [vn_t.clone() for _ in range(n_copies)]
    k_caches = [kc_t.clone() for _ in range(n_copies)]
    v_caches = [vc_t.clone() for _ in range(n_copies)]
    del q_t, kn_t, vn_t, kc_t, vc_t
    cache_seqlens = torch.full((batch,), cache_len, dtype=torch.int32, device=device)

    def attn_fn(i):
        idx = i % n_copies
        cache_seqlens.fill_(cache_len)
        return flash_attn_with_kvcache(qs[idx], k_caches[idx], v_caches[idx],
                                       k=k_news[idx], v=v_news[idx],
                                       cache_seqlens=cache_seqlens,
                                       causal=True)

    flops_per_op = 4.0 * batch * num_heads * seq_len * (cache_len + seq_len / 2.0) * head_dim

    # Memory: read Q + K_new + V_new + K_cache(cache_len) + V_cache(cache_len), write O + cache append
    bytes_read = (batch * seq_len * num_heads * head_dim          # Q
                  + batch * seq_len * num_kv_heads * head_dim * 2  # K_new + V_new
                  + batch * cache_len * num_kv_heads * head_dim * 2) * elem_size  # existing cache
    bytes_write = (batch * seq_len * num_heads * head_dim          # O
                   + batch * seq_len * num_kv_heads * head_dim * 2) * elem_size  # cache append
    bytes_per_op = bytes_read + bytes_write

elif mode == 'decode':
    if cache_len == 0:
        cache_len = 4096
    # bytes per set: Q + K_cache + V_cache
    set_bytes = (batch * seq_len * num_heads * head_dim
                 + batch * cache_len * num_kv_heads * head_dim * 2) * elem_size
    n_copies = _num_copies(set_bytes)

    print(f'Mode: decode (cache_len={cache_len})')
    print(f'Allocating {n_copies} sets (of {total_runs} runs): '
          f'Q[{batch}x{seq_len}x{num_heads}x{head_dim}]  '
          f'KV_cache[{batch}x{cache_len}x{num_kv_heads}x{head_dim}] ({dtype_str})')

    q_t = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    kc_t = torch.randn(batch, cache_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    vc_t = torch.randn(batch, cache_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    qs = [q_t.clone() for _ in range(n_copies)]
    k_caches = [kc_t.clone() for _ in range(n_copies)]
    v_caches = [vc_t.clone() for _ in range(n_copies)]
    del q_t, kc_t, vc_t
    cache_seqlens = torch.full((batch,), cache_len, dtype=torch.int32, device=device)

    def attn_fn(i):
        idx = i % n_copies
        return flash_attn_with_kvcache(qs[idx], k_caches[idx], v_caches[idx],
                                       cache_seqlens=cache_seqlens,
                                       causal=True)

    flops_per_op = 4.0 * batch * num_heads * seq_len * cache_len * head_dim

    # Memory: read Q + K_cache + V_cache, write O
    bytes_per_op = (batch * seq_len * num_heads * head_dim          # Q + O (read+write)
                    * 2
                    + batch * cache_len * num_kv_heads * head_dim * 2) * elem_size  # K_cache + V_cache

# Warmup (use indices 0..warmup-1)
print(f'Warmup ({warmup} iters)...')
for i in range(warmup):
    out = attn_fn(i)
torch.cuda.synchronize(device)

# Benchmark (use indices warmup..warmup+iters-1)
print(f'Benchmarking ({iters} iters)...')
torch.cuda.synchronize(device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for i in range(warmup, warmup + iters):
    out = attn_fn(i)
end_event.record()
torch.cuda.synchronize(device)

total_ms = start_event.elapsed_time(end_event)
avg_ms = total_ms / iters
tflops = (flops_per_op * iters) / (total_ms / 1000.0) / 1e12
mem_bw_gbs = (bytes_per_op * iters) / (total_ms / 1000.0) / 1e9

print()
print('=' * 60)
if mode == 'prefill':
    print(f'  FA2 Prefill Results (causal, cache_len={cache_len})')
    print(f'  B={batch}, S={seq_len}, H={num_heads}, KVH={num_kv_heads}, D={head_dim}  {dtype_str}')
else:
    print(f'  FA2 Decode Results (cache_len={cache_len})')
    print(f'  B={batch}, Q={seq_len}, H={num_heads}, KVH={num_kv_heads}, D={head_dim}  {dtype_str}')
print('=' * 60)
print(f'  Avg latency:  {avg_ms:.3f} ms')
print(f'  Avg TFLOPS:   {tflops:.2f}')
print(f'  Mem BW:       {mem_bw_gbs:.2f} GB/s')
print(f'  FLOPS/op:     {flops_per_op:.2e}')
print(f'  Bytes/op:     {bytes_per_op:.2e}')
print('=' * 60)
