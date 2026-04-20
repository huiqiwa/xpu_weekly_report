#!/bin/bash
# GEMM benchmark with real-time GPU frequency monitoring
# Usage: ./bench_gemm_with_monitor.sh [GPU_ID] [M] [K] [N] [DTYPE] [ITERS]
#   GPU_ID: GPU device index (default: 7)
#   M:      Matrix M dimension (default: 4096)
#   K:      Matrix K dimension (default: 4096)
#   N:      Matrix N dimension (default: 4096)
#   DTYPE:  Data type: float16/bfloat16/float32/tfloat32 (default: bfloat16)
#   ITERS:  Number of iterations (default: 200)

GPU_ID=${1:-7}
M=${2:-4096}
K=${3:-8192}
N=${4:-8192}
DTYPE=${5:-bfloat16}
ITERS=${6:-3000}
WARMUP=500
SAMPLE_MS=200  # GPU frequency sampling interval in ms

echo "============================================"
echo " GEMM Benchmark with GPU Frequency Monitor"
echo "============================================"
echo " GPU:    $GPU_ID"
echo " Shape:  M=$M, K=$K, N=$N"
echo " Dtype:  $DTYPE"
echo " Iters:  $ITERS (warmup: $WARMUP)"
echo " Sample: every ${SAMPLE_MS}ms"
echo "============================================"

# Temp file to collect SM clocks for summary (only clocks during load)
FREQ_TMP=$(mktemp /tmp/gpu_freq_XXXXXX)
# Marker file: benchmark sets this when compute starts
BENCH_RUNNING="/tmp/gpu_bench_running_$$"

# Start GPU frequency monitor in background (200ms sampling via nvidia-smi query loop)
(
    while true; do
        line=$(nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
            --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null)
        [ -z "$line" ] && continue
        IFS=', ' read -r sm_clk mem_clk gpu_util mem_util pwr temp <<< "$line"
        ts=$(date +%H:%M:%S.%N | cut -c1-12)
        # Only record frequency samples when benchmark is running
        if [ -f "$BENCH_RUNNING" ] && [ "$gpu_util" -gt 0 ] 2>/dev/null; then
            echo "$sm_clk" >> "$FREQ_TMP"
        fi
        printf "\r  [Monitor] %s | SM: %s MHz | Mem: %s MHz | Util: %s%% | Power: %sW | Temp: %s°C  " \
            "$ts" "$sm_clk" "$mem_clk" "$gpu_util" "$pwr" "$temp"
        sleep "0.$(printf '%03d' $SAMPLE_MS)"
    done
) &
MONITOR_PID=$!

# Give monitor a moment to start
sleep 0.5

# Run GEMM benchmark
echo ""
echo ">>> Starting GEMM benchmark..."
echo ""

python3 -c "
import torch
import time
import sys
import os

gpu_id = int(sys.argv[1])
M, K, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dtype_str = sys.argv[5]
iters = int(sys.argv[6])
warmup = int(sys.argv[7])
marker = sys.argv[8]

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

# Signal monitor: benchmark compute starts now
open(marker, 'w').close()

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

# Per-iteration timing
per_iter_events = []
for i in range(min(iters, 50)):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    c = torch.matmul(a, b)
    e.record()
    torch.cuda.synchronize(device)
    per_iter_events.append(s.elapsed_time(e))

# Remove marker
try:
    os.remove(marker)
except OSError:
    pass

min_ms = min(per_iter_events)
max_ms = max(per_iter_events)
peak_tflops = flops_per_op / (min_ms / 1000.0) / 1e12

print()
print('=' * 50)
print(f'  GEMM Results: [{M} x {K}] x [{K} x {N}] {dtype_str}')
print('=' * 50)
print(f'  Avg latency:  {avg_ms:.3f} ms')
print(f'  Min latency:  {min_ms:.3f} ms')
print(f'  Max latency:  {max_ms:.3f} ms')
print(f'  Avg TFLOPS:   {tflops:.2f}')
print(f'  Peak TFLOPS:  {peak_tflops:.2f}')
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
" "$GPU_ID" "$M" "$K" "$N" "$DTYPE" "$ITERS" "$WARMUP" "$BENCH_RUNNING"

# Stop monitor
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null
sleep 0.3

# Clear the monitor's \r line remnant
echo ""
echo ""

# Print frequency summary (only samples collected during benchmark)
echo "=== GPU Frequency Summary (during benchmark) ==="
if [ -f "$FREQ_TMP" ] && [ -s "$FREQ_TMP" ]; then
    sm_sum=0; sm_count=0; sm_min=999999; sm_max=0
    while read -r val; do
        [ "$val" -gt 0 ] 2>/dev/null || continue
        sm_sum=$((sm_sum + val))
        sm_count=$((sm_count + 1))
        [ "$val" -lt "$sm_min" ] && sm_min=$val
        [ "$val" -gt "$sm_max" ] && sm_max=$val
    done < "$FREQ_TMP"
    if [ "$sm_count" -gt 0 ]; then
        sm_avg=$((sm_sum / sm_count))
        echo "  SM Clock:  avg=${sm_avg} MHz, min=${sm_min} MHz, max=${sm_max} MHz (${sm_count} samples)"
        # Estimated theoretical peak at measured avg frequency
        # 5090D: 680 TCs * 128 OPs/TC = 87040 OPs/cycle, peak = 87040 * freq
        est_peak=$(echo "$sm_avg" | awk '{printf "%.1f", 680 * ($1/1000) * 128 / 1000}')
        echo "  Estimated peak @ avg freq: ${est_peak} TFLOPS"
    fi
else
    echo "  No frequency data collected (benchmark may have been too short)"
fi
rm -f "$FREQ_TMP" "$BENCH_RUNNING"
echo "============================================"
