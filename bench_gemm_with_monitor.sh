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
ITERS=${6:-200}
WARMUP=50

echo "============================================"
echo " GEMM Benchmark with GPU Frequency Monitor"
echo "============================================"
echo " GPU:    $GPU_ID"
echo " Shape:  M=$M, K=$K, N=$N"
echo " Dtype:  $DTYPE"
echo " Iters:  $ITERS (warmup: $WARMUP)"
echo "============================================"

# Frequency log file
FREQ_LOG="/tmp/gpu_freq_monitor_${GPU_ID}_$(date +%Y%m%d_%H%M%S).csv"

# Start GPU frequency monitor in background
echo "timestamp,sm_clk_mhz,mem_clk_mhz,gpu_util,mem_util,power_w,temp_c" > "$FREQ_LOG"
nvidia-smi dmon -i "$GPU_ID" -s cput -d 1 2>/dev/null | \
    awk -v logfile="$FREQ_LOG" '
    BEGIN { OFS="," }
    /^#/ { next }
    NF >= 7 {
        # dmon -s cput columns: idx mclk pclk pwr gtemp mtemp sm mem enc dec jpg ofa rxpci txpci
        sm_clk = $3    # pclk (SM/processor clock)
        mem_clk = $2   # mclk (memory clock)
        pwr = $4
        temp = $5
        sm_util = $7
        mem_util = $8
        cmd = "date +%H:%M:%S"
        cmd | getline ts
        close(cmd)
        printf "%s,%s,%s,%s,%s,%s,%s\n", ts, sm_clk, mem_clk, sm_util, mem_util, pwr, temp >> logfile
        fflush(logfile)
        printf "\r  [Monitor] SM: %s MHz | Mem: %s MHz | Util: %s%% | Power: %sW | Temp: %s°C  ", sm_clk, mem_clk, sm_util, pwr, temp
    }
    ' &
MONITOR_PID=$!

# Give monitor a moment to start
sleep 1

# Run GEMM benchmark
echo ""
echo ">>> Starting GEMM benchmark..."
echo ""

python3 -c "
import torch
import time
import sys

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

latencies = []
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
# We'll print a reference
print()
print('  Reference (5090D BF16 theoretical peak):')
print('    @ 2.407 GHz (boost):  209.5 TFLOPS')
print('    @ 3.090 GHz (max):    268.6 TFLOPS')
print(f'  Efficiency vs boost:    {tflops/209.5*100:.1f}%')
print(f'  Efficiency vs max clk:  {tflops/268.6*100:.1f}%')
print()
" "$GPU_ID" "$M" "$K" "$N" "$DTYPE" "$ITERS" "$WARMUP"

# Stop monitor
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null
sleep 0.5

# Clear the monitor's \r line remnant, then print log path
echo ""
echo ""
echo ">>> Frequency log saved to: $FREQ_LOG"

# Print frequency summary
echo ""
echo "=== GPU Frequency Summary ==="
if [ -f "$FREQ_LOG" ]; then
    awk -F',' 'NR>1 && $2+0 > 0 {
        sum+=$2; count++
        if($2+0 > max) max=$2+0
        if(min==0 || $2+0 < min) min=$2+0
    } END {
        if(count>0) {
            printf "  SM Clock:  avg=%.0f MHz, min=%d MHz, max=%d MHz (%d samples)\n", sum/count, min, max, count
            printf "  Estimated peak @ avg freq: %.1f TFLOPS\n", 680 * (sum/count/1000) * 128 / 1000
        }
    }' "$FREQ_LOG"
fi
echo "============================================"
