#!/bin/bash
# GEMM benchmark with real-time GPU frequency monitoring
# Usage: ./bench_gemm_with_monitor.sh [GPU_ID] [DTYPE] [M] [K] [N] [ITERS]
#   GPU_ID: GPU device index (default: 7)
#   DTYPE:  Data type: float16/bfloat16/float32/tfloat32/int8/fp8_e4m3/fp8_e5m2/fp4_e2m1 (default: bfloat16)
#   M/K/N:  Matrix dimensions (defaults based on dtype in bench_gemm.py)
#   ITERS:  Number of iterations (default based on dtype in bench_gemm.py)

GPU_ID=${1:-7}
DTYPE=${2:-bfloat16}
M=${3:-}
K=${4:-}
N=${5:-}
ITERS=${6:-}
SAMPLE_MS=200  # GPU frequency sampling interval in ms
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build python args: gpu_id dtype [M K N [iters [warmup]]]
PYARGS=("$GPU_ID" "$DTYPE")
if [ -n "$M" ] && [ -n "$K" ] && [ -n "$N" ]; then
    PYARGS+=("$M" "$K" "$N")
    [ -n "$ITERS" ] && PYARGS+=("$ITERS")
fi

echo "============================================"
echo " GEMM Benchmark with GPU Frequency Monitor"
echo "============================================"
echo " GPU:    $GPU_ID"
echo " Dtype:  $DTYPE"
[ -n "$M" ] && echo " Shape:  M=$M, K=$K, N=$N" || echo " Shape:  (dtype default)"
[ -n "$ITERS" ] && echo " Iters:  $ITERS" || echo " Iters:  (dtype default)"
echo " Sample: every ${SAMPLE_MS}ms"
echo "============================================"

# Frequency stats (accumulated in-memory)
sm_sum=0; sm_count=0; sm_min=999999; sm_max=0

# Launch Python benchmark in background subprocess
# Buffer Python output to avoid mixing with monitor
BENCH_OUTPUT="/tmp/bench_output_$$"

echo ""
echo ">>> Starting GEMM benchmark..."
echo ""
python3 "$SCRIPT_DIR/bench_gemm.py" "${PYARGS[@]}" > "$BENCH_OUTPUT" 2>&1 &
BENCH_PID=$!

# Main process: real-time GPU frequency monitoring
while kill -0 "$BENCH_PID" 2>/dev/null; do
    line=$(nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
        --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null)
    [ -z "$line" ] && { sleep "0.$(printf '%03d' $SAMPLE_MS)"; continue; }
    IFS=', ' read -r sm_clk mem_clk gpu_util mem_util pwr temp <<< "$line"
    ts=$(date +%H:%M:%S.%N | cut -c1-12)
    # Accumulate frequency stats when GPU is under load
    if [ "$gpu_util" -gt 0 ] 2>/dev/null; then
        sm_sum=$((sm_sum + sm_clk))
        sm_count=$((sm_count + 1))
        [ "$sm_clk" -lt "$sm_min" ] && sm_min=$sm_clk
        [ "$sm_clk" -gt "$sm_max" ] && sm_max=$sm_clk
    fi
    printf "  [Monitor] %s | SM: %s MHz | Mem: %s MHz | Util: %s%% | Power: %sW | Temp: %s°C\n" \
        "$ts" "$sm_clk" "$mem_clk" "$gpu_util" "$pwr" "$temp"
    sleep "0.$(printf '%03d' $SAMPLE_MS)"
done

# Wait for benchmark to finish
wait "$BENCH_PID"

# Clear the monitor's \r line remnant, then print benchmark results
echo ""
echo ""
cat "$BENCH_OUTPUT"
rm -f "$BENCH_OUTPUT"

# Print frequency summary
echo "=== GPU Frequency Summary (during benchmark) ==="
if [ "$sm_count" -gt 0 ]; then
    sm_avg=$((sm_sum / sm_count))
    echo "  SM Clock:  avg=${sm_avg} MHz, min=${sm_min} MHz, max=${sm_max} MHz (${sm_count} samples)"
    # Estimated theoretical peak at measured avg frequency
    # 5090D: 680 TCs * 128 OPs/TC = 87040 OPs/cycle, peak = 87040 * freq
    est_peak=$(echo "$sm_avg" | awk '{printf "%.1f", 680 * ($1/1000) * 128 / 1000}')
    echo "  Estimated peak @ avg freq: ${est_peak} TFLOPS"
else
    echo "  No frequency data collected (benchmark may have been too short)"
fi
echo "============================================"
