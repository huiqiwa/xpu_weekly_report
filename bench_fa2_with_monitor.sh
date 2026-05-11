#!/bin/bash
# Flash Attention 2 benchmark with real-time GPU frequency monitoring
# Usage: ./bench_fa2_with_monitor.sh [GPU_ID] [MODE] [DTYPE] [BATCH] [SEQ_LEN] [NUM_HEADS] [NUM_KV_HEADS] [HEAD_DIM] [ITERS]
#   GPU_ID:      GPU device index (default: 7)
#   MODE:        prefill / decode (default: prefill)
#   DTYPE:       float16 / bfloat16 (default: bfloat16)
#   BATCH:       Batch size (default: mode-dependent)
#   SEQ_LEN:     Sequence length (default: mode-dependent)
#   NUM_HEADS:   Number of query heads (default: 32)
#   NUM_KV_HEADS: Number of KV heads (default: mode-dependent)
#   HEAD_DIM:    Head dimension (default: 128)
#   ITERS:       Number of iterations (default: mode-dependent)

GPU_ID=${1:-7}
MODE=${2:-prefill}
DTYPE=${3:-bfloat16}
BATCH=${4:-}
SEQ_LEN=${5:-}
NUM_HEADS=${6:-}
NUM_KV_HEADS=${7:-}
HEAD_DIM=${8:-}
ITERS=${9:-}
SAMPLE_MS=200  # GPU frequency sampling interval in ms
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build python args
PYARGS=("$GPU_ID" "$MODE" "$DTYPE")
if [ -n "$BATCH" ]; then
    PYARGS+=("$BATCH")
    [ -n "$SEQ_LEN" ] && PYARGS+=("$SEQ_LEN")
    [ -n "$NUM_HEADS" ] && PYARGS+=("$NUM_HEADS")
    [ -n "$NUM_KV_HEADS" ] && PYARGS+=("$NUM_KV_HEADS")
    [ -n "$HEAD_DIM" ] && PYARGS+=("$HEAD_DIM")
    [ -n "$ITERS" ] && PYARGS+=("$ITERS")
fi

echo "============================================"
echo " FA2 Benchmark with GPU Frequency Monitor"
echo "============================================"
echo " GPU:      $GPU_ID"
echo " Mode:     $MODE"
echo " Dtype:    $DTYPE"
[ -n "$BATCH" ] && echo " Batch:    $BATCH" || echo " Batch:    (mode default)"
[ -n "$SEQ_LEN" ] && echo " SeqLen:   $SEQ_LEN" || echo " SeqLen:   (mode default)"
[ -n "$NUM_HEADS" ] && echo " Heads:    $NUM_HEADS" || echo " Heads:    (mode default)"
[ -n "$NUM_KV_HEADS" ] && echo " KV Heads: $NUM_KV_HEADS" || echo " KV Heads: (mode default)"
[ -n "$HEAD_DIM" ] && echo " HeadDim:  $HEAD_DIM" || echo " HeadDim:  (mode default)"
[ -n "$ITERS" ] && echo " Iters:    $ITERS" || echo " Iters:    (mode default)"
echo " Sample:   every ${SAMPLE_MS}ms"
echo "============================================"

# Frequency stats (accumulated in-memory)
sm_sum=0; sm_count=0; sm_min=999999; sm_max=0

# Launch Python benchmark in background subprocess
BENCH_OUTPUT="/tmp/bench_fa2_output_$$"

echo ""
echo ">>> Starting FA2 benchmark..."
echo ""
python3 "$SCRIPT_DIR/bench_fa2.py" "${PYARGS[@]}" > "$BENCH_OUTPUT" 2>&1 &
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

# Print benchmark results
echo ""
echo ""
cat "$BENCH_OUTPUT"
rm -f "$BENCH_OUTPUT"

# Print frequency summary
echo "=== GPU Frequency Summary (during benchmark) ==="
if [ "$sm_count" -gt 0 ]; then
    sm_avg=$((sm_sum / sm_count))
    echo "  SM Clock:  avg=${sm_avg} MHz, min=${sm_min} MHz, max=${sm_max} MHz (${sm_count} samples)"
else
    echo "  No frequency data collected (benchmark may have been too short)"
fi
echo "============================================"
