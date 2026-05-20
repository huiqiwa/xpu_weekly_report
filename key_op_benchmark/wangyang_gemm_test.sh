#!/bin/bash
# GEMM micro benchmark: sweep all M/K/N combinations for bfloat16
# Usage: ./wangyang_gemm_test.sh [GPU_ID]

set -e

GPU_ID=${1:-7}
WARMUP=2
ITERS=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# M, K, N each from this list => 4x4x4 = 64 cases
DIMS=(1024 2048 4096 8192)
DTYPE="bfloat16"

# Get GPU model name
GPU_NAME=$(nvidia-smi --id=$GPU_ID --query-gpu=name --format=csv,noheader 2>/dev/null | xargs)

echo "============================================"
echo " GEMM BF16 Sweep — 64 Cases"
echo "============================================"
echo " GPU ID:   $GPU_ID"
echo " GPU Model: $GPU_NAME"
echo " Warmup:   $WARMUP"
echo " Iters:    $ITERS"
echo " Dtype:    $DTYPE"
echo " Dims:     ${DIMS[*]}"
echo " Total:    64 cases"
echo "============================================"
echo ""

best_tflops=0
best_shape=""
case_idx=0

for M in "${DIMS[@]}"; do
    for K in "${DIMS[@]}"; do
        for N in "${DIMS[@]}"; do
            case_idx=$((case_idx + 1))
            echo -n "  [${case_idx}/64] ${DTYPE} M=${M}, K=${K}, N=${N} => "
            output=$(python3 "$SCRIPT_DIR/bench_gemm.py" "$GPU_ID" "$DTYPE" "$M" "$K" "$N" "$ITERS" "$WARMUP" 2>&1)
            tflops=$(echo "$output" | grep -oP 'Avg TFLOPS:\s+\K[0-9.]+' || echo "0")
            latency=$(echo "$output" | grep -oP 'Avg latency:\s+\K[0-9.]+' || echo "N/A")
            echo "TFLOPS=${tflops}, latency=${latency} ms"

            tflops_x100=$(echo "$tflops" | awk '{printf "%d", $1 * 100}')
            best_x100=$(echo "$best_tflops" | awk '{printf "%d", $1 * 100}')
            if [ "$tflops_x100" -gt "$best_x100" ]; then
                best_tflops=$tflops
                best_shape="M=${M}, K=${K}, N=${N}"
            fi
        done
    done
done

echo ""
echo "============================================"
echo " GPU Model: $GPU_NAME"
echo " BEST: ${best_tflops} TFLOPS @ ${best_shape}"
echo "============================================"
