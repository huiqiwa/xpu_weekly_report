#!/bin/bash
# GEMM micro benchmark: grid search peak TFLOPS for 7 data types
# Usage: ./gemm_micro_test.sh [GPU_ID]

set -e

GPU_ID=${1:-7}
WARMUP=2
ITERS=10
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# M values to sweep
M_LIST=(1024 2048 3072 4096 8192 12288 16384 20480 24576 28672 32768)
# (K, N) pairs to test
KN_LIST="1024,8192 4096,4096 8192,1024 8192,8192"
# All dtypes to benchmark
DTYPES="float32 tfloat32 bfloat16 float16 int8 fp8_e4m3 nvfp4"

echo "============================================"
echo " GEMM Grid Search — Peak TFLOPS"
echo "============================================"
echo " GPU:     $GPU_ID"
echo " Warmup:  $WARMUP"
echo " Iters:   $ITERS"
echo " Dtypes:  $DTYPES"
echo " M list:  ${M_LIST[*]}"
echo " KxN:     $KN_LIST"
echo "============================================"
echo ""

for dtype in $DTYPES; do
    best_tflops=0
    best_shape=""

    echo "============================================"
    echo " Sweeping dtype: $dtype"
    echo "============================================"

    for kn in $KN_LIST; do
        K=${kn%%,*}
        N=${kn##*,}
        for M in "${M_LIST[@]}"; do
            echo -n "  ${dtype} M=${M}, K=${K}, N=${N} => "
            output=$(python3 "$SCRIPT_DIR/bench_gemm.py" "$GPU_ID" "$dtype" "$M" "$K" "$N" "$ITERS" "$WARMUP" 2>&1)
            # Extract TFLOPS from output line "  Avg TFLOPS:   xx.xx"
            tflops=$(echo "$output" | grep -oP 'Avg TFLOPS:\s+\K[0-9.]+' || echo "0")
            latency=$(echo "$output" | grep -oP 'Avg latency:\s+\K[0-9.]+' || echo "N/A")
            echo "TFLOPS=${tflops}, latency=${latency} ms"

            # Track best (compare as integers scaled x100 to avoid bash float issues)
            tflops_x100=$(echo "$tflops" | awk '{printf "%d", $1 * 100}')
            best_x100=$(echo "$best_tflops" | awk '{printf "%d", $1 * 100}')
            if [ "$tflops_x100" -gt "$best_x100" ]; then
                best_tflops=$tflops
                best_shape="M=${M}, K=${K}, N=${N}"
            fi
        done
    done

    echo "--------------------------------------------"
    echo " >>> ${dtype} BEST: ${best_tflops} TFLOPS @ ${best_shape}"
    echo "--------------------------------------------"
    echo ""
done

echo "============================================"
echo " All grid searches completed."
echo "============================================"
