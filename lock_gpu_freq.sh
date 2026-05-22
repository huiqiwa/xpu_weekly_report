#!/bin/bash
# Lock GPU SM clock to maximum frequency for stable benchmarking
# Usage: ./lock_gpu_freq.sh [OPTIONS] [GPU_IDs]
#   GPU_IDs: comma-separated GPU indices (default: all GPUs)
#   --lock, -l:    Lock frequency to max (default action)
#   --unlock, -u:  Unlock frequency (restore auto boost)
#   --status, -s:  Show current frequency settings

ACTION="lock"
if [ "$1" = "--unlock" ] || [ "$1" = "-u" ]; then
    ACTION="unlock"
    shift
elif [ "$1" = "--status" ] || [ "$1" = "-s" ]; then
    ACTION="status"
    shift
elif [ "$1" = "--lock" ] || [ "$1" = "-l" ]; then
    ACTION="lock"
    shift
fi

# Use sudo for non-root users
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

GPU_IDS=${1:-$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//')}

case "$ACTION" in
    status)
        echo "=== GPU Frequency Status ==="
        nvidia-smi --query-gpu=index,name,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,power.draw,power.limit,persistence_mode --format=csv -i "$GPU_IDS"
        echo ""
        echo "=== Clock Policy (GPU 0) ==="
        nvidia-smi -i 0 -q -d CLOCK 2>/dev/null | grep -A 5 "Clock Policy"
        echo ""
        echo "=== Locked Clocks (GPU 0) ==="
        nvidia-smi -i 0 -q -d CLOCK 2>/dev/null | grep -A 10 "Clocks$" | head -12
        echo ""
        echo "=== GPU/Memory Clock Limits ==="
        nvidia-smi -i 0 -q 2>/dev/null | grep -iE "lock|clamp|limit|enforced" | head -10 || true
        nvidia-smi -i 0 -q -d SUPPORTED_CLOCKS 2>/dev/null | head -5
        ;;
    unlock)
        echo "Unlocking GPU frequency for GPU: $GPU_IDS"
        $SUDO nvidia-smi -i "$GPU_IDS" -rgc
        $SUDO nvidia-smi -i "$GPU_IDS" -rmc
        echo "GPU frequency unlocked (auto boost restored)"
        ;;
    lock)
        # Query max SM clock
        MAX_GFX=$(nvidia-smi -i 0 --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | tr -d ' ')
        MAX_MEM=$(nvidia-smi -i 0 --query-gpu=clocks.max.memory --format=csv,noheader,nounits | tr -d ' ')

        echo "Locking GPU frequency for GPU: $GPU_IDS"
        echo "  Max SM clock:  ${MAX_GFX} MHz"
        echo "  Max Mem clock: ${MAX_MEM} MHz"

        # Enable persistence mode
        $SUDO nvidia-smi -i "$GPU_IDS" -pm 1

        # Lock SM clock to max
        $SUDO nvidia-smi -i "$GPU_IDS" -lgc "$MAX_GFX","$MAX_GFX"

        # Lock memory clock to max
        $SUDO nvidia-smi -i "$GPU_IDS" -lmc "$MAX_MEM","$MAX_MEM"

        # Set CPU governor to performance
        echo ""
        echo "Setting CPU governor to performance..."
        $SUDO sh -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null'
        echo "  CPU governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"

        echo ""
        echo "GPU frequency locked. Verifying:"
        nvidia-smi --query-gpu=index,clocks.current.graphics,clocks.current.memory --format=csv -i "$GPU_IDS"
        ;;
esac
