#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${1:?Usage: $0 <output_dir>}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/commit-info.txt"

declare -A PROJECTS=(
    ["xpu-perf"]="$WORKSPACE_DIR/xpu-perf"
    ["oneDNN"]="$WORKSPACE_DIR/oneDNN"
    ["vllm-xpu-kernels"]="$WORKSPACE_DIR/vllm-xpu-kernels"
    ["sycl-tla"]="$WORKSPACE_DIR/sycl-tla"
    ["ipex"]="$WORKSPACE_DIR/intel-extension-for-pytorch"
    ["auto-round"]="$WORKSPACE_DIR/frameworks.ai.lpot.auto-round"
)

PROJECT_ORDER=("xpu-perf" "oneDNN" "vllm-xpu-kernels" "sycl-tla" "ipex" "auto-round")

: > "$OUTPUT_FILE"

echo "=== Commit Info (collected at $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S %Z')) ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

for name in "${PROJECT_ORDER[@]}"; do
    dir="${PROJECTS[$name]}"
    echo "--- $name ---" >> "$OUTPUT_FILE"

    if [ ! -d "$dir/.git" ]; then
        echo "  [WARN] Not a git repository: $dir" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        continue
    fi

    branch=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")
    commit=$(git -C "$dir" rev-parse HEAD 2>/dev/null || echo "N/A")
    commit_short=$(git -C "$dir" rev-parse --short HEAD 2>/dev/null || echo "N/A")
    commit_date=$(git -C "$dir" log -1 --format='%ci' 2>/dev/null || echo "N/A")
    commit_msg=$(git -C "$dir" log -1 --format='%s' 2>/dev/null || echo "N/A")
    remote_url=$(git -C "$dir" remote get-url origin 2>/dev/null || echo "N/A")
    # Mask credentials in URL (e.g. https://token@github.com/...)
    remote_url=$(echo "$remote_url" | sed -E 's#(https?://)([^@]+)@#\1***@#')

    echo "  Branch:      $branch" >> "$OUTPUT_FILE"
    echo "  Commit:      $commit_short ($commit)" >> "$OUTPUT_FILE"
    echo "  Date:        $commit_date" >> "$OUTPUT_FILE"
    echo "  Message:     $commit_msg" >> "$OUTPUT_FILE"
    echo "  Remote:      $remote_url" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "Commit info written to: $OUTPUT_FILE"