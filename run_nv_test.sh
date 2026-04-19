SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
XPU_PERF_DIR="${SCRIPT_DIR}/../xpu-perf"
cd "$XPU_PERF_DIR/micro_perf"

TIME=$(TZ=Asia/Shanghai date +%Y-%m-%d-%H-%M-%S)
REPORT_DIR="$SCRIPT_DIR/nv_reports/reports_$TIME"

mkdir -p "$REPORT_DIR"

python launch.py --device 0,1,2,3,4,5,6,7 \
    --task_dir workloads \
    --backend GPU \
    --task all \
    --report_dir "$REPORT_DIR" &> "$REPORT_DIR/logs_$TIME.txt"