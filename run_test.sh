SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE=0,1
CCL_DEVICE=0,1
REPORT_DIR="$(realpath -m "${1:-${SCRIPT_DIR}/reports/reports_$(TZ='Asia/Shanghai' date +%Y-%m-%d-%H-%M-%S)}")"

export RenderCompressedBuffersEnabled=0 
export NEOReadDebugKeys=1
export CCL_SYCL_CCL_BARRIER=1

XPU_PERF_DIR="${SCRIPT_DIR}/../xpu-perf"

source "$SCRIPT_DIR/activate_env.sh"

# Prepare xpu-perf: clean, update, patch, build extensions
bash "${SCRIPT_DIR}/prepare_xpu_perf.sh"

mkdir -p "$REPORT_DIR"

cd "${XPU_PERF_DIR}/micro_perf"

# python launch.py --task_dir workloads/basic --device 0,1,2,3 --backend INTEL --task all --report_dir all_reports  &> /yupengzh/xpu-perf-logs/basic.txt

# python launch.py --task_dir workloads/llm --device 0,1,2,3 --backend INTEL --task all --report_dir all_reports  &> /yupengzh/xpu-perf-logs/llm.txt

HANG_TIMEOUT=300  # seconds (5 minutes)

_kill_xpu_orphans() {
    local devices=$1  # e.g. "0,1,2,3,4,5,6,7"
    # Get PIDs occupying specified XPU devices (exclude xpu-smi itself)
    local pids
    pids=$(sudo xpu-smi ps 2>/dev/null | awk -v devs=",$devices," '
        NR==1 {next}
        {
            pid=$1; dev=$3
            if (pid == "xpu-smi" || $2 == "xpu-smi") next
            if (index(devs, "," dev ",") > 0) seen[pid]=1
        }
        END { for (p in seen) print p }
    ')
    if [[ -n "$pids" ]]; then
        echo "[CLEANUP] Killing processes on XPU device(s) $devices: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null
        sleep 2
    fi
}

_run_with_timeout() {
    local op_name=$1
    local log_file="$REPORT_DIR/$op_name.txt"
    shift
    "$@" &> "$log_file" &
    local pid=$!
    local last_size=-1
    local stall_start=""

    while kill -0 "$pid" 2>/dev/null; do
        sleep 10
        local cur_size
        cur_size=$(stat -c '%s' "$log_file" 2>/dev/null || echo 0)
        if [[ "$cur_size" == "$last_size" ]]; then
            if [[ -z "$stall_start" ]]; then
                stall_start=$(date +%s)
            fi
            local now=$(date +%s)
            if (( now - stall_start >= HANG_TIMEOUT )); then
                echo "[HANG] $op_name: no output for ${HANG_TIMEOUT}s, killing (pid=$pid)"
                kill -9 "$pid" 2>/dev/null
                # Kill any orphan child processes
                pkill -9 -P "$pid" 2>/dev/null
                wait "$pid" 2>/dev/null
                echo "[HANG] killed" >> "$log_file"
                return 1
            fi
        else
            stall_start=""
            last_size=$cur_size
        fi
    done
    wait "$pid"
}

run_test() {
    local op_name=$1
    if ls -d "$REPORT_DIR"/INTEL/*/$op_name &>/dev/null; then
        echo "[SKIP] $op_name: result already exists"
        return
    fi
    _kill_xpu_orphans "$DEVICE"
    _run_with_timeout "$op_name" python launch.py --task_dir workloads --device $DEVICE --backend INTEL --task $op_name --report_dir $REPORT_DIR
    sleep 10
}

run_ccl_test() {
    local op_name=$1
    if ls -d "$REPORT_DIR"/INTEL/*/$op_name &>/dev/null; then
        echo "[SKIP] $op_name: result already exists"
        return
    fi
    _kill_xpu_orphans "$CCL_DEVICE"
    _run_with_timeout "$op_name" python launch.py --task_dir workloads --device $CCL_DEVICE --backend INTEL --task $op_name --report_dir $REPORT_DIR
    sleep 30
}

run_all() {

# Norm & Quant
run_test scale_dynamic_quant
run_test head_rms_norm
run_test head_rms_norm_dynamic_quant
run_test add_rms_norm_dynamic_quant

# Attention & rope & kvcache 
run_test rotary_embedding
run_test multimodal_rotary_embedding
run_test store_kv_cache
run_test dequant_kv_cache
run_test flash_attention


# gemm & group_gemm & moe_ops
run_test moe_gating_gemm
run_test quant_matmul
run_test moe_quant_group_gemm
run_test moe_softmax_topk
run_test moe_scatter_dynamic_quant
run_test moe_swiglu_dynamic_quant
run_test swiglu_dynamic_quant
run_test moe_gather

# tensor_gemm_ops
run_test gemm

# vector_activation_ops
run_test gelu
run_test silu

# vector_index_ops
run_test embedding
run_test gather
run_test index_add
run_test index_select
run_test scatter

# vector_linear_ops
run_test add
run_test cast
run_test mul
run_test sub

# vector_norm_ops
run_test layer_norm
run_test rms_norm
run_test softmax

# vector_reduction_ops
run_test reduce_max
run_test reduce_min
run_test reduce_sum
run_test topk

# vector_sfu_ops
run_test cos
run_test div
run_test exp
run_test log
run_test sin
run_test sqrt

# sage_attention
run_test sage_attention_page
run_test sage_attention_decode_page
run_test sage_attention_v1

# xccl ops
run_ccl_test device2device
run_ccl_test device2host
run_ccl_test host2device
sleep 30
run_ccl_test all_reduce
run_ccl_test all_gather
run_ccl_test reduce_scatter
run_ccl_test all_to_all

}

run_all
