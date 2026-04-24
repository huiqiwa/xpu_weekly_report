SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE=0,1
CCL_DEVICE=0,1
REPORT_DIR="$(realpath -m "${1:-${SCRIPT_DIR}/reports/reports_$(TZ='Asia/Shanghai' date +%Y-%m-%d-%H-%M-%S)}")"

export RenderCompressedBuffersEnabled=0 
export NEOReadDebugKeys=1
export CCL_SYCL_CCL_BARRIER=1

XPU_PERF_DIR="${SCRIPT_DIR}/../xpu-perf"

source "$SCRIPT_DIR/activate_ipex_env.sh"

# Prepare xpu-perf: clean, update, patch, build extensions
bash "${SCRIPT_DIR}/prepare_xpu_perf.sh" ipex

mkdir -p "$REPORT_DIR"

cd "${XPU_PERF_DIR}/micro_perf"

# python launch.py --task_dir workloads/basic --device 0,1,2,3 --backend INTEL --task all --report_dir all_reports  &> /yupengzh/xpu-perf-logs/basic.txt

# python launch.py --task_dir workloads/llm --device 0,1,2,3 --backend INTEL --task all --report_dir all_reports  &> /yupengzh/xpu-perf-logs/llm.txt

run_test() {
    op_name=$1
    if ls -d "$REPORT_DIR"/INTEL/*/$op_name &>/dev/null; then
        echo "[SKIP] $op_name: result already exists"
        return
    fi
    python launch.py --task_dir workloads --device $DEVICE --backend INTEL --task $op_name --report_dir $REPORT_DIR &> $REPORT_DIR/$op_name.txt
    sleep 10
}

run_ccl_test() {
    op_name=$1
    if ls -d "$REPORT_DIR"/INTEL/*/$op_name &>/dev/null; then
        echo "[SKIP] $op_name: result already exists"
        return
    fi
    python launch.py --task_dir workloads --device $CCL_DEVICE --backend INTEL --task $op_name --report_dir $REPORT_DIR &> $REPORT_DIR/$op_name.txt
    sleep 30
}

run_all() {

# Norm & Quant
run_test scale_dynamic_quant
run_test head_rms_norm
run_test add_rms_norm_dynamic_quant
run_test rms_norm

# Attention & rope & kvcache
run_test rotary_embedding
run_test store_kv_cache

# gemm & group_gemm & moe_ops
run_test moe_gating_gemm
run_test quant_matmul
run_test moe_quant_group_gemm
run_test moe_softmax_topk
run_test moe_scatter_dynamic_quant
run_test moe_swiglu_dynamic_quant
run_test moe_gather

# sage_attention
run_test sage_attention_page
run_test sage_attention_decode_page

}

run_all
