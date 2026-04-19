SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE=0,1,2,3,4,5,6,7
CCL_DEVICE=4,5
REPORT_DIR="${1:-${SCRIPT_DIR}/reports/reports_$(TZ='Asia/Shanghai' date +%Y-%m-%d-%H-%M-%S)}"

export RenderCompressedBuffersEnabled=0 
export NEOReadDebugKeys=1
export CCL_SYCL_CCL_BARRIER=1

XPU_PERF_DIR="${SCRIPT_DIR}/../xpu-perf"

# Ensure xpu-perf repo is clean and up-to-date
cd "$XPU_PERF_DIR"
git checkout -- .
git clean -fd
git pull --ff-only || { echo "[ERROR] Failed to pull latest xpu-perf code."; exit 1; }

# Comment out ipex rms_norm provider to avoid work-group size RuntimeError
sed -i 's/@ProviderRegistry.register_vendor_impl("rms_norm", "ipex")/#@ProviderRegistry.register_vendor_impl("rms_norm", "ipex")/' \
    "${XPU_PERF_DIR}/micro_perf/backends/INTEL/ops/ipex/rms_norm.py"

ADJUST_SCRIPT="${XPU_PERF_DIR}/micro_perf/backends/INTEL/ops/xccl/adjust_batch_size.sh"
bash "$ADJUST_SCRIPT" b60

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
sleep 30
run_ccl_test all_reduce
run_ccl_test all_gather
run_ccl_test reduce_scatter
run_test device2device
sleep 30
run_ccl_test device2host
run_ccl_test host2device
run_ccl_test all_to_all

}

run_all
