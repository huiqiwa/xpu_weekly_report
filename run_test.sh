SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE=1,2,3,4
REPORT_DIR="${SCRIPT_DIR}/reports/reports_$(TZ='Asia/Shanghai' date +%Y-%m-%d-%H-%M-%S)"

mkdir -p "$REPORT_DIR"

cd "${SCRIPT_DIR}/../xpu-perf/micro_perf"

# python launch.py --task_dir workloads/basic --device 0,1,2,3 --backend INTEL --task all --report_dir all_reports  &> /yupengzh/xpu-perf-logs/basic.txt

# python launch.py --task_dir workloads/llm --device 0,1,2,3 --backend INTEL --task all --report_dir all_reports  &> /yupengzh/xpu-perf-logs/llm.txt

run_test() {
    op_name=$1
    python launch.py --task_dir workloads --device $DEVICE --backend INTEL --task $op_name --report_dir $REPORT_DIR &> $REPORT_DIR/$op_name.txt
    sleep 2
}

run_ccl_test() {
    op_name=$1
    python launch.py --task_dir workloads --backend INTEL --task $op_name --report_dir $REPORT_DIR &> $REPORT_DIR/$op_name.txt
    sleep 2
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

# xccl ops
run_ccl_test all_reduce
run_ccl_test all_gather
run_ccl_test reduce_scatter
run_ccl_test all_to_all
run_test device2device
run_ccl_test device2host
run_ccl_test host2device

# sage_attention
run_test sage_attention_page
run_test sage_attention_decode_page

}

run_all