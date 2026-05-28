#!/usr/bin/env python3
"""Generate a single static HTML with per-op ECharts from CSV report data.
Each op gets a fixed config for x-axis candidates and filter columns.
The HTML is fully static — no dynamic JS DOM building, works with file:// directly.
"""

import csv, glob, json, os, sys
from collections import OrderedDict

import re as _re
_script_dir = os.path.dirname(os.path.abspath(__file__))
_reports_root = os.path.join(_script_dir, "reports")

def _resolve_device_dir(report_base):
    """Given a reports_* directory, resolve to the first INTEL/<device> subdir."""
    intel_root = os.path.join(report_base, "INTEL")
    devices = [p for p in sorted(glob.glob(os.path.join(glob.escape(intel_root), "*"))) if os.path.isdir(p)]
    if not devices:
        raise FileNotFoundError(f"No device directory found under: {intel_root}")
    return devices[0]

def _resolve_default_report_dirs():
    """Resolve report directories. Returns (new_dir, old_dir_or_None)."""
    candidates = sorted(
        [d for d in glob.glob(os.path.join(glob.escape(_reports_root), "reports_*")) if os.path.isdir(d)]
    )
    if not candidates:
        raise FileNotFoundError(f"No report directories found under: {_reports_root}")
    new_dir = _resolve_device_dir(candidates[-1])
    old_dir = _resolve_device_dir(candidates[-2]) if len(candidates) >= 2 else None
    return new_dir, old_dir

def _extract_timestamp(path):
    m = _re.search(r'reports_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', path)
    return m.group(1) if m else None

# ── CLI argument parsing ──────────────────────────────────────
# Usage:
#   gen_charts.py                       -> compare latest two reports
#   gen_charts.py <dir>                 -> single report
#   gen_charts.py <old_dir> <new_dir>   -> compare old vs new
#   gen_charts.py <old_dir> <new_dir> <output.html>
COMPARE_MODE = False
REPORT_DIR_OLD = None

if len(sys.argv) == 1:
    # No args: use latest two
    REPORT_DIR, REPORT_DIR_OLD = _resolve_default_report_dirs()
    COMPARE_MODE = REPORT_DIR_OLD is not None
elif len(sys.argv) == 2:
    # Single report dir: resolve it as new, auto-find previous as old
    REPORT_DIR = _resolve_device_dir(sys.argv[1]) if os.path.isdir(os.path.join(sys.argv[1], "INTEL")) else sys.argv[1]
    _ts_new_dir = _extract_timestamp(REPORT_DIR) or ""
    _prev_candidates = sorted([
        d for d in glob.glob(os.path.join(glob.escape(_reports_root), "reports_*")) if os.path.isdir(d)
        and (_extract_timestamp(d) or "") < _ts_new_dir
    ])
    if _prev_candidates:
        REPORT_DIR_OLD = _resolve_device_dir(_prev_candidates[-1])
        COMPARE_MODE = True
elif len(sys.argv) >= 3:
    # Two report dirs (sort by timestamp so arg order doesn't matter)
    dirs = [sys.argv[1], sys.argv[2]]
    dirs = [_resolve_device_dir(d) if os.path.isdir(os.path.join(d, "INTEL")) else d for d in dirs]
    ts = [_extract_timestamp(d) or "" for d in dirs]
    if ts[0] > ts[1]:
        dirs = [dirs[1], dirs[0]]
    REPORT_DIR_OLD = dirs[0]
    REPORT_DIR = dirs[1]
    COMPARE_MODE = True

# Extract timestamp from report dir path
_ts_match = _re.search(r'reports_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', REPORT_DIR)
os.makedirs(_reports_root, exist_ok=True)
_default_out = os.path.join(_reports_root, "reports-{}.html".format(_ts_match.group(1))) if _ts_match else os.path.join(_reports_root, "all-ops-chart.html")
OUT_HTML = sys.argv[3] if len(sys.argv) > 3 else _default_out
ARCH_NAME = os.path.basename(REPORT_DIR.rstrip('/'))

# Labels for compare mode
_ts_new = _extract_timestamp(REPORT_DIR) or "This Week"
_ts_old = _extract_timestamp(REPORT_DIR_OLD) if REPORT_DIR_OLD else "Last Week"
# Short date for chart display (YYYY-MM-DD)
LABEL_NEW = _ts_new[:10] if _ts_new and len(_ts_new) >= 10 else _ts_new
LABEL_OLD = _ts_old[:10] if _ts_old and len(_ts_old) >= 10 else _ts_old
# Legend names shown in the legend selector
LEGEND_NEW = "This Week (" + LABEL_NEW + ")"
LEGEND_OLD = "Last Week (" + LABEL_OLD + ")"

# ── Per-op config ──────────────────────────────────────────────
# x_fields: columns usable as X axis (first is default)
# filters: columns shown as dropdown filters (first value selected by default)
# If an op is not listed here, we auto-detect from its CSV header.

OP_CONFIG = {
    # Norm & Quant
    "scale_dynamic_quant":          {"x_fields": ["num_tokens", "hidden_size"], "filters": ["arg_type", "provider", "dtype", "dst_dtype"]},
    "head_rms_norm":                {"x_fields": ["num_tokens"], "filters": ["arg_type", "provider", "dtype", "head_dim", "total_head_num", "norm_head_num", "norm_head_start"]},
    "head_rms_norm_dynamic_quant":  {"x_fields": ["num_tokens"], "filters": ["arg_type", "provider", "dtype", "dst_dtype", "head_num", "head_dim"]},
    "add_rms_norm_dynamic_quant":   {"x_fields": ["num_tokens", "hidden_size"], "filters": ["arg_type", "provider", "dtype", "dst_dtype", "add_residual"]},

    # Attention & rope & kvcache
    "rotary_embedding":             {"x_fields": ["q_seq_len", "q_len", "batch_size", "cache_len"], "filters": ["schema", "arg_type", "provider", "dtype", "mode", "attn_mode", "q_head_num", "kv_head_num", "head_dim", "rope_offset", "rope_dim"]},
    "store_kv_cache":               {"x_fields": ["q_len", "batch_size", "cache_len"], "filters": ["arg_type", "provider", "dtype", "cache_dtype", "attn_mode", "paged_cache_layout", "store_mode", "block_size", "q_head_num", "kv_head_num", "head_dim"]},
    "dequant_kv_cache":             {"x_fields": ["q_len", "batch_size", "cache_len"], "filters": ["arg_type", "provider", "dtype", "dst_dtype", "attn_mode", "linear_cache_layout", "block_size", "q_head_num", "kv_head_num", "head_dim"]},
    "flash_attention":              {"x_fields": ["q_len", "batch_size", "cache_len"], "filters": ["arg_type", "provider", "dtype", "cache_dtype", "attn_mode", "block_size", "q_head_num", "kv_head_num", "head_dim"]},

    # gemm & group_gemm & moe_ops
    "moe_gating_gemm":             {"x_fields": ["num_tokens", "hidden_size"], "filters": ["arg_type", "provider", "dtype", "compute_dtype", "dst_dtype", "num_experts", "sp_size"]},
    "quant_matmul":                {"x_fields": ["num_tokens", "hidden_size", "new_hidden_size"], "filters": ["arg_type", "provider", "dtype", "dst_dtype", "w_dtype", "compute_dtype", "sp_size"]},
    "moe_quant_group_gemm":        {"x_fields": ["num_tokens", "hidden_size", "new_hidden_size"], "filters": ["arg_type", "provider", "dtype", "w_dtype", "compute_dtype", "dst_dtype", "num_experts", "topk", "ep_size", "sp_size"]},
    "moe_softmax_topk":            {"x_fields": ["num_tokens", "num_experts"], "filters": ["arg_type", "provider", "dtype", "compute_mode", "topk", "sp_size"]},
    "moe_scatter_dynamic_quant":   {"x_fields": ["num_tokens", "hidden_size"], "filters": ["arg_type", "provider", "dtype", "dst_dtype", "topk", "num_experts", "ep_size"]},
    "moe_swiglu_dynamic_quant":    {"x_fields": ["num_tokens", "hidden_size"], "filters": ["arg_type", "provider", "dtype", "dst_dtype", "topk", "num_experts", "ep_size"]},
    "swiglu_dynamic_quant":        {"x_fields": ["num_tokens", "hidden_size"], "filters": ["arg_type", "provider", "dtype", "dst_dtype"]},
    "moe_gather":                  {"x_fields": ["num_tokens", "hidden_size"], "filters": ["arg_type", "provider", "dtype", "num_experts", "topk", "ep_size", "sp_size"]},
    "moe_quant_group_gemm_combine": {"x_fields": ["num_tokens", "hidden_size", "new_hidden_size"], "filters": ["arg_type", "provider", "dtype", "w_dtype", "compute_dtype", "dst_dtype", "num_experts", "topk", "ep_size", "sp_size"]},
    "quant_group_gemm_reduce_sum":  {"x_fields": ["num_tokens", "hidden_size", "new_hidden_size"], "filters": ["arg_type", "provider", "dtype", "w_dtype", "dst_dtype", "sp_size"]},
    "qk_rms_norm":                  {"x_fields": ["num_tokens"], "filters": ["arg_type", "provider", "dtype", "q_head_num", "kv_head_num", "qk_head_dim", "v_head_dim", "sp_size"]},

    # tensor_gemm
    "gemm":                        {"x_fields": ["M", "K", "N"], "filters": ["arg_type", "provider", "dtype", "dst_dtype"]},

    # vector_activation
    "gelu":                        {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},
    "silu":                        {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},

    # vector_index
    "embedding":                   {"x_fields": ["dim_size", "src_batch_size"], "filters": ["arg_type", "provider", "dtype"]},
    "gather":                      {"x_fields": ["dim_size", "src_batch_size"], "filters": ["arg_type", "provider", "dtype"]},
    "index_add":                   {"x_fields": ["dim_size", "src_batch_size"], "filters": ["arg_type", "provider", "dtype"]},
    "index_select":                {"x_fields": ["dim_size", "src_batch_size"], "filters": ["arg_type", "provider", "dtype"]},
    "scatter":                     {"x_fields": ["dim_size", "src_batch_size"], "filters": ["arg_type", "provider", "dtype"]},

    # vector_linear
    "add":                         {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},
    "cast":                        {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},
    "mul":                         {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},
    "sub":                         {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},

    # vector_norm
    "layer_norm":                  {"x_fields": ["dim_size"], "filters": ["arg_type", "provider", "dtype", "batch_size"]},
    "rms_norm":                    {"x_fields": ["num_tokens", "dim_size", "hidden_size", "batch_size"], "filters": ["schema", "arg_type", "provider", "dtype", "add_residual", "batch_size", "hidden_size"]},
    "softmax":                     {"x_fields": ["dim_size"], "filters": ["arg_type", "provider", "dtype", "batch_size"]},

    # vector_reduction
    "reduce_max":                  {"x_fields": ["dim_size"], "filters": ["arg_type", "provider", "dtype", "batch_size"]},
    "reduce_min":                  {"x_fields": ["dim_size"], "filters": ["arg_type", "provider", "dtype", "batch_size"]},
    "reduce_sum":                  {"x_fields": ["dim_size"], "filters": ["arg_type", "provider", "dtype", "batch_size"]},
    "topk":                        {"x_fields": ["dim_size"], "filters": ["arg_type", "provider", "dtype", "batch_size", "k"]},

    # vector_sfu
    "cos":                         {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},
    "div":                         {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},
    "exp":                         {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},
    "log":                         {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},
    "sin":                         {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},
    "sqrt":                        {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "dim_size"]},

    # xccl
    "all_reduce":                  {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "world_size", "dim_size"], "y_options": ["algo_bw(GB/s)", "bus_bw(GB/s)", "latency(us)"]},
    "all_gather":                  {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "world_size", "dim_size"], "y_options": ["algo_bw(GB/s)", "bus_bw(GB/s)", "latency(us)"]},
    "reduce_scatter":              {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "world_size", "dim_size"], "y_options": ["algo_bw(GB/s)", "bus_bw(GB/s)", "latency(us)"]},
    "all_to_all":                  {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "world_size", "dim_size"], "y_options": ["algo_bw(GB/s)", "bus_bw(GB/s)", "latency(us)"]},
    "device2device":               {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "world_size", "dim_size"], "y_options": ["mem_bw(GB/s)", "MBU(%)", "latency(us)"]},
    "device2host":                 {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "world_size", "dim_size"], "y_options": ["algo_bw(GB/s)", "bus_bw(GB/s)", "latency(us)"]},
    "host2device":                 {"x_fields": ["batch_size"], "filters": ["arg_type", "provider", "dtype", "world_size", "dim_size"], "y_options": ["algo_bw(GB/s)", "bus_bw(GB/s)", "latency(us)"]},

    # sage_attention
    "sage_attention_page":         {"x_fields": ["q_head_num", "k_seq_len"], "filters": ["arg_type", "provider", "dtype", "mode", "kv_head_num", "head_dim", "batch_size", "block_size"]},
    "sage_attention_decode_page":  {"x_fields": ["q_head_num", "k_seq_len"], "filters": ["arg_type", "provider", "dtype", "mode", "kv_head_num", "head_dim", "batch_size", "block_size"]},
    "sage_attention_v1":           {"x_fields": ["q_head_num", "k_seq_len"], "filters": ["arg_type", "provider", "dtype", "mode", "kv_head_num", "head_dim", "batch_size", "block_size"]},
}

OP_ORDER = [
    "scale_dynamic_quant", "head_rms_norm", "head_rms_norm_dynamic_quant", "add_rms_norm_dynamic_quant",
    "rotary_embedding", "store_kv_cache", "dequant_kv_cache", "flash_attention",
    "moe_gating_gemm", "quant_matmul", "moe_quant_group_gemm", "moe_softmax_topk",
    "moe_scatter_dynamic_quant", "moe_swiglu_dynamic_quant", "swiglu_dynamic_quant", "moe_gather",
    "moe_quant_group_gemm_combine", "quant_group_gemm_reduce_sum", "qk_rms_norm",
    "gemm",
    "gelu", "silu",
    "embedding", "gather", "index_add", "index_select", "scatter",
    "add", "cast", "mul", "sub",
    "layer_norm", "rms_norm", "softmax",
    "reduce_max", "reduce_min", "reduce_sum", "topk",
    "cos", "div", "exp", "log", "sin", "sqrt",
    "all_reduce", "all_gather", "reduce_scatter", "all_to_all", "device2device", "device2host", "host2device",
    "sage_attention_page", "sage_attention_decode_page", "sage_attention_v1",
]

# Hardware specs for utilization calculation
HW_MEM_BW_GBs = 456.0        # Memory bandwidth in GB/s
HW_PEAK_TFLOPS_FP16 = 98.5   # fp16/bf16 peak compute
HW_PEAK_TFLOPS_FP32 = 12.28  # fp32 peak compute
HW_PEAK_TFLOPS_INT8 = 197.0  # int8 peak compute

Y_OPTIONS = [
    ("calc_flops_power(tflops)", "calc_flops_power (tflops)"),
    ("mem_bw(GB/s)", "mem_bw (GB/s)"),
    ("MFU(%)", "MFU (%)"),
    ("MBU(%)", "MBU (%)"),
    ("latency(us)", "latency (us)"),
]

# Label map for all known metric columns (used when y_options overrides Y_OPTIONS)
Y_LABEL_MAP = {v: lb for v, lb in Y_OPTIONS}
Y_LABEL_MAP.update({
    "algo_bw(GB/s)": "algo_bw (GB/s)",
    "bus_bw(GB/s)": "bus_bw (GB/s)",
})

# ── Data loading ──────────────────────────────────────────────

def load_ops(report_dir):
    csv_files = sorted(glob.glob(os.path.join(glob.escape(report_dir), "*", "*", "*.csv")))
    ops = OrderedDict()
    for path in csv_files:
        parts = path.split(os.sep)
        op_name = parts[-3]
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if op_name not in ops:
            ops[op_name] = rows
        else:
            ops[op_name].extend(rows)

    # Add computed 'schema' column for rotary_embedding
    # attn_ops schema: has 'mode' (prefill/decode/prefill_session_cache) + 'q_seq_len'
    # pre_fa_ops schema: has 'attn_mode' (prefill/decode) + 'q_len', no 'mode'
    if "rotary_embedding" in ops:
        for r in ops["rotary_embedding"]:
            if r.get("mode") not in (None, ""):
                r["schema"] = "attn_ops"
            else:
                r["schema"] = "pre_fa_ops"

    # Add computed 'schema' column for rms_norm
    # norm_ops schema: arg_type=llm, uses num_tokens + hidden_size
    # rms_norm schema: arg_type=default, uses dim_size + batch_size
    if "rms_norm" in ops:
        for r in ops["rms_norm"]:
            if r.get("arg_type") == "llm":
                r["schema"] = "norm_ops"
            else:
                r["schema"] = "rms_norm"

    return ops

def unique_values(rows, col):
    seen = []
    for r in rows:
        v = r.get(col)
        if v is not None and v != "" and v not in seen:
            seen.append(v)
    return seen

def get_config(op_name, header):
    """Get config for an op. If not in OP_CONFIG, auto-detect."""
    if op_name in OP_CONFIG:
        cfg = OP_CONFIG[op_name]
        x_fields = [c for c in cfg["x_fields"] if c in header]
        filters = [c for c in cfg["filters"] if c in header]
        if not x_fields:
            x_fields = [header[0]]
        return x_fields, filters

    skip = {"sku_name", "op_name", "kernels", "latency(us)", "read_bytes(B)", "write_bytes(B)",
            "io_bytes(B)", "mem_bw(GB/s)", "calc_flops", "calc_flops_power(tflops)", "calc_mem_ratio"}
    param_cols = [c for c in header if c not in skip]
    x_fields, filters = [], []
    for c in param_cols:
        if c in ("provider", "dtype", "arg_type"):
            filters.append(c)
        else:
            x_fields.append(c)
    if not x_fields and param_cols:
        x_fields = [param_cols[0]]
    return x_fields or ["batch_size"], filters

# ── HTML generation ──────────────────────────────────────────

def esc(s):
    return str(s).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

def gen_op_section(op_name, rows, old_rows=None):
    header = list(rows[0].keys())
    x_fields, filter_cols = get_config(op_name, header)

    # Per-op Y_OPTIONS override
    _op_cfg = OP_CONFIG.get(op_name, {})
    _y_keys = _op_cfg.get("y_options")
    if _y_keys:
        op_y_options = [(v, Y_LABEL_MAP.get(v, v)) for v in _y_keys]
    else:
        op_y_options = Y_OPTIONS

    # Only keep filters with values (skip cols where >half the rows are empty), and x_fields with >1 unique value
    # When "schema" column exists, use per-schema thresholds so schema-specific cols aren't excluded
    has_schema_col = "schema" in filter_cols
    filter_info = []
    for col in filter_cols:
        uvals = unique_values(rows, col)
        if has_schema_col and col != "schema":
            # Keep if significant in ANY schema group
            dominated = False
            for sv in unique_values(rows, "schema"):
                sr = [r for r in rows if r.get("schema") == sv]
                ne = sum(1 for r in sr if r.get(col) not in (None, ""))
                if uvals and ne > len(sr) / 2:
                    dominated = True
                    break
            if dominated:
                filter_info.append((col, uvals))
        else:
            non_empty = sum(1 for r in rows if r.get(col) not in (None, ""))
            if uvals and non_empty > len(rows) / 2:
                filter_info.append((col, uvals))

    if has_schema_col:
        x_fields_multi = []
        for f in x_fields:
            if len(unique_values(rows, f)) <= 1:
                continue
            for sv in unique_values(rows, "schema"):
                sr = [r for r in rows if r.get("schema") == sv]
                ne = sum(1 for r in sr if r.get(f) not in (None, ""))
                if ne > len(sr) / 2:
                    x_fields_multi.append(f)
                    break
    else:
        x_fields_multi = [f for f in x_fields if len(unique_values(rows, f)) > 1
                           and sum(1 for r in rows if r.get(f) not in (None, "")) > len(rows) / 2]
    if not x_fields_multi:
        x_fields_multi = x_fields[:1]

    # Track single-value x_fields that are being dropped — needed to constrain old data in compare mode
    _dropped_xfields = {}  # field -> single value (string)
    for f in x_fields:
        if f not in x_fields_multi:
            uv = unique_values(rows, f)
            if len(uv) == 1:
                _dropped_xfields[f] = str(uv[0])

    x_fields = x_fields_multi

    # JSON rows — compute MFU and MBU
    json_rows = []
    for r in rows:
        row = {}
        for k, v in r.items():
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                row[k] = v
        # MBU = mem_bw / HW_MEM_BW * 100
        mem_bw = row.get("mem_bw(GB/s)", 0) or 0
        row["MBU(%)"] = round(float(mem_bw) / HW_MEM_BW_GBs * 100, 2) if HW_MEM_BW_GBs else 0
        # MFU = calc_flops_power / peak_tflops * 100 (peak depends on dtype)
        flops = row.get("calc_flops_power(tflops)", 0) or 0
        if op_name in ("sage_attention_page", "sage_attention_decode_page", "sage_attention_v1"):
            peak = HW_PEAK_TFLOPS_INT8
        else:
            dtype = str(row.get("dtype", "")).lower()
            if dtype in ("float32", "fp32", "tf32", "tfloat32"):
                peak = HW_PEAK_TFLOPS_FP32
            elif dtype in ("int8",):
                peak = HW_PEAK_TFLOPS_INT8
            else:
                peak = HW_PEAK_TFLOPS_FP16
        row["MFU(%)"] = round(float(flops) / peak * 100, 2) if peak else 0
        json_rows.append(row)

    op_id = op_name.replace(" ", "_").replace("-", "_")
    data_var = "data_" + op_id

    # Decide default Y: pick MFU or MBU based on which has higher max, or first op_y_options
    max_mfu = max((r.get("MFU(%)", 0) for r in json_rows), default=0)
    max_mbu = max((r.get("MBU(%)", 0) for r in json_rows), default=0)
    _pref = "MFU(%)" if max_mfu >= max_mbu else "MBU(%)"
    default_y = _pref if any(v == _pref for v, _ in op_y_options) else op_y_options[0][0]

    # Collect unique values for filters and x_fields (needed for validation & defaults)
    filter_uval_map = {col: uvals for col, uvals in filter_info}
    xfield_uval_map = {f: unique_values(rows, f) for f in x_fields}

    # Find best default filter combination:
    # Group rows by (filter values + non-selected x_field values), pick the group
    # with best max(default_y) that has >=2 rows (prefer multi-point charts).
    # Default x-field is x_fields[0].
    default_x = x_fields[0] if x_fields else None
    all_filter_cols = [col for col, _ in filter_info]
    non_x_fields = [f for f in x_fields if f != default_x]
    group_keys = all_filter_cols + non_x_fields

    from collections import defaultdict
    groups = defaultdict(list)
    for i, r in enumerate(json_rows):
        # Use original string values (from rows) for grouping to match dropdown options
        orig = rows[i] if i < len(rows) else {}
        key = tuple(str(orig.get(c, "")) for c in group_keys)
        groups[key].append(r)

    best_score = -1
    best_key = None
    for key, grp in groups.items():
        # Skip groups with empty filter values
        if any(v == "" for v in key):
            continue
        score = max(r.get(default_y, 0) for r in grp)
        # Strongly prefer groups with >=2 rows to get meaningful charts
        if len(grp) < 2:
            score *= 0.5
        if score > best_score:
            best_score = score
            best_key = key

    default_filters = {}  # col -> default value (string)
    if best_key:
        for i, col in enumerate(group_keys):
            sv = best_key[i]
            valid_vals = [str(u) for u in (filter_uval_map.get(col) or xfield_uval_map.get(col, []))]
            if sv and sv in valid_vals:
                default_filters[col] = sv

    y_opts = "".join(
        '<option value="{v}"{sel}>{lb}</option>'.format(
            v=esc(v), lb=esc(lb),
            sel=' selected' if v == default_y else ''
        ) for v, lb in op_y_options
    )

    # Build unique values for each x_field (for their filter dropdowns)
    x_field_uvals = [(f, xfield_uval_map[f]) for f in x_fields]

    x_opts = "".join('<option value="{v}">{v}</option>'.format(v=esc(f)) for f in x_fields)

    # Build dropdowns for x_fields (shown as filters when not selected as X)
    xfilter_html = ""
    for f, uvals in x_field_uvals:
        opts = "".join('<option value="{v}"{sel}>{v}</option>'.format(
            v=esc(v), sel=' selected' if str(v) == default_filters.get(f) else ''
        ) for v in uvals)
        xfilter_html += (
            '\n        <div class="cg" id="xf_{oid}_{col}_wrap"><label>{col}</label>'
            '<select id="xf_{oid}_{col}" onchange="render_{oid}()">{opts}</select></div>'
        ).format(col=esc(f), oid=op_id, opts=opts)

    pre_filter_html = ""
    filter_html = ""
    for col, uvals in filter_info:
        opts = "".join('<option value="{v}"{sel}>{v}</option>'.format(
            v=esc(v), sel=' selected' if str(v) == default_filters.get(col) else ''
        ) for v in uvals)
        snippet = (
            '\n        <div class="cg"><label>{col}</label>'
            '<select id="f_{oid}_{col}" onchange="render_{oid}()">{opts}</select></div>'
        ).format(col=esc(col), oid=op_id, opts=opts)
        if col == "schema":
            pre_filter_html += snippet
        else:
            filter_html += snippet

    # Schema-aware logic: compute per-schema valid columns
    has_schema_filter = any(col == "schema" for col, _ in filter_info)
    schema_js_vars = ""
    schema_render_prefix = ""

    # Always generate rebuildSel helper for cascading dropdowns
    schema_js_vars += (
        'function rebuildSel_{oid}(sel, opts) {{\n'
        '  var cur = sel.value;\n'
        '  sel.innerHTML = "";\n'
        '  for (var i = 0; i < opts.length; i++) {{ var o = document.createElement("option"); o.value = opts[i]; o.textContent = opts[i]; sel.appendChild(o); }}\n'
        '  if (opts.indexOf(cur) >= 0) sel.value = cur; }}\n'
    ).format(oid=op_id)

    if has_schema_filter:
        schema_col_map = {}
        schema_x_map = {}
        for col, uvals in filter_info:
            if col == "schema":
                for sv in uvals:
                    sr = [r for r in rows if r.get("schema") == sv]
                    schema_col_map[sv] = [fc for fc, _ in filter_info if fc != "schema"
                                          and any(r.get(fc) not in (None, "") for r in sr)]
                    schema_x_map[sv] = [xf for xf in x_fields
                                        if any(r.get(xf) not in (None, "") for r in sr)]
                break

        non_schema_filters = [col for col, _ in filter_info if col != "schema"]

        schema_js_vars += 'var schemaCols_{oid} = {cm};\nvar schemaXFs_{oid} = {xm};\n'.format(
            oid=op_id, cm=json.dumps(schema_col_map), xm=json.dumps(schema_x_map))

        schema_render_prefix = '  var schemaVal = document.getElementById("f_{oid}_schema").value;\n'.format(oid=op_id)
        schema_render_prefix += '  var validFs = schemaCols_{oid}[schemaVal] || [];\n'.format(oid=op_id)
        schema_render_prefix += '  var validXs = schemaXFs_{oid}[schemaVal] || [];\n'.format(oid=op_id)
        # Rebuild X axis dropdown
        schema_render_prefix += '  rebuildSel_{oid}(document.getElementById("x_{oid}"), validXs);\n'.format(oid=op_id)
        # Show/hide filter dropdowns
        for col in non_schema_filters:
            schema_render_prefix += '  document.getElementById("f_{oid}_{col}").parentElement.style.display = (validFs.indexOf("{col}") >= 0) ? "" : "none";\n'.format(oid=op_id, col=col)

    # Build xfilter visibility (show/hide x_field dropdowns based on X axis and schema)
    xfilter_visibility = ""
    if has_schema_filter:
        for f, _ in x_field_uvals:
            xfilter_visibility += (
                '  document.getElementById("xf_{oid}_{col}_wrap").style.display = '
                '(xF === "{col}" || validXs.indexOf("{col}") < 0) ? "none" : "";\n'
            ).format(oid=op_id, col=f)
    else:
        for f, _ in x_field_uvals:
            xfilter_visibility += (
                '  document.getElementById("xf_{oid}_{col}_wrap").style.display = '
                '(xF === "{col}") ? "none" : "";\n'
            ).format(oid=op_id, col=f)

    # ── Cascading filter block ──────────────────────────────────
    # Each filter progressively narrows data; subsequent dropdowns only show remaining values.
    cascade_block = ""
    cascade_info = ""

    for col, _ in filter_info:
        safe_col = col.replace("(", "_").replace(")", "_")
        if col == "schema":
            # Schema filter: apply directly, no option rebuild needed
            cascade_block += '  var fv_{sc} = document.getElementById("f_{oid}_{col}").value;\n'.format(
                sc=safe_col, oid=op_id, col=col)
            cascade_block += '  sub = sub.filter(function(r) {{ return String(r["{col}"]) === fv_{sc}; }});\n'.format(
                col=col, sc=safe_col)
            cascade_info += '  info += "  |  {col}: " + document.getElementById("f_{oid}_{col}").value;\n'.format(
                col=col, oid=op_id)
            continue

        if has_schema_filter:
            cascade_block += '  if (validFs.indexOf("{col}") >= 0) {{\n'.format(col=col)
            ind = "    "
        else:
            ind = "  "

        cascade_block += '{ind}var _uv = []; for (var _i = 0; _i < sub.length; _i++) {{ var _v = String(sub[_i]["{col}"]); if (_v !== "" && _uv.indexOf(_v) < 0) _uv.push(_v); }}\n'.format(ind=ind, col=col)
        cascade_block += '{ind}rebuildSel_{oid}(document.getElementById("f_{oid}_{col}"), _uv);\n'.format(ind=ind, oid=op_id, col=col)
        cascade_block += '{ind}var fv_{sc} = document.getElementById("f_{oid}_{col}").value;\n'.format(ind=ind, sc=safe_col, oid=op_id, col=col)
        cascade_block += '{ind}sub = sub.filter(function(r) {{ return String(r["{col}"]) === fv_{sc}; }});\n'.format(ind=ind, col=col, sc=safe_col)

        if has_schema_filter:
            cascade_block += '  }\n'
            cascade_info += '  if (validFs.indexOf("{col}") >= 0) info += "  |  {col}: " + document.getElementById("f_{oid}_{col}").value;\n'.format(col=col, oid=op_id)
        else:
            cascade_info += '  info += "  |  {col}: " + document.getElementById("f_{oid}_{col}").value;\n'.format(col=col, oid=op_id)

    # X-field cascading filters (non-selected x_fields act as filters)
    for f, _ in x_field_uvals:
        safe_f = f.replace("(", "_").replace(")", "_")
        if has_schema_filter:
            cascade_block += '  if ("{col}" !== xF && validXs.indexOf("{col}") >= 0) {{\n'.format(col=f)
        else:
            cascade_block += '  if ("{col}" !== xF) {{\n'.format(col=f)

        cascade_block += '    var _uv = []; for (var _i = 0; _i < sub.length; _i++) {{ var _v = String(sub[_i]["{col}"]); if (_v !== "" && _uv.indexOf(_v) < 0) _uv.push(_v); }}\n'.format(col=f)
        cascade_block += '    rebuildSel_{oid}(document.getElementById("xf_{oid}_{col}"), _uv);\n'.format(oid=op_id, col=f)
        cascade_block += '    var xfv_{sc} = document.getElementById("xf_{oid}_{col}").value;\n'.format(sc=safe_f, oid=op_id, col=f)
        cascade_block += '    sub = sub.filter(function(r) {{ return String(r["{col}"]) === xfv_{sc}; }});\n'.format(col=f, sc=safe_f)
        cascade_block += '  }\n'

        if has_schema_filter:
            cascade_info += '  if ("{col}" !== xF && validXs.indexOf("{col}") >= 0) info += "  |  {col}: " + document.getElementById("xf_{oid}_{col}").value;\n'.format(col=f, oid=op_id)
        else:
            cascade_info += '  if ("{col}" !== xF) info += "  |  {col}: " + document.getElementById("xf_{oid}_{col}").value;\n'.format(col=f, oid=op_id)

    y_map_json = json.dumps({v: lb for v, lb in op_y_options}, ensure_ascii=False)

    # Process old_rows for comparison mode
    old_data_var = "old_data_" + op_id
    old_json_rows = []
    if old_rows:
        for r in old_rows:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = v
            mem_bw = row.get("mem_bw(GB/s)", 0) or 0
            row["MBU(%)"] = round(float(mem_bw) / HW_MEM_BW_GBs * 100, 2) if HW_MEM_BW_GBs else 0
            flops = row.get("calc_flops_power(tflops)", 0) or 0
            if op_name in ("sage_attention_page", "sage_attention_decode_page", "sage_attention_v1"):
                peak = HW_PEAK_TFLOPS_INT8
            else:
                dtype = str(row.get("dtype", "")).lower()
                if dtype in ("float32", "fp32", "tf32", "tfloat32"):
                    peak = HW_PEAK_TFLOPS_FP32
                elif dtype in ("int8",):
                    peak = HW_PEAK_TFLOPS_INT8
                else:
                    peak = HW_PEAK_TFLOPS_FP16
            row["MFU(%)"] = round(float(flops) / peak * 100, 2) if peak else 0
            old_json_rows.append(row)

    has_old = bool(old_json_rows)
    old_data_js = 'var {} = {};\n'.format(old_data_var, json.dumps(old_json_rows, separators=(",", ":"))) if has_old else ""

    # Build series JS - two lines in compare mode
    if has_old:
        # Build old cascade block: filter old data using current dropdown values
        # Use tolerant filtering: if no rows match a filter value, skip that filter
        old_cascade_block = ""
        for col, _ in filter_info:
            safe_col = col.replace("(", "_").replace(")", "_")
            if col == "schema":
                old_cascade_block += '  var ofv_{sc} = document.getElementById("f_{oid}_{col}").value;\n'.format(
                    sc=safe_col, oid=op_id, col=col)
                old_cascade_block += '  oldSub = oldSub.filter(function(r) {{ return String(r["{col}"]) === ofv_{sc}; }});\n'.format(
                    col=col, sc=safe_col)
                continue
            if has_schema_filter:
                old_cascade_block += '  if (validFs.indexOf("{col}") >= 0) {{\n'.format(col=col)
                ind = "    "
            else:
                ind = "  "
            old_cascade_block += '{ind}var ofv_{sc} = document.getElementById("f_{oid}_{col}").value;\n'.format(ind=ind, sc=safe_col, oid=op_id, col=col)
            old_cascade_block += '{ind}oldSub = oldSub.filter(function(r) {{ return String(r["{col}"]) === ofv_{sc}; }});\n'.format(ind=ind, col=col, sc=safe_col)
            if has_schema_filter:
                old_cascade_block += '  }\n'

        # X-field filters for old data
        for f, _ in x_field_uvals:
            safe_f = f.replace("(", "_").replace(")", "_")
            if has_schema_filter:
                old_cascade_block += '  if ("{col}" !== xF && validXs.indexOf("{col}") >= 0) {{\n'.format(col=f)
            else:
                old_cascade_block += '  if ("{col}" !== xF) {{\n'.format(col=f)
            old_cascade_block += '    var oxfv_{sc} = document.getElementById("xf_{oid}_{col}").value;\n'.format(sc=safe_f, oid=op_id, col=f)
            old_cascade_block += '    oldSub = oldSub.filter(function(r) {{ return String(r["{col}"]) === oxfv_{sc}; }});\n'.format(col=f, sc=safe_f)
            old_cascade_block += '  }\n'

        # Constrain old data by single-value x_fields that were dropped from new data
        # (e.g., new data only has K=4096, so old data must also be filtered to K=4096)
        for dropped_f, dropped_v in _dropped_xfields.items():
            old_cascade_block += '  // Implicit filter: new data only has {col}={val}\n'.format(col=dropped_f, val=dropped_v)
            old_cascade_block += '  oldSub = oldSub.filter(function(r) {{ return String(r["{col}"]) === "{val}"; }});\n'.format(col=dropped_f, val=dropped_v)

        series_js = (
            '\n  // Filter old data with same filter values\n'
            '  var oldSub = ' + old_data_var + '.slice();\n'
            + old_cascade_block +
            '  oldSub.sort(function(a,b){ return Number(a[xF]) - Number(b[xF]); });\n'
            '  var oldXD = [], oldYD = [];\n'
            '  for (var i=0; i<oldSub.length; i++) { oldXD.push(String(oldSub[i][xF])); oldYD.push(oldSub[i][yF]); }\n'
            '  // Align old data to new x-axis (new data is primary)\n'
            '  var oldYAligned = xD.map(function(x) { var idx = oldXD.indexOf(x); return idx >= 0 ? oldYD[idx] : null; });\n'
            '  var hasOldData = oldYAligned.some(function(v) { return v !== null; });\n'
        )

        series_template = '''(function() {
      var s = [{ name: "''' + LEGEND_NEW + '''", type: "line", smooth: true, symbol: "circle", symbolSize: 7,
                data: yD.map(function(v) {
                  if (v === null) return null;
                  if ((yF === "MFU(%)" || yF === "MBU(%)") && Number(v) > 100)
                    return { value: v, itemStyle: { color: "#f56c6c" }, label: { color: "#f56c6c", fontWeight: "bold" } };
                  return v;
                }),
                lineStyle: { width: 2.5 }, areaStyle: { opacity: 0.08 },
                label: { show: true, fontSize: 11, color: "#409eff", position: "top",
                         formatter: function(p) { return p.value == null ? "" : p.value; } },
                itemStyle: { color: "#409eff" },
                markLine: (yF === "MFU(%)" || yF === "MBU(%)") ? {
                  silent: true, symbol: "none",
                  data: [{ yAxis: 100, label: { formatter: "100%", position: "insideEndTop" },
                           lineStyle: { color: "#f56c6c", type: "dashed", width: 2 } }]
                } : {} }];
      if (hasOldData) {
        s.push({ name: "''' + LEGEND_OLD + '''", type: "line", smooth: true, symbol: "diamond", symbolSize: 6,
                data: oldYAligned,
                lineStyle: { width: 2, type: "dashed" }, areaStyle: { opacity: 0.04 },
                label: { show: false },
                itemStyle: { color: "#909399" } });
      }
      return s;
    })()'''
    else:
        series_js = ""
        series_template = '''[{ name: yL, type: "line", smooth: true, symbol: "circle", symbolSize: 7,
                data: yD.map(function(v) {
                  if ((yF === "MFU(%)" || yF === "MBU(%)") && Number(v) > 100)
                    return { value: v, itemStyle: { color: "#f56c6c" }, label: { color: "#f56c6c", fontWeight: "bold" } };
                  return v;
                }),
                lineStyle: { width: 2.5 }, areaStyle: { opacity: 0.08 },
                label: { show: true, fontSize: 11, color: "#333", position: "top" },
                itemStyle: { color: "#409eff" },
                markLine: (yF === "MFU(%)" || yF === "MBU(%)") ? {
                  silent: true, symbol: "none",
                  data: [{ yAxis: 100, label: { formatter: "100%", position: "insideEndTop" },
                           lineStyle: { color: "#f56c6c", type: "dashed", width: 2 } }]
                } : {} }]'''

    legend_js = '    legend: { show: hasOldData, top: 55 },\n' if has_old else ''
    tooltip_js = '''    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },''' if has_old else '''    tooltip: { trigger: "axis", axisPointer: { type: "cross" },
      formatter: function(params) {
        var p = params[0];
        return "<b>" + xF + ": " + p.name + "</b><br/>" + p.marker + " " + yL + ": <b>" + p.data + "</b>";
      }
    },'''

    section = '''
<div class="op" id="s_{oid}">
  <h2>{name}</h2>
  <div class="panel">
    {pre_filter_html}
    <div class="cg"><label>X 轴</label>
    <select id="x_{oid}" onchange="render_{oid}()">{x_opts}</select></div>
    <div class="cg"><label>Y 轴</label>
    <select id="y_{oid}" onchange="render_{oid}()">{y_opts}</select></div>
    {filter_html}{xfilter_html}
  </div>
  <div class="chart" id="c_{oid}"></div>
</div>
<script>
var {dvar} = {jdata};
{old_data_js}{schema_vars}var chart_{oid} = echarts.init(document.getElementById("c_{oid}"));
function render_{oid}() {
{schema_prefix}  var xF = document.getElementById("x_{oid}").value;
  var yF = document.getElementById("y_{oid}").value;
  var yL = {ymap}[yF] || yF;
{xf_vis}  var rows = {dvar};
  var sub = rows.slice();
{cascade_block}  sub.sort(function(a,b){ return Number(a[xF]) - Number(b[xF]); });
  var xD = [], yD = [];
  for (var i=0; i<sub.length; i++) { xD.push(String(sub[i][xF])); yD.push(sub[i][yF]); }
  var info = "X: " + xF + "  |  Y: " + yL;
{cascade_info}{series_js}  var yMax = Math.max.apply(null, yD.map(Number));
  var yAxisCfg = { name: yL, type: "value",
              axisLine: { show: true, lineStyle: { color: "#409eff" } },
              splitLine: { lineStyle: { type: "dashed", color: "#eee" } } };
  if ((yF === "MFU(%)" || yF === "MBU(%)") && yMax < 100) { yAxisCfg.min = 0; yAxisCfg.max = 100; }
  chart_{oid}.setOption({
    title: { text: "{name_esc}", subtext: info, left: "center", top: 0, subtextStyle: { fontSize: 12 } },
{legend_js}{tooltip_js}
    grid: { top: {grid_top}, left: 80, right: 80, bottom: 60 },
    xAxis: { name: xF, type: "category", data: xD, axisLabel: { rotate: 30 },
              nameLocation: "middle", nameGap: 40,
              axisTick: { alignWithLabel: true },
              splitLine: { show: true, lineStyle: { type: "dashed", color: "#eee" } } },
    yAxis: yAxisCfg,
    series: {series_template}
  }, true);
}
render_{oid}();
window.addEventListener("resize", function(){ chart_{oid}.resize(); });
</script>
'''
    # Use manual replacement instead of .format() to avoid issues with
    # literal JS braces in cascade_block, series_js, etc.
    replacements = [
        ('{name_esc}', op_name.replace('"', '\\"')),
        ('{oid}', op_id),
        ('{name}', esc(op_name)),
        ('{x_opts}', x_opts),
        ('{y_opts}', y_opts),
        ('{pre_filter_html}', pre_filter_html),
        ('{filter_html}', filter_html),
        ('{xfilter_html}', xfilter_html),
        ('{dvar}', data_var),
        ('{old_data_js}', old_data_js),
        ('{ymap}', y_map_json),
        ('{schema_vars}', schema_js_vars),
        ('{schema_prefix}', schema_render_prefix),
        ('{xf_vis}', xfilter_visibility),
        ('{cascade_block}', cascade_block),
        ('{cascade_info}', cascade_info),
        ('{series_js}', series_js),
        ('{legend_js}', legend_js),
        ('{tooltip_js}', tooltip_js),
        ('{grid_top}', str(120 if has_old else 80)),
        ('{series_template}', series_template),
        ('{jdata}', json.dumps(json_rows, separators=(",", ":"))),
    ]
    for k, v in replacements:
        section = section.replace(k, v)
    return section


def gen_html(ops, ordered_names, old_ops=None):
    nav_links = "".join('<a href="#s_{n}">{n}</a>'.format(n=n) for n in ordered_names)

    sections = ""
    for name in ordered_names:
        if name in ops:
            old_rows = old_ops.get(name) if old_ops else None
            sections += gen_op_section(name, ops[name], old_rows=old_rows)

    return '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>XPU Perf — ''' + ARCH_NAME + ''' Test Report</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 24px; background: #f0f2f5; color: #333; }
h1 { font-size: 24px; margin-bottom: 12px; color: #1a1a2e; }
.nav {
    display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px;
    background: #fff; padding: 14px 18px; border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.nav a {
    padding: 5px 12px; border-radius: 5px; font-size: 13px;
    text-decoration: none; color: #409eff; border: 1px solid #d9ecff;
}
.nav a:hover { background: #409eff; color: #fff; border-color: #409eff; }
.op { margin-bottom: 36px; }
.op h2 {
    font-size: 18px; margin-bottom: 10px; color: #303133;
    border-left: 4px solid #409eff; padding-left: 10px;
}
.panel {
    display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-end;
    background: #fff; padding: 14px 18px; border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 8px;
}
.cg { display: flex; flex-direction: column; gap: 4px; }
.cg label { font-size: 12px; font-weight: 600; color: #606266; }
select {
    padding: 6px 10px; border-radius: 5px; border: 1px solid #dcdfe6;
    font-size: 13px; background: #fff; min-width: 100px; cursor: pointer;
}
select:focus { outline: none; border-color: #409eff; box-shadow: 0 0 0 2px rgba(64,158,255,0.2); }
.chart {
    width: 100%%; height: 480px; background: #fff; border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 12px;
}
</style>
</head>
<body>
<h1>XPU Perf — ''' + ARCH_NAME + ''' Test Report</h1>
<div class="nav">''' + nav_links + '''</div>
''' + sections + '''
</body>
</html>'''


# ── Main ──────────────────────────────────────────────
ops = load_ops(REPORT_DIR)
old_ops = load_ops(REPORT_DIR_OLD) if COMPARE_MODE and REPORT_DIR_OLD else None
ordered = [n for n in OP_ORDER if n in ops]
remaining = sorted(n for n in ops if n not in ordered)
ordered.extend(remaining)

html = gen_html(ops, ordered, old_ops=old_ops)
with open(OUT_HTML, "w") as f:
    f.write(html)

mode_str = " (compare: {} vs {})".format(LABEL_OLD, LABEL_NEW) if COMPARE_MODE else ""
print("Generated {} with {} ops from {} rows.{}".format(
    OUT_HTML, len([n for n in ordered if n in ops]), sum(len(v) for v in ops.values()), mode_str))
