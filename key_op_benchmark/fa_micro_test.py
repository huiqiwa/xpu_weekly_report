#!/usr/bin/env python3
"""
Flash Attention micro benchmark: sweep test cases from flash_attention.json, run bf16 only.

Reads all flash_attention definitions from workloads_report and calls bench_fa2.py
for each combination of (mode, heads, batch, cache_len, q_len).

Usage:
    python fa_micro_test.py [--gpu GPU_ID] [--json FA_JSON] [--output CSV] [--iters N] [--warmup N]
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Flash Attention micro benchmark")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--json", type=str, nargs="+", default=None,
                        help="Path(s) to flash_attention JSON files (default: auto-detect)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: fa_results_<gpu>_<timestamp>.csv)")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    return parser.parse_args()


def find_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def find_fa_json_files(script_dir):
    """Auto-detect flash_attention JSON files from workloads_report."""
    base = os.path.normpath(os.path.join(
        script_dir, "..", "..", "xpu-perf", "projects", "micro_perf", "workloads_report"))
    found = []
    # Primary: vendor_test_report/flash_attention.json
    primary = os.path.join(base, "llm", "vendor_test_report", "flash_attention.json")
    if os.path.isfile(primary):
        found.append(primary)
    # Also search other JSON files that may contain flash_attention key
    for root, dirs, files in os.walk(base, followlinks=True):
        for f in files:
            if f.endswith(".json"):
                path = os.path.join(root, f)
                if path in found:
                    continue
                try:
                    with open(path, 'r') as fh:
                        data = json.load(fh)
                    if "flash_attention" in data:
                        found.append(path)
                except (json.JSONDecodeError, IOError):
                    pass
    return found


def _has_full_bf16(entry):
    """Check if an entry has a full bfloat16 dtype configuration."""
    # Format 1: combined dtype field like "dtype.cache_dtype.qk_compute_dtype.pv_compute_dtype.dst_dtype"
    for key in entry:
        if "dtype" in key and "." in key and isinstance(entry[key], list):
            # List of dtype combos, check if any is all bfloat16
            for combo in entry[key]:
                if isinstance(combo, list) and all(d == "bfloat16" for d in combo):
                    return True
            # Has a combined dtype field but no full bf16 combo
            return False

    # Format 2: separate dtype/cache_dtype fields
    dtype_fields = ["dtype", "cache_dtype", "qk_compute_dtype", "pv_compute_dtype"]
    found_any = False
    for field in dtype_fields:
        if field in entry:
            found_any = True
            val = entry[field]
            if isinstance(val, list):
                if "bfloat16" not in val:
                    return False
            elif val != "bfloat16":
                return False
    return found_any  # True if we checked fields and all were bf16


def load_cases_from_file(json_path):
    """Load flash_attention test cases from a JSON file, bf16 only.

    Returns list of dicts: {mode, batch, q_len, num_heads, num_kv_heads, head_dim, cache_len}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    fa_entries = data.get("flash_attention", [])
    cases = []

    for entry in fa_entries:
        attn_mode = entry.get("attn_mode", "prefill")

        # Filter: skip entries that don't have a full bf16 dtype config
        if not _has_full_bf16(entry):
            continue

        # Extract head configs
        head_configs = []
        if "q_head_num.kv_head_num.head_dim" in entry:
            head_configs = entry["q_head_num.kv_head_num.head_dim"]
        elif "q_head_num" in entry and "kv_head_num" in entry and "head_dim" in entry:
            q_h = entry["q_head_num"]
            kv_h = entry["kv_head_num"]
            hd = entry["head_dim"]
            q_h = q_h if isinstance(q_h, list) else [q_h]
            kv_h = kv_h if isinstance(kv_h, list) else [kv_h]
            hd = hd if isinstance(hd, list) else [hd]
            for q in q_h:
                for k in kv_h:
                    for d in hd:
                        head_configs.append([q, k, d])

        # Extract batch, cache_len, q_len
        # Two formats: separate fields or combined "batch_size.cache_len.q_len"
        if "batch_size.cache_len.q_len" in entry:
            # Combined format: list of [batch, cache_len, q_len]
            combos = entry["batch_size.cache_len.q_len"]
            for combo in combos:
                batch, cl, ql = combo[0], combo[1], combo[2]
                for hc in head_configs:
                    cases.append({
                        "mode": attn_mode,
                        "batch": batch,
                        "q_len": ql,
                        "num_heads": hc[0],
                        "num_kv_heads": hc[1],
                        "head_dim": hc[2],
                        "cache_len": cl,
                    })
        else:
            # Separate fields
            batch_list = entry.get("batch_size", 1)
            if not isinstance(batch_list, list):
                batch_list = [batch_list]

            cache_list = entry.get("cache_len", [0])
            if not isinstance(cache_list, list):
                cache_list = [cache_list]

            q_len_list = entry.get("q_len", [1])
            if not isinstance(q_len_list, list):
                q_len_list = [q_len_list]

            for batch in batch_list:
                for cl in cache_list:
                    for ql in q_len_list:
                        for hc in head_configs:
                            cases.append({
                                "mode": attn_mode,
                                "batch": batch,
                                "q_len": ql,
                                "num_heads": hc[0],
                                "num_kv_heads": hc[1],
                                "head_dim": hc[2],
                                "cache_len": cl,
                            })

    return cases


def run_command(cmd, timeout=600):
    """Run command and return stdout+stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def parse_tflops_latency(output):
    """Extract TFLOPS, latency, and mem_bw from bench_fa2.py output."""
    tflops = 0.0
    latency = 0.0
    mem_bw = 0.0
    m = re.search(r'Avg TFLOPS:\s+([0-9.]+)', output)
    if m:
        tflops = float(m.group(1))
    m = re.search(r'Avg latency:\s+([0-9.]+)', output)
    if m:
        latency = float(m.group(1))
    m = re.search(r'Mem BW:\s+([0-9.]+)', output)
    if m:
        mem_bw = float(m.group(1))
    return tflops, latency, mem_bw


def main():
    args = parse_args()
    script_dir = find_script_dir()
    bench_script = os.path.join(script_dir, "bench_fa2.py")

    if not os.path.isfile(bench_script):
        print(f"Error: bench_fa2.py not found at {bench_script}")
        sys.exit(1)

    # Find JSON files
    json_paths = args.json
    if json_paths is None:
        json_paths = find_fa_json_files(script_dir)
    if not json_paths:
        print("Error: cannot find any flash_attention JSON files. Specify with --json")
        sys.exit(1)

    # Load all cases
    all_cases = []
    for jp in json_paths:
        print(f"Loading cases from: {jp}")
        cases = load_cases_from_file(jp)
        all_cases.extend(cases)
        print(f"  -> {len(cases)} test cases")

    # Deduplicate
    seen = set()
    unique_cases = []
    for c in all_cases:
        key = (c["mode"], c["batch"], c["q_len"], c["num_heads"],
               c["num_kv_heads"], c["head_dim"], c["cache_len"])
        if key not in seen:
            seen.add(key)
            unique_cases.append(c)
    all_cases = unique_cases
    print(f"\nTotal unique test cases: {len(all_cases)}")

    # Get GPU name
    import torch
    torch.cuda.set_device(args.gpu)
    gpu_name = torch.cuda.get_device_name(args.gpu).replace(" ", "_")

    # Output CSV
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(script_dir, f"fa_results_{gpu_name}_{timestamp}.csv")

    print(f"\nOutput CSV: {args.output}")
    print(f"GPU: {args.gpu} ({gpu_name})")
    print(f"Iters: {args.iters}, Warmup: {args.warmup}")
    print(f"Dtype: bfloat16 (only)")
    print()

    fieldnames = ["mode", "batch", "q_len", "cache_len",
                  "num_heads", "num_kv_heads", "head_dim",
                  "latency_ms", "tflops", "mem_bw_gbs"]

    csv_file = open(args.output, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    total = len(all_cases)
    for idx, case in enumerate(all_cases, 1):
        mode = case["mode"]
        batch = case["batch"]
        q_len = case["q_len"]
        num_heads = case["num_heads"]
        num_kv_heads = case["num_kv_heads"]
        head_dim = case["head_dim"]
        cache_len = case["cache_len"]

        print(f"[{idx}/{total}] {mode} B={batch} Q={q_len} CL={cache_len} "
              f"H={num_heads} KVH={num_kv_heads} D={head_dim}", end=" ... ", flush=True)

        cmd = [
            "python3", bench_script,
            str(args.gpu),
            mode,
            "bfloat16",
            str(batch),
            str(q_len),
            str(num_heads),
            str(num_kv_heads),
            str(head_dim),
            str(args.iters),
            str(args.warmup),
            str(cache_len),
        ]

        output = run_command(cmd)
        tflops, latency, mem_bw = parse_tflops_latency(output)

        writer.writerow({
            "mode": mode,
            "batch": batch,
            "q_len": q_len,
            "cache_len": cache_len,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "latency_ms": f"{latency:.3f}",
            "tflops": f"{tflops:.2f}",
            "mem_bw_gbs": f"{mem_bw:.2f}",
        })
        csv_file.flush()

        if tflops > 0:
            print(f"{tflops:.2f} TFLOPS, {latency:.3f} ms, {mem_bw:.1f} GB/s")
        else:
            # Print error snippet for debugging
            err_line = output.strip().split('\n')[-1] if output.strip() else "no output"
            print(f"FAILED ({err_line})")

    csv_file.close()

    print()
    print(f"{'='*60}")
    print(f" Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
