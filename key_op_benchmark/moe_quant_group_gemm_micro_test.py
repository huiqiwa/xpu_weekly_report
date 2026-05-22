#!/usr/bin/env python3
"""
MoE Quant Group GEMM micro benchmark: sweep cases from moe_quant_group_gemm.json,
test 5 dtypes via bench_moe_quant_group_gemm_with_flashinfer.py.

Dtypes: bfloat16, fp8_per_tensor, fp8_per_block, mxfp4, nvfp4

Usage:
    python moe_quant_group_gemm_micro_test.py [--gpu GPU_ID] [--json JSON] [--output CSV] [--iters N] [--warmup N] [--dtype DTYPE ...]
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
    parser = argparse.ArgumentParser(description="MoE Quant Group GEMM micro benchmark")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to moe_quant_group_gemm.json (default: auto-detect)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: auto-generated)")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--dtype", type=str, nargs="+", default=None,
                        choices=["bfloat16", "fp8_per_tensor", "fp8_per_block", "mxfp4", "nvfp4"],
                        help="Dtypes to test (default: all)")
    return parser.parse_args()


# All supported dtypes
ALL_DTYPES = ["bfloat16", "fp8_per_tensor", "fp8_per_block", "mxfp4", "nvfp4"]


def find_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def find_json(script_dir):
    """Auto-detect moe_quant_group_gemm.json location."""
    candidates = [
        os.path.join(script_dir, "moe_quant_group_gemm.json"),
        os.path.join(script_dir, "..", "..", "xpu-perf", "projects", "micro_perf",
                     "workloads_report", "llm", "vendor_test_report", "moe_quant_group_gemm.json"),
        "/workspace/xpu-perf/projects/micro_perf/workloads_report/llm/vendor_test_report/moe_quant_group_gemm.json",
    ]
    for c in candidates:
        path = os.path.normpath(c)
        if os.path.isfile(path):
            return path
    return None


def load_cases(json_path):
    """Load test cases from moe_quant_group_gemm.json.

    Returns list of dicts: {num_tokens, num_experts, topk, hidden_size, new_hidden_size, ep_size}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    cases = []
    for entry in data.get("moe_quant_group_gemm", []):
        ep_sizes = entry.get("ep_size", [1])
        expert_configs = entry.get("num_experts.topk.hidden_size.new_hidden_size", [])
        num_tokens_list = entry.get("num_tokens", [])

        for ec in expert_configs:
            num_experts, topk, hidden_size, new_hidden_size = ec[0], ec[1], ec[2], ec[3]
            for ep_size in ep_sizes:
                for num_tokens in num_tokens_list:
                    cases.append({
                        "num_tokens": num_tokens,
                        "num_experts": num_experts,
                        "topk": topk,
                        "hidden_size": hidden_size,
                        "new_hidden_size": new_hidden_size,
                        "ep_size": ep_size,
                    })
    return cases


def run_command(cmd, timeout=600):
    """Run command and return stdout+stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"ERROR: {e}"


def parse_tflops_latency_bw(output):
    """Extract TFLOPS, latency and mem_bw from benchmark output."""
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


def run_bench(script_path, gpu_id, dtype_str, case, iters, warmup):
    """Run bench_moe_quant_group_gemm_with_flashinfer.py with given parameters."""
    cmd = [
        "python3", script_path,
        str(gpu_id),
        dtype_str,
        str(case["num_tokens"]),
        str(case["hidden_size"]),
        str(case["new_hidden_size"]),
        str(case["num_experts"]),
        str(case["topk"]),
        str(case["ep_size"]),
        str(iters),
        str(warmup),
    ]
    output = run_command(cmd)
    return parse_tflops_latency_bw(output)


def main():
    args = parse_args()
    script_dir = find_script_dir()

    # Find bench script
    bench_script = os.path.join(script_dir, "bench_moe_quant_group_gemm_with_flashinfer.py")
    if not os.path.isfile(bench_script):
        print(f"Error: cannot find bench_moe_quant_group_gemm_with_flashinfer.py in {script_dir}")
        sys.exit(1)

    # Find json
    json_path = args.json
    if json_path is None:
        json_path = find_json(script_dir)
    if json_path is None or not os.path.isfile(json_path):
        print("Error: cannot find moe_quant_group_gemm.json. Specify with --json")
        sys.exit(1)

    print(f"Loading cases from: {json_path}")
    cases = load_cases(json_path)
    print(f"Total parameter combinations: {len(cases)}")

    # Dtypes to test
    dtypes = args.dtype if args.dtype else ALL_DTYPES

    # Get GPU name
    import torch
    torch.cuda.set_device(args.gpu)
    gpu_name = torch.cuda.get_device_name(args.gpu).replace(" ", "_")

    # Output CSV
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(script_dir,
                                   f"moe_group_gemm_results_{gpu_name}_{timestamp}.csv")

    print(f"Output CSV: {args.output}")
    print(f"GPU: {args.gpu} ({gpu_name})")
    print(f"Iters: {args.iters}, Warmup: {args.warmup}")
    print(f"Dtypes: {dtypes}")
    print()

    # CSV fields
    fieldnames = ["backend", "dtype", "num_tokens", "hidden_size", "new_hidden_size",
                  "num_experts", "topk", "ep_size", "latency_ms", "tflops", "mem_bw_GBs"]

    csv_file = open(args.output, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    total_runs = len(dtypes) * len(cases)
    run_idx = 0

    for dtype_str in dtypes:
        print(f"{'='*60}")
        print(f" Dtype: {dtype_str}")
        print(f"{'='*60}")

        for case in cases:
            run_idx += 1
            tag = (f"tokens={case['num_tokens']}, E={case['num_experts']}, "
                   f"topk={case['topk']}, K={case['hidden_size']}, "
                   f"N={case['new_hidden_size']}, ep={case['ep_size']}")
            print(f"  [{run_idx}/{total_runs}] {dtype_str} {tag}", end=" ... ", flush=True)

            tflops, latency, mem_bw = run_bench(bench_script, args.gpu, dtype_str, case,
                                                  args.iters, args.warmup)

            writer.writerow({
                "backend": "flashinfer",
                "dtype": dtype_str,
                "num_tokens": case["num_tokens"],
                "hidden_size": case["hidden_size"],
                "new_hidden_size": case["new_hidden_size"],
                "num_experts": case["num_experts"],
                "topk": case["topk"],
                "ep_size": case["ep_size"],
                "latency_ms": f"{latency:.3f}",
                "tflops": f"{tflops:.2f}",
                "mem_bw_GBs": f"{mem_bw:.2f}",
            })
            csv_file.flush()
            print(f"{tflops:.2f} TFLOPS, {latency:.3f} ms, {mem_bw:.2f} GB/s")

    csv_file.close()

    print()
    print(f"{'='*60}")
    print(f" Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
