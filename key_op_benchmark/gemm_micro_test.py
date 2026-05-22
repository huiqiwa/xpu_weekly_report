#!/usr/bin/env python3
"""
GEMM micro benchmark: sweep M/K/N from gemm.json, test 3 backends × 4 dtypes.

Backends:
  1. bench_gemm.py          (PyTorch)
  2. bench_gemm_with_flashinfer.py (FlashInfer, skip FP32)
  3. cublas_gemm binaries   (gemm_fp32, gemm_bf16, gemm_fp8, gemm_nvfp4)

Dtypes: float32, bfloat16, fp8_e4m3, nvfp4

Usage:
    python gemm_micro_test.py [--gpu GPU_ID] [--json GEMM_JSON] [--output CSV] [--iters N] [--warmup N]
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
    parser = argparse.ArgumentParser(description="GEMM micro benchmark grid search")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to gemm.json (default: auto-detect)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: gemm_results_<timestamp>.csv)")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--backend", type=str, nargs="+", default=None,
                        choices=["pytorch", "flashinfer", "cublas"],
                        help="Backends to test (default: all)")
    return parser.parse_args()


# Dtype configurations
# (display_name, bench_gemm_dtype, flashinfer_dtype, cublas_binary)
DTYPE_CONFIGS = [
    ("fp32",    "float32",   None,        "gemm_fp32"),
    ("bf16",    "bfloat16",  "bfloat16",  "gemm_bf16"),
    ("fp8",     "fp8_e4m3",  "fp8_e4m3",  "gemm_fp8"),
    ("nvfp4",   "nvfp4",     "nvfp4",     "gemm_nvfp4"),
]


def find_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def find_gemm_json(script_dir):
    """Auto-detect gemm.json location."""
    candidates = [
        os.path.join(script_dir, "gemm.json"),
        os.path.join(script_dir, "..", "..", "xpu-perf", "projects", "micro_perf",
                     "workloads_report", "basic", "tensor_gemm_ops", "gemm.json"),
    ]
    for c in candidates:
        path = os.path.normpath(c)
        if os.path.isfile(path):
            return path
    return None


def load_cases(json_path):
    """Load M, K, N combinations from gemm.json."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    cases = []
    for case in data.get("cases", []):
        m_list = case.get("M", [])
        kn_list = case.get("K.N", [])
        for kn in kn_list:
            K, N = kn[0], kn[1]
            for M in m_list:
                cases.append((M, K, N))
    return cases


def run_command(cmd, timeout=300):
    """Run command and return stdout."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"ERROR: {e}"


def parse_tflops_latency(output):
    """Extract TFLOPS and latency from benchmark output."""
    tflops = 0.0
    latency = 0.0
    m = re.search(r'Avg TFLOPS:\s+([0-9.]+)', output)
    if m:
        tflops = float(m.group(1))
    m = re.search(r'Avg latency:\s+([0-9.]+)', output)
    if m:
        latency = float(m.group(1))
    return tflops, latency


def run_bench_gemm(script_dir, gpu_id, dtype_str, M, K, N, iters, warmup):
    """Run bench_gemm.py (PyTorch backend)."""
    script = os.path.join(script_dir, "bench_gemm.py")
    if not os.path.isfile(script):
        return 0.0, 0.0
    cmd = ["python3", script, str(gpu_id), dtype_str, str(M), str(K), str(N),
           str(iters), str(warmup)]
    output = run_command(cmd)
    return parse_tflops_latency(output)


def run_flashinfer(script_dir, gpu_id, dtype_str, M, K, N, iters, warmup):
    """Run bench_gemm_with_flashinfer.py."""
    script = os.path.join(script_dir, "bench_gemm_with_flashinfer.py")
    if not os.path.isfile(script):
        return 0.0, 0.0
    cmd = ["python3", script, str(gpu_id), dtype_str, str(M), str(K), str(N),
           str(iters), str(warmup)]
    output = run_command(cmd)
    return parse_tflops_latency(output)


def run_cublas(script_dir, gpu_id, binary_name, M, K, N, iters, warmup):
    """Run cublas gemm per-dtype binary."""
    binary = os.path.join(script_dir, "cublas_gemm", binary_name)
    if not os.path.isfile(binary):
        return 0.0, 0.0
    cmd = [binary, str(gpu_id), str(M), str(K), str(N), str(iters), str(warmup)]
    output = run_command(cmd)
    return parse_tflops_latency(output)


def main():
    args = parse_args()
    script_dir = find_script_dir()

    # Find gemm.json
    json_path = args.json
    if json_path is None:
        json_path = find_gemm_json(script_dir)
    if json_path is None or not os.path.isfile(json_path):
        print(f"Error: cannot find gemm.json. Specify with --json")
        sys.exit(1)

    print(f"Loading cases from: {json_path}")
    cases = load_cases(json_path)
    print(f"Total M×K×N combinations: {len(cases)}")

    # Get GPU name for output filename
    import torch
    torch.cuda.set_device(args.gpu)
    gpu_name = torch.cuda.get_device_name(args.gpu).replace(" ", "_")

    # Output CSV
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(script_dir, f"gemm_results_{gpu_name}_{timestamp}.csv")

    print(f"Output CSV: {args.output}")
    print(f"GPU: {args.gpu} ({gpu_name})")
    print(f"Iters: {args.iters}, Warmup: {args.warmup}")
    print(f"Dtypes: {[d[0] for d in DTYPE_CONFIGS]}")
    print()

    # CSV header: each backend is a separate row, tflops last
    fieldnames = ["backend", "dtype", "M", "K", "N", "latency_ms", "tflops"]

    # Open CSV for incremental writing
    csv_file = open(args.output, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    # Backends to test in order
    BACKENDS = args.backend if args.backend else ["pytorch", "flashinfer", "cublas"]

    total_runs = len(BACKENDS) * len(DTYPE_CONFIGS) * len(cases)
    run_idx = 0

    for backend in BACKENDS:
        print(f"{'='*60}")
        print(f" Backend: {backend}")
        print(f"{'='*60}")

        for dtype_name, bench_dtype, fi_dtype, cublas_bin in DTYPE_CONFIGS:
            # FlashInfer skips FP32
            if backend == "flashinfer" and fi_dtype is None:
                print(f"  [skip] {backend} / {dtype_name}")
                continue

            print(f"  --- {dtype_name} ---")

            for M, K, N in cases:
                run_idx += 1
                print(f"  [{run_idx}/{total_runs}] {backend} {dtype_name} M={M}, K={K}, N={N}", end=" ... ", flush=True)

                if backend == "pytorch":
                    tflops, latency = run_bench_gemm(
                        script_dir, args.gpu, bench_dtype, M, K, N, args.iters, args.warmup)
                elif backend == "flashinfer":
                    tflops, latency = run_flashinfer(
                        script_dir, args.gpu, fi_dtype, M, K, N, args.iters, args.warmup)
                elif backend == "cublas":
                    tflops, latency = run_cublas(
                        script_dir, args.gpu, cublas_bin, M, K, N, args.iters, args.warmup)

                writer.writerow({"dtype": dtype_name, "M": M, "K": K, "N": N,
                                 "backend": backend, "latency_ms": f"{latency:.3f}",
                                 "tflops": f"{tflops:.2f}"})
                csv_file.flush()
                print(f"{tflops:.2f} TFLOPS, {latency:.3f} ms")

    csv_file.close()

    print()
    print(f"{'='*60}")
    print(f" Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
