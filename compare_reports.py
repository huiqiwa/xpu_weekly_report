#!/usr/bin/env python3
"""Compare performance between two report directories.

Usage:
    python compare_reports.py <report_dir_1> <report_dir_2>

The newer/older order is determined by timestamp in directory names.
Output: per-op/provider latency ratio (old/new), so >1 means new is faster.
"""
import csv
import os
import statistics
import sys


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get_op_providers(base):
    result = {}
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".csv"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, base)
                parts = rel.split(os.sep)
                if len(parts) >= 3:
                    op, provider = parts[0], parts[1]
                    result[(op, provider)] = full
    return result


PERF_COLS = {
    "latency(us)", "read_bytes(B)", "write_bytes(B)", "io_bytes(B)",
    "mem_bw(GB/s)", "calc_flops", "calc_flops_power(tflops)",
    "calc_mem_ratio", "kernels", "sku_name", "op_name", "provider",
}


def compare(old_base, new_base, old_label, new_label):
    old_ops = get_op_providers(old_base)
    new_ops = get_op_providers(new_base)

    common_ops = sorted(set(old_ops) & set(new_ops))
    only_old = sorted(set(old_ops) - set(new_ops))
    only_new = sorted(set(new_ops) - set(old_ops))

    print(f"Common: {len(common_ops)} | Only {old_label}: {len(only_old)} | Only {new_label}: {len(only_new)}")
    if only_old:
        print(f"Only in {old_label}:")
        for op, prov in only_old:
            print(f"  {op}/{prov}")
    if only_new:
        print(f"Only in {new_label}:")
        for op, prov in only_new:
            print(f"  {op}/{prov}")

    print(f"\nRatio = new / old for tflops & bw.  >1 means new is better.  << = any tier deviates >5% from 1.0")
    print(f"Tiers: cases sorted by new value, split into low/mid/high (each ~33%), showing median ratio")
    print(f"diff: +N = new has N extra cases, -N = old has N extra cases\n")

    old_col = f"#old({old_label})"
    new_col = f"#new({new_label})"
    cw = max(len(old_col), len(new_col), 5)

    header = (
        f"{'op':>30} {'provider':>25} | "
        f"{old_col:>{cw}} {new_col:>{cw}} "
        f"{'diff':>6} | "
        f"{'tf_low':>7} {'tf_mid':>7} {'tf_high':>7} | "
        f"{'bw_low':>7} {'bw_mid':>7} {'bw_high':>7}"
    )
    print(header)
    print("-" * len(header))

    def tier_medians(vals):
        """Split sorted values into low/mid/high thirds, return median of each."""
        if not vals:
            return None, None, None
        s = sorted(vals)
        n = len(s)
        t1 = max(n // 3, 1)
        t2 = max(2 * n // 3, t1 + 1) if n > 1 else t1
        low = statistics.median(s[:t1])
        mid = statistics.median(s[t1:t2]) if t1 < t2 else low
        high = statistics.median(s[t2:]) if t2 < n else mid
        return low, mid, high

    for op, provider in common_ops:
        old_data = load_csv(old_ops[(op, provider)])
        new_data = load_csv(new_ops[(op, provider)])
        if not old_data or not new_data:
            continue

        key_cols_old = [c for c in old_data[0].keys() if c not in PERF_COLS]
        key_cols_new = [c for c in new_data[0].keys() if c not in PERF_COLS]
        key_cols = [c for c in key_cols_old if c in key_cols_new]
        old_map = {tuple(r.get(k, "") for k in key_cols): r for r in old_data}
        new_map = {tuple(r.get(k, "") for k in key_cols): r for r in new_data}
        common = set(old_map) & set(new_map)
        n_common = len(common)
        n_only_new = len(new_map) - n_common
        n_only_old = len(old_map) - n_common

        tflops_pairs = []  # (new_val, ratio)
        bw_pairs = []      # (new_val, ratio)
        for key in common:
            ot = float(old_map[key].get("calc_flops_power(tflops)", 0))
            nt = float(new_map[key].get("calc_flops_power(tflops)", 0))
            if ot > 0 and nt > 0:
                tflops_pairs.append((nt, nt / ot))
            ob = float(old_map[key].get("mem_bw(GB/s)", 0))
            nb = float(new_map[key].get("mem_bw(GB/s)", 0))
            if ob > 0 and nb > 0:
                bw_pairs.append((nb, nb / ob))

        # Sort by new value, then extract ratios in that order
        tflops_ratios = [r for _, r in sorted(tflops_pairs)]
        bw_ratios = [r for _, r in sorted(bw_pairs)]

        tf_lo, tf_mi, tf_hi = tier_medians(tflops_ratios)
        bw_lo, bw_mi, bw_hi = tier_medians(bw_ratios)

        diff = n_only_new - n_only_old
        s_diff = f"+{diff}" if diff > 0 else (str(diff) if diff < 0 else "")
        prefix = (
            f"{op:>30} {provider:>25} | "
            f"{len(old_data):>{cw}} {len(new_data):>{cw}} "
            f"{s_diff:>6} | "
        )

        def fmt_tier(lo, mi, hi):
            if lo is None:
                return f"{'':>7} {'':>7} {'':>7}"
            return f"{lo:>6.2f}x {mi:>6.2f}x {hi:>6.2f}x"

        flag = ""
        for v in [tf_lo, tf_mi, tf_hi, bw_lo, bw_mi, bw_hi]:
            if v is not None and abs(v - 1) > 0.05:
                flag = " <<"
                break

        tf_part = fmt_tier(tf_lo, tf_mi, tf_hi)
        bw_part = fmt_tier(bw_lo, bw_mi, bw_hi)
        print(f"{prefix}{tf_part} | {bw_part}{flag}")


def find_intel_base(report_dir):
    """Find the INTEL/GPU_NAME/ subdirectory containing op results."""
    intel_dir = os.path.join(report_dir, "INTEL")
    if not os.path.isdir(intel_dir):
        return None
    for name in os.listdir(intel_dir):
        candidate = os.path.join(intel_dir, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def extract_timestamp(dirname):
    import re
    m = re.search(r"(\d{4}(-\d{2}){5})", os.path.basename(dirname))
    return m.group(1).replace("-", "") if m else None


def make_label(dirname):
    """Extract a short label like '04-13' or '04-19' from the directory name."""
    import re
    m = re.search(r"\d{4}-(\d{2}-\d{2})-\d{2}-\d{2}-\d{2}", os.path.basename(dirname))
    return m.group(1) if m else os.path.basename(dirname)[:10]


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <report_dir_1> <report_dir_2>")
        sys.exit(1)

    dir1 = os.path.realpath(sys.argv[1])
    dir2 = os.path.realpath(sys.argv[2])

    for d in [dir1, dir2]:
        if not os.path.isdir(d):
            print(f"[ERROR] Not a directory: {d}")
            sys.exit(1)

    ts1 = extract_timestamp(dir1)
    ts2 = extract_timestamp(dir2)

    if ts1 and ts2:
        if ts1 <= ts2:
            old_dir, new_dir = dir1, dir2
        else:
            old_dir, new_dir = dir2, dir1
    else:
        old_dir, new_dir = dir1, dir2

    old_base = find_intel_base(old_dir)
    new_base = find_intel_base(new_dir)

    if not old_base:
        print(f"[ERROR] No INTEL/<gpu>/ found in {old_dir}")
        sys.exit(1)
    if not new_base:
        print(f"[ERROR] No INTEL/<gpu>/ found in {new_dir}")
        sys.exit(1)

    old_label = make_label(old_dir)
    new_label = make_label(new_dir)

    # Output to file in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts_old = extract_timestamp(old_dir) or "unknown"
    ts_new = extract_timestamp(new_dir) or "unknown"
    # Format timestamps as YYYY-MM-DD
    def fmt_ts(ts):
        if len(ts) >= 8:
            return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        return ts
    out_file = os.path.join(script_dir, f"compare_result_{fmt_ts(ts_old)}_vs_{fmt_ts(ts_new)}.txt")

    import io
    buf = io.StringIO()
    orig_stdout = sys.stdout

    # Print to both stdout and buffer
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(orig_stdout, buf)

    print(f"Old: {old_dir}")
    print(f"New: {new_dir}")
    print()

    compare(old_base, new_base, old_label, new_label)

    sys.stdout = orig_stdout

    with open(out_file, "w") as f:
        f.write(buf.getvalue())
    print(f"\nResult saved to: {out_file}")


if __name__ == "__main__":
    main()
