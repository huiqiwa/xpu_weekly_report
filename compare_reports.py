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
    # collective communication columns
    "algo_size(B)", "bus_size(B)", "algo_bw(GB/s)", "bus_bw(GB/s)",
    "algo_bw_sum(GB/s)", "bus_bw_sum(GB/s)",
    "latency_list(us)", "algo_bw_list(GB/s)", "bus_bw_list(GB/s)",
}


def is_collective_op(csv_path):
    """Check if a CSV file is a collective communication op (has bus_bw column)."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        return header is not None and "bus_bw(GB/s)" in header


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


def fmt_val(v):
    """Format a single ratio value with inline arrow, padded to 9 chars."""
    if v is None:
        return " " * 9
    if v > 1.05:
        arrow = " ↑"
    elif v < 0.95:
        arrow = " ↓"
    else:
        arrow = "  "
    s = f"{v:.2f}x{arrow}"
    return f"{s:>9}"


def fmt_tier(lo, mi, hi):
    if lo is None:
        return f"{'':>9} {'':>9} {'':>9}"
    return f"{fmt_val(lo)} {fmt_val(mi)} {fmt_val(hi)}"


def compute_ratios(old_ops, new_ops, common_ops, bw_col="mem_bw(GB/s)"):
    """Compute per-op ratio data grouped by dtype. Returns list of result dicts."""
    results = []
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

        # Find dtype column index in key_cols
        dtype_idx = None
        if "dtype" in key_cols:
            dtype_idx = key_cols.index("dtype")

        # Group by dtype
        dtype_groups = {}  # dtype -> list of (old_row, new_row)
        for key in common:
            dt = key[dtype_idx] if dtype_idx is not None else "all"
            dtype_groups.setdefault(dt, []).append((old_map[key], new_map[key]))

        # Also count per-dtype sizes
        dtype_old_counts = {}
        dtype_new_counts = {}
        for r in old_data:
            dt = r.get("dtype", "all")
            dtype_old_counts[dt] = dtype_old_counts.get(dt, 0) + 1
        for r in new_data:
            dt = r.get("dtype", "all")
            dtype_new_counts[dt] = dtype_new_counts.get(dt, 0) + 1

        all_dtypes = sorted(set(list(dtype_groups.keys()) + list(dtype_old_counts.keys()) + list(dtype_new_counts.keys())))

        for dt in all_dtypes:
            pairs = dtype_groups.get(dt, [])
            n_old = dtype_old_counts.get(dt, 0)
            n_new = dtype_new_counts.get(dt, 0)
            n_common = len(pairs)

            tflops_pairs = []
            bw_pairs = []
            for old_row, new_row in pairs:
                ot = float(old_row.get("calc_flops_power(tflops)", 0))
                nt = float(new_row.get("calc_flops_power(tflops)", 0))
                if ot > 0 and nt > 0:
                    tflops_pairs.append((nt, nt / ot))
                ob = float(old_row.get(bw_col, 0))
                nb = float(new_row.get(bw_col, 0))
                if ob > 0 and nb > 0:
                    bw_pairs.append((nb, nb / ob))

            tflops_ratios = [r for _, r in sorted(tflops_pairs)]
            bw_ratios = [r for _, r in sorted(bw_pairs)]

            results.append({
                "op": op, "provider": provider, "dtype": dt,
                "n_old": n_old, "n_new": n_new,
                "diff": (n_new - n_common) - (n_old - n_common),
                "tf_tiers": tier_medians(tflops_ratios),
                "bw_tiers": tier_medians(bw_ratios),
            })
    return results


def compare(old_base, new_base, old_label, new_label):
    old_ops = get_op_providers(old_base)
    new_ops = get_op_providers(new_base)

    common_ops = sorted(set(old_ops) & set(new_ops))
    only_old = sorted(set(old_ops) - set(new_ops))
    only_new = sorted(set(new_ops) - set(old_ops))

    # Classify ops: collective vs compute
    collective_ops = []
    compute_ops = []
    for op, provider in common_ops:
        if is_collective_op(new_ops[(op, provider)]):
            collective_ops.append((op, provider))
        else:
            compute_ops.append((op, provider))

    print(f"Common: {len(common_ops)} (compute: {len(compute_ops)}, collective: {len(collective_ops)})")
    print(f"Only {old_label}: {len(only_old)} | Only {new_label}: {len(only_new)}")
    if only_old:
        print(f"Only in {old_label}:")
        for op, prov in only_old:
            print(f"  {op}/{prov}")
    if only_new:
        print(f"Only in {new_label}:")
        for op, prov in only_new:
            print(f"  {op}/{prov}")

    old_col = f"#old({old_label})"
    new_col = f"#new({new_label})"
    cw = max(len(old_col), len(new_col), 5)

    # --- Compute / Memory ops table ---
    print(f"\n=== Compute / Memory Ops ===")
    print(f"Ratio = new / old for tflops & mem_bw.  >1 means new is better.")
    print(f"↑ = up >5%, ↓ = down >5%")
    print(f"Tiers: cases sorted by new value, split into low/mid/high (each ~33%), showing median ratio")
    print(f"diff: +N = new has N extra cases, -N = old has N extra cases\n")

    header = (
        f"{'op':>30} {'provider':>25} {'dtype':>8} | "
        f"{old_col:>{cw}} {new_col:>{cw}} "
        f"{'diff':>6} | "
        f"{'tf_low':>9} {'tf_mid':>9} {'tf_high':>9} | "
        f"{'bw_low':>9} {'bw_mid':>9} {'bw_high':>9}"
    )
    print(header)
    print("-" * len(header))

    for r in compute_ratios(old_ops, new_ops, compute_ops, bw_col="mem_bw(GB/s)"):
        diff = r["diff"]
        s_diff = f"+{diff}" if diff > 0 else (str(diff) if diff < 0 else "")
        tf_lo, tf_mi, tf_hi = r["tf_tiers"]
        bw_lo, bw_mi, bw_hi = r["bw_tiers"]
        prefix = (
            f"{r['op']:>30} {r['provider']:>25} {r['dtype']:>8} | "
            f"{r['n_old']:>{cw}} {r['n_new']:>{cw}} "
            f"{s_diff:>6} | "
        )
        print(f"{prefix}{fmt_tier(tf_lo, tf_mi, tf_hi)} | {fmt_tier(bw_lo, bw_mi, bw_hi)}")

    # --- Collective communication ops table ---
    if collective_ops:
        print(f"\n=== Collective Communication Ops (grouped by world_size) ===")
        print(f"Ratio = new / old for bus_bw.  >1 means new is better.")
        print(f"↑ = up >5%, ↓ = down >5%\n")

        coll_header = (
            f"{'op':>30} {'provider':>25} {'world_size':>10} {'dtype':>8} | "
            f"{old_col:>{cw}} {new_col:>{cw}} "
            f"{'diff':>6} | "
            f"{'bw_low':>9} {'bw_mid':>9} {'bw_high':>9}"
        )
        print(coll_header)
        print("-" * len(coll_header))

        # Collect all rows then sort by (world_size, op, provider, dtype)
        coll_rows = []
        for op, provider in collective_ops:
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

            # Group by (world_size, dtype)
            groups = {}  # (ws, dtype) -> list of (old_row, new_row)
            for key in common:
                old_row = old_map[key]
                new_row = new_map[key]
                ws = old_row.get("world_size", "")
                dt = old_row.get("dtype", "all")
                groups.setdefault((ws, dt), []).append((old_row, new_row))

            # Count per (ws, dtype)
            old_counts = {}
            new_counts = {}
            for r in old_data:
                k = (r.get("world_size", ""), r.get("dtype", "all"))
                old_counts[k] = old_counts.get(k, 0) + 1
            for r in new_data:
                k = (r.get("world_size", ""), r.get("dtype", "all"))
                new_counts[k] = new_counts.get(k, 0) + 1

            all_keys = sorted(set(list(groups.keys()) + list(old_counts.keys()) + list(new_counts.keys())),
                              key=lambda x: (int(x[0]) if x[0].isdigit() else 0, x[1]))

            for ws, dt in all_keys:
                pairs = groups.get((ws, dt), [])
                n_old = old_counts.get((ws, dt), 0)
                n_new = new_counts.get((ws, dt), 0)
                n_common = len(pairs)
                if not pairs:
                    continue

                bw_pairs = []
                for old_row, new_row in pairs:
                    ob = float(old_row.get("bus_bw(GB/s)", 0))
                    nb = float(new_row.get("bus_bw(GB/s)", 0))
                    if ob > 0 and nb > 0:
                        bw_pairs.append((nb, nb / ob))

                bw_ratios = [r for _, r in sorted(bw_pairs)]
                bw_lo, bw_mi, bw_hi = tier_medians(bw_ratios)

                diff = (n_new - n_common) - (n_old - n_common)
                s_diff = f"+{diff}" if diff > 0 else (str(diff) if diff < 0 else "")
                coll_rows.append((int(ws) if ws.isdigit() else 0, op, provider, dt,
                                  n_old, n_new, s_diff, bw_lo, bw_mi, bw_hi))

        for ws_int, op, provider, dt, n_old, n_new, s_diff, bw_lo, bw_mi, bw_hi in sorted(coll_rows):
            prefix = (
                f"{op:>30} {provider:>25} {ws_int:>10} {dt:>8} | "
                f"{n_old:>{cw}} {n_new:>{cw}} "
                f"{s_diff:>6} | "
            )
            print(f"{prefix}{fmt_tier(bw_lo, bw_mi, bw_hi)}")


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
