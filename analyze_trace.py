#!/usr/bin/env python3
"""Analyze Chrome trace JSON from torch.profiler to extract top CUDA kernels."""

import gzip
import json
import sys
from collections import defaultdict

def analyze_trace(trace_path: str, top_n: int = 50):
    print(f"Loading {trace_path} ...")
    with gzip.open(trace_path, "rt") as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    print(f"Total events: {len(events)}")

    # Categorize events
    kernel_times = defaultdict(lambda: {"count": 0, "total_us": 0, "min_us": float("inf"), "max_us": 0})
    cat_totals = defaultdict(float)
    cpu_op_times = defaultdict(lambda: {"count": 0, "total_us": 0})

    for ev in events:
        if ev.get("ph") != "X":  # duration events only
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        dur = ev.get("dur", 0)  # microseconds

        cat_totals[cat] += dur

        if "kernel" in cat.lower() or "cuda" in cat.lower() or cat == "gpu_memcpy":
            entry = kernel_times[name]
            entry["count"] += 1
            entry["total_us"] += dur
            entry["min_us"] = min(entry["min_us"], dur)
            entry["max_us"] = max(entry["max_us"], dur)

        if cat in ("cpu_op", "user_annotation", "python_function"):
            entry = cpu_op_times[name]
            entry["count"] += 1
            entry["total_us"] += dur

    # Print category breakdown
    print("\n" + "=" * 80)
    print("EVENT CATEGORY BREAKDOWN")
    print("=" * 80)
    total_all = sum(cat_totals.values()) or 1
    for cat, total_us in sorted(cat_totals.items(), key=lambda x: -x[1]):
        if total_us < 1000:  # skip trivial
            continue
        print(f"  {cat:<40s}  {total_us/1e6:>10.2f}s  ({total_us/total_all*100:>5.1f}%)")

    # Print top CUDA kernels
    print("\n" + "=" * 80)
    print(f"TOP {top_n} CUDA KERNELS BY TOTAL TIME (rank 0)")
    print("=" * 80)
    sorted_kernels = sorted(kernel_times.items(), key=lambda x: -x[1]["total_us"])

    total_kernel_us = sum(v["total_us"] for v in kernel_times.values()) or 1

    print(f"{'Kernel':<80s} {'Calls':>7s} {'Total(ms)':>11s} {'Avg(ms)':>10s} {'Min(ms)':>9s} {'Max(ms)':>9s} {'%':>7s}")
    print("-" * 140)
    for name, info in sorted_kernels[:top_n]:
        display_name = name[:80]
        avg_ms = info["total_us"] / info["count"] / 1000
        print(f"{display_name:<80s} {info['count']:>7d} {info['total_us']/1000:>11.2f} {avg_ms:>10.2f} {info['min_us']/1000:>9.2f} {info['max_us']/1000:>9.2f} {info['total_us']/total_kernel_us*100:>6.1f}%")

    print("-" * 140)
    print(f"{'TOTAL KERNEL TIME':<80s} {sum(v['count'] for v in kernel_times.values()):>7d} {total_kernel_us/1000:>11.2f}")

    # Print top CPU ops
    print("\n" + "=" * 80)
    print(f"TOP 30 CPU OPS BY TOTAL TIME")
    print("=" * 80)
    sorted_cpu = sorted(cpu_op_times.items(), key=lambda x: -x[1]["total_us"])
    print(f"{'Op':<80s} {'Calls':>7s} {'Total(ms)':>11s} {'Avg(ms)':>10s}")
    print("-" * 115)
    for name, info in sorted_cpu[:30]:
        display_name = name[:80]
        avg_ms = info["total_us"] / info["count"] / 1000
        print(f"{display_name:<80s} {info['count']:>7d} {info['total_us']/1000:>11.2f} {avg_ms:>10.2f}")

    # NCCL communication kernels
    print("\n" + "=" * 80)
    print("NCCL COMMUNICATION KERNELS")
    print("=" * 80)
    nccl_kernels = {k: v for k, v in kernel_times.items() if "nccl" in k.lower() or "ncclKernel" in k}
    if nccl_kernels:
        nccl_total = sum(v["total_us"] for v in nccl_kernels.values())
        for name, info in sorted(nccl_kernels.items(), key=lambda x: -x[1]["total_us"]):
            display_name = name[:80]
            avg_ms = info["total_us"] / info["count"] / 1000
            print(f"{display_name:<80s} {info['count']:>7d} {info['total_us']/1000:>11.2f}ms  avg={avg_ms:.2f}ms")
        print(f"\nTotal NCCL time: {nccl_total/1000:.2f}ms ({nccl_total/total_kernel_us*100:.1f}% of all kernel time)")
    else:
        print("  No NCCL kernels found.")

    # Save results
    output_path = trace_path.replace(".json.gz", "_analysis.txt")
    # Redirect stdout to file too
    return sorted_kernels, nccl_kernels


if __name__ == "__main__":
    trace_path = sys.argv[1] if len(sys.argv) > 1 else "/fact_home/qiliu/worldmodel/dreamzero/profile_output_8gpu/trace_rank0.json.gz"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    analyze_trace(trace_path, top_n)
