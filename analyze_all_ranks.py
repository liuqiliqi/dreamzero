#!/usr/bin/env python3
"""Cross-rank analysis of all 8 GPU traces for DreamZero 8-GPU profiling."""

import gzip
import json
import os
import sys
from collections import defaultdict

TRACE_DIR = sys.argv[1] if len(sys.argv) > 1 else "/fact_home/qiliu/worldmodel/dreamzero/profile_output_8gpu"
NUM_RANKS = 8

# TP group mapping: (2, 4) mesh => ip=2, tp=4
TP_GROUPS = {
    "Instance 0 (rank 0-3)": [0, 1, 2, 3],
    "Instance 1 (rank 4-7)": [4, 5, 6, 7],
}


def analyze_single_rank(rank: int):
    """Parse trace for one rank, return summary dict."""
    path = os.path.join(TRACE_DIR, f"trace_rank{rank}.json.gz")
    print(f"  Loading rank {rank} ({os.path.getsize(path) / 1e9:.1f} GB) ...", flush=True)

    with gzip.open(path, "rt") as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])

    gpu_kernels = defaultdict(lambda: {"count": 0, "total_us": 0, "min_us": float("inf"), "max_us": 0})
    cpu_cuda_calls = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0})

    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        dur = ev.get("dur", 0)

        if cat in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"):
            entry = gpu_kernels[name]
            entry["count"] += 1
            entry["total_us"] += dur
            entry["min_us"] = min(entry["min_us"], dur)
            entry["max_us"] = max(entry["max_us"], dur)
            entry["cat"] = cat
        elif cat in ("cuda_runtime", "cuda_driver"):
            entry = cpu_cuda_calls[name]
            entry["count"] += 1
            entry["total_us"] += dur
            entry["max_us"] = max(entry["max_us"], dur)

    # Classify kernels
    def classify(name, cat):
        if "nvjet" in name or "xmma" in name or "gemm" in name.lower() or "cutlass" in name.lower():
            return "GEMM"
        elif "flash" in name.lower():
            return "FlashAttention"
        elif "nccl" in name.lower():
            return "NCCL"
        elif "triton" in name:
            return "Triton"
        elif "fprop_implicit_gemm" in name or "cudnn" in name.lower() or "nhwc" in name:
            return "Conv"
        elif "elementwise" in name or "vectorized" in name or "Fill" in name:
            return "Elementwise"
        elif cat in ("gpu_memcpy", "gpu_memset"):
            return "MemOps"
        elif cat == "gpu_user_annotation":
            return "Annotation"
        else:
            return "Other"

    by_type = defaultdict(lambda: {"count": 0, "total_us": 0})
    for name, info in gpu_kernels.items():
        t = classify(name, info.get("cat", ""))
        by_type[t]["count"] += info["count"]
        by_type[t]["total_us"] += info["total_us"]

    total_gpu_us = sum(v["total_us"] for v in gpu_kernels.values())

    # Top kernels (excluding annotations)
    real_kernels = {k: v for k, v in gpu_kernels.items() if v.get("cat") != "gpu_user_annotation"}
    top_kernels = sorted(real_kernels.items(), key=lambda x: -x[1]["total_us"])[:20]

    # NCCL breakdown
    nccl_kernels = {k: v for k, v in gpu_kernels.items() if "nccl" in k.lower() and v.get("cat") == "kernel"}
    nccl_total_us = sum(v["total_us"] for v in nccl_kernels.values())

    # cudaDeviceSynchronize
    sync_info = cpu_cuda_calls.get("cudaDeviceSynchronize", {"count": 0, "total_us": 0, "max_us": 0})

    return {
        "rank": rank,
        "total_events": len(events),
        "total_gpu_us": total_gpu_us,
        "by_type": dict(by_type),
        "top_kernels": top_kernels,
        "nccl_kernels": nccl_kernels,
        "nccl_total_us": nccl_total_us,
        "sync_us": sync_info["total_us"],
        "sync_count": sync_info["count"],
        "sync_max_us": sync_info["max_us"],
        "total_real_kernel_us": sum(v["total_us"] for v in real_kernels.values()),
    }


def print_report(results):
    out = []
    def p(s=""):
        out.append(s)

    p("=" * 100)
    p("DREAMZERO 8-GPU CROSS-RANK PROFILING REPORT")
    p("Mesh: (2, 4) = 2 instances x 4 TP GPUs")
    p("Profile Level: 3 (torch.profiler Chrome trace)")
    p("=" * 100)

    # ── 1. Per-rank overview ──
    p("\n" + "─" * 100)
    p("1. PER-RANK GPU KERNEL TIME OVERVIEW")
    p("─" * 100)
    p(f"{'Rank':<6s} {'Total GPU(ms)':>14s} {'Real Kernel(ms)':>16s} {'NCCL(ms)':>10s} {'NCCL%':>7s} {'Sync(ms)':>10s} {'SyncMax(ms)':>12s} {'Events':>12s}")
    p("-" * 100)
    for r in results:
        nccl_pct = r["nccl_total_us"] / r["total_real_kernel_us"] * 100 if r["total_real_kernel_us"] > 0 else 0
        p(f"  {r['rank']:<4d} {r['total_gpu_us']/1000:>14.1f} {r['total_real_kernel_us']/1000:>16.1f} {r['nccl_total_us']/1000:>10.1f} {nccl_pct:>6.1f}% {r['sync_us']/1000:>10.1f} {r['sync_max_us']/1000:>12.1f} {r['total_events']:>12,d}")

    # ── 2. Load balance within TP groups ──
    p("\n" + "─" * 100)
    p("2. TP GROUP LOAD BALANCE")
    p("─" * 100)
    for group_name, ranks in TP_GROUPS.items():
        group_results = [r for r in results if r["rank"] in ranks]
        kernel_times = [r["total_real_kernel_us"] / 1000 for r in group_results]
        nccl_times = [r["nccl_total_us"] / 1000 for r in group_results]
        avg_k = sum(kernel_times) / len(kernel_times)
        max_k = max(kernel_times)
        min_k = min(kernel_times)
        imbalance = (max_k - min_k) / avg_k * 100 if avg_k > 0 else 0

        p(f"\n  {group_name}:")
        p(f"    Real kernel time: min={min_k:.1f}ms  max={max_k:.1f}ms  avg={avg_k:.1f}ms  imbalance={imbalance:.1f}%")
        p(f"    NCCL time:        min={min(nccl_times):.1f}ms  max={max(nccl_times):.1f}ms  avg={sum(nccl_times)/len(nccl_times):.1f}ms")
        for r in group_results:
            p(f"      rank {r['rank']}: kernel={r['total_real_kernel_us']/1000:.1f}ms  NCCL={r['nccl_total_us']/1000:.1f}ms")

    # ── 3. Cross-instance comparison ──
    p("\n" + "─" * 100)
    p("3. INSTANCE 0 vs INSTANCE 1 COMPARISON")
    p("─" * 100)
    for group_name, ranks in TP_GROUPS.items():
        group_results = [r for r in results if r["rank"] in ranks]
        total_kernel = sum(r["total_real_kernel_us"] for r in group_results)
        total_nccl = sum(r["nccl_total_us"] for r in group_results)
        p(f"  {group_name}: total_kernel={total_kernel/1000:.1f}ms  total_nccl={total_nccl/1000:.1f}ms")

    # ── 4. Kernel type breakdown (averaged across ranks) ──
    p("\n" + "─" * 100)
    p("4. KERNEL TYPE BREAKDOWN (AVERAGED ACROSS ALL RANKS)")
    p("─" * 100)
    all_types = set()
    for r in results:
        all_types.update(r["by_type"].keys())

    type_totals = {}
    for t in all_types:
        vals = [r["by_type"].get(t, {"total_us": 0})["total_us"] for r in results]
        type_totals[t] = {"avg_ms": sum(vals) / len(vals) / 1000, "min_ms": min(vals) / 1000, "max_ms": max(vals) / 1000}

    grand_avg = sum(v["avg_ms"] for v in type_totals.values())
    p(f"{'Type':<20s} {'Avg(ms)':>10s} {'Min(ms)':>10s} {'Max(ms)':>10s} {'Avg%':>7s} {'Spread%':>9s}")
    p("-" * 70)
    for t, info in sorted(type_totals.items(), key=lambda x: -x[1]["avg_ms"]):
        pct = info["avg_ms"] / grand_avg * 100 if grand_avg > 0 else 0
        spread = (info["max_ms"] - info["min_ms"]) / info["avg_ms"] * 100 if info["avg_ms"] > 0 else 0
        p(f"  {t:<18s} {info['avg_ms']:>10.1f} {info['min_ms']:>10.1f} {info['max_ms']:>10.1f} {pct:>6.1f}% {spread:>8.1f}%")

    # ── 5. Top kernels comparison across ranks ──
    p("\n" + "─" * 100)
    p("5. TOP 15 KERNELS — PER-RANK TIME (ms)")
    p("─" * 100)

    # Collect all kernel names from rank 0 top list, then show all ranks
    ref = results[0]
    top_names = [name for name, _ in ref["top_kernels"][:15]]

    # Build a full kernel->rank->time mapping
    kernel_rank_map = defaultdict(lambda: [0.0] * NUM_RANKS)
    for r in results:
        for name, info in r["top_kernels"]:
            kernel_rank_map[name][r["rank"]] = info["total_us"] / 1000

    header = f"{'Kernel':<60s}" + "".join(f"{'R'+str(i):>8s}" for i in range(NUM_RANKS)) + f"{'Spread%':>9s}"
    p(header)
    p("-" * (60 + 8 * NUM_RANKS + 9))
    for name in top_names:
        vals = kernel_rank_map[name]
        avg = sum(vals) / len(vals) if vals else 0
        spread = (max(vals) - min(vals)) / avg * 100 if avg > 0 else 0
        short = name[:58]
        row = f"{short:<60s}" + "".join(f"{v:>8.1f}" for v in vals) + f"{spread:>8.1f}%"
        p(row)

    # ── 6. NCCL breakdown per rank ──
    p("\n" + "─" * 100)
    p("6. NCCL COMMUNICATION BREAKDOWN PER RANK")
    p("─" * 100)
    all_nccl_names = set()
    for r in results:
        all_nccl_names.update(r["nccl_kernels"].keys())

    nccl_rank_map = defaultdict(lambda: [0.0] * NUM_RANKS)
    nccl_count_map = defaultdict(lambda: [0] * NUM_RANKS)
    for r in results:
        for name, info in r["nccl_kernels"].items():
            nccl_rank_map[name][r["rank"]] = info["total_us"] / 1000
            nccl_count_map[name][r["rank"]] = info["count"]

    header = f"{'NCCL Kernel':<70s}" + "".join(f"{'R'+str(i):>9s}" for i in range(NUM_RANKS))
    p(header)
    p("-" * (70 + 9 * NUM_RANKS))
    for name in sorted(all_nccl_names, key=lambda n: -sum(nccl_rank_map[n])):
        vals = nccl_rank_map[name]
        counts = nccl_count_map[name]
        short = name[:68]
        row = f"{short:<70s}" + "".join(f"{v:>9.1f}" for v in vals)
        p(row)
        row2 = f"{'  (calls)':<70s}" + "".join(f"{c:>9d}" for c in counts)
        p(row2)

    # ── 7. Bottleneck summary ──
    p("\n" + "─" * 100)
    p("7. BOTTLENECK ANALYSIS & OPTIMIZATION SUGGESTIONS")
    p("─" * 100)

    # Find the slowest rank
    slowest = max(results, key=lambda r: r["total_real_kernel_us"])
    fastest = min(results, key=lambda r: r["total_real_kernel_us"])
    p(f"\n  Slowest rank: {slowest['rank']} ({slowest['total_real_kernel_us']/1000:.1f}ms)")
    p(f"  Fastest rank: {fastest['rank']} ({fastest['total_real_kernel_us']/1000:.1f}ms)")
    p(f"  Overall imbalance: {(slowest['total_real_kernel_us'] - fastest['total_real_kernel_us']) / fastest['total_real_kernel_us'] * 100:.1f}%")

    # NCCL fraction
    avg_nccl = sum(r["nccl_total_us"] for r in results) / len(results)
    avg_kernel = sum(r["total_real_kernel_us"] for r in results) / len(results)
    p(f"\n  Avg NCCL fraction: {avg_nccl/avg_kernel*100:.1f}% of kernel time")

    # cudaDeviceSynchronize
    avg_sync = sum(r["sync_us"] for r in results) / len(results)
    p(f"  Avg cudaDeviceSynchronize: {avg_sync/1000:.1f}ms ({sum(r['sync_count'] for r in results)//len(results)} calls)")

    # Top kernel concentration
    ref_top5_total = sum(info["total_us"] for _, info in ref["top_kernels"][:5])
    ref_all = ref["total_real_kernel_us"]
    p(f"\n  Top 5 kernels (rank 0) account for {ref_top5_total/ref_all*100:.1f}% of kernel time")

    p("\n  Optimization priorities:")
    p("  1. GEMM kernels (nvjet) — largest compute consumer; consider FP8/INT8 quantization")
    p("  2. NCCL AllReduce — communication overhead; overlap with compute or reduce tensor sizes")
    p("  3. FlashAttention — already optimized, but check if sequence lengths can be reduced")
    p("  4. cudaDeviceSynchronize — unnecessary sync points; audit for removal")
    p("  5. Triton fused ops (LayerNorm etc.) — check if custom kernels can be fused further")

    p("\n" + "=" * 100)

    return "\n".join(out)


def main():
    import sys
    print("=" * 60, flush=True)
    print("Cross-rank analysis: 8 GPUs", flush=True)
    print("=" * 60, flush=True)

    results = []
    for rank in range(NUM_RANKS):
        r = analyze_single_rank(rank)
        results.append(r)
        print(f"  rank {rank} done: {r['total_real_kernel_us']/1000:.1f}ms kernel time, {r['nccl_total_us']/1000:.1f}ms NCCL", flush=True)

    report = print_report(results)
    print(report)

    # Save report
    output_path = os.path.join(TRACE_DIR, "cross_rank_report.txt")
    with open(output_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
