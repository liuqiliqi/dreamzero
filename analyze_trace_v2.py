#!/usr/bin/env python3
"""Analyze Chrome trace - separate CPU-side and GPU-side events properly."""

import gzip
import json
import sys
from collections import defaultdict

def analyze_trace(trace_path: str):
    print(f"Loading {trace_path} ...")
    with gzip.open(trace_path, "rt") as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    print(f"Total events: {len(events)}")

    # Separate GPU kernels vs CPU-side CUDA calls
    gpu_kernels = defaultdict(lambda: {"count": 0, "total_us": 0, "min_us": float("inf"), "max_us": 0})
    cpu_cuda_calls = defaultdict(lambda: {"count": 0, "total_us": 0, "min_us": float("inf"), "max_us": 0})

    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        dur = ev.get("dur", 0)

        # GPU-side: actual kernel execution on the device
        if cat in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"):
            entry = gpu_kernels[name]
            entry["count"] += 1
            entry["total_us"] += dur
            entry["min_us"] = min(entry["min_us"], dur)
            entry["max_us"] = max(entry["max_us"], dur)
            entry["cat"] = cat

        # CPU-side: CUDA runtime/driver calls on the host
        elif cat in ("cuda_runtime", "cuda_driver"):
            entry = cpu_cuda_calls[name]
            entry["count"] += 1
            entry["total_us"] += dur
            entry["min_us"] = min(entry["min_us"], dur)
            entry["max_us"] = max(entry["max_us"], dur)

    # ── GPU Kernels ──
    total_gpu_us = sum(v["total_us"] for v in gpu_kernels.values()) or 1
    sorted_gpu = sorted(gpu_kernels.items(), key=lambda x: -x[1]["total_us"])

    print("\n" + "=" * 90)
    print("GPU-SIDE KERNELS (actual device execution time)")
    print("=" * 90)
    print(f"{'Kernel':<80s} {'Cat':<20s} {'Calls':>7s} {'Total(ms)':>11s} {'Avg(ms)':>10s} {'%':>7s}")
    print("-" * 140)
    for name, info in sorted_gpu[:50]:
        pct = info["total_us"] / total_gpu_us * 100
        print(f"{name[:80]:<80s} {info.get('cat',''):<20s} {info['count']:>7d} {info['total_us']/1000:>11.2f} {info['total_us']/info['count']/1000:>10.2f} {pct:>6.1f}%")
    print("-" * 140)
    print(f"TOTAL GPU KERNEL TIME: {total_gpu_us/1000:.2f} ms")

    # Group GPU kernels by type
    print("\n" + "=" * 90)
    print("GPU KERNEL TIME BY TYPE")
    print("=" * 90)
    groups = {
        "GEMM (nvjet/cublas)": [],
        "Flash Attention": [],
        "NCCL Communication": [],
        "Triton fused ops": [],
        "Conv (cudnn)": [],
        "Elementwise": [],
        "Memcpy/Memset": [],
        "Other": [],
    }
    for name, info in gpu_kernels.items():
        cat = info.get("cat", "")
        if "nvjet" in name or "xmma" in name or "gemm" in name.lower() or "cutlass" in name.lower():
            groups["GEMM (nvjet/cublas)"].append(info)
        elif "flash" in name.lower() or "flash_fwd" in name or "flash_bwd" in name:
            groups["Flash Attention"].append(info)
        elif "nccl" in name.lower():
            groups["NCCL Communication"].append(info)
        elif "triton" in name:
            groups["Triton fused ops"].append(info)
        elif "fprop_implicit_gemm" in name or "cudnn" in name.lower() or "nhwc" in name:
            groups["Conv (cudnn)"].append(info)
        elif "elementwise" in name or "vectorized" in name or "Fill" in name:
            groups["Elementwise"].append(info)
        elif cat in ("gpu_memcpy", "gpu_memset"):
            groups["Memcpy/Memset"].append(info)
        else:
            groups["Other"].append(info)

    for group_name, items in sorted(groups.items(), key=lambda x: -sum(i["total_us"] for i in x[1])):
        total = sum(i["total_us"] for i in items)
        count = sum(i["count"] for i in items)
        if total < 100:
            continue
        print(f"  {group_name:<30s}  calls={count:>6d}  total={total/1000:>10.2f}ms  ({total/total_gpu_us*100:>5.1f}%)")

    # ── CPU-side CUDA calls ──
    total_cpu_us = sum(v["total_us"] for v in cpu_cuda_calls.values()) or 1
    sorted_cpu = sorted(cpu_cuda_calls.items(), key=lambda x: -x[1]["total_us"])

    print("\n" + "=" * 90)
    print("CPU-SIDE CUDA RUNTIME/DRIVER CALLS (host overhead)")
    print("=" * 90)
    print(f"{'Call':<50s} {'Calls':>7s} {'Total(ms)':>11s} {'Avg(ms)':>10s} {'Max(ms)':>10s} {'%':>7s}")
    print("-" * 100)
    for name, info in sorted_cpu[:20]:
        pct = info["total_us"] / total_cpu_us * 100
        print(f"{name[:50]:<50s} {info['count']:>7d} {info['total_us']/1000:>11.2f} {info['total_us']/info['count']/1000:>10.2f} {info['max_us']/1000:>10.2f} {pct:>6.1f}%")
    print("-" * 100)
    print(f"TOTAL CPU-SIDE CUDA CALL TIME: {total_cpu_us/1000:.2f} ms")

    print(f"\n{'='*90}")
    print("OVERLAP NOTE:")
    print(f"  GPU kernel total:     {total_gpu_us/1000:>10.2f} ms")
    print(f"  CPU CUDA call total:  {total_cpu_us/1000:>10.2f} ms")
    print(f"  These are on DIFFERENT timelines (GPU device vs CPU host).")
    print(f"  cudaDeviceSynchronize on CPU overlaps with GPU kernel execution.")
    print(f"  cudaLaunchKernel on CPU may or may not overlap with GPU work.")
    print(f"{'='*90}")


if __name__ == "__main__":
    trace_path = sys.argv[1] if len(sys.argv) > 1 else "/fact_home/qiliu/worldmodel/dreamzero/profile_output_8gpu/trace_rank0.json.gz"
    analyze_trace(trace_path)
