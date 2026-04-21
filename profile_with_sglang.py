#!/usr/bin/env python3
"""Profile DreamZero inference using SGLang's profiling tools.

Three profiling levels:
  1 — Stage-level GPU timing via DeviceTimer (low overhead)
  2 — NVTX per-layer hooks on DiT via PytHooks (run under nsys)
  3 — Full kernel-level Chrome trace via torch.profiler (single chunk)

Usage:
    # Level 1: Quick stage-level timing
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
        --standalone --nproc_per_node=2 profile_with_sglang.py \
        --model-path /fact_data/qiliu/dreamzero_weights/DreamZero-DROID \
        --profile-level 1 --num-chunks 5

    # Level 2: NVTX (run under nsys)
    nsys profile -o dreamzero_nvtx python -m torch.distributed.run \
        --standalone --nproc_per_node=2 profile_with_sglang.py \
        --model-path /fact_data/qiliu/dreamzero_weights/DreamZero-DROID \
        --profile-level 2 --no-compile --num-chunks 3

    # Level 3: Kernel-level Chrome trace
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
        --standalone --nproc_per_node=2 profile_with_sglang.py \
        --model-path /fact_data/qiliu/dreamzero_weights/DreamZero-DROID \
        --profile-level 3 --output-dir ./profile_output --num-chunks 5
"""

import dataclasses
import datetime
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from functools import wraps

import importlib.util

import cv2
import numpy as np
import torch
import torch.distributed as dist
import tyro
from torch.distributed.device_mesh import init_device_mesh

# Import SGLang profiling tools directly from file to avoid pulling in the full
# sglang package (which has heavy dependencies like pybase64 that may not be installed).
_SGLANG_UTILS = os.path.join(
    os.path.dirname(__file__), "..", "..", "GPU_ana", "sglang", "python",
    "sglang", "srt", "utils",
)


def _import_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_device_timer_mod = _import_from_file(
    "sglang_device_timer", os.path.join(_SGLANG_UTILS, "device_timer.py")
)
DeviceTimer = _device_timer_mod.DeviceTimer

logger = logging.getLogger(__name__)

# ── Constants from local_benchmark.py ──
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "debug_image")
CAMERA_FILES = {
    "video.exterior_image_1_left": "exterior_image_1_left.mp4",
    "video.exterior_image_2_left": "exterior_image_2_left.mp4",
    "video.wrist_image_left": "wrist_image_left.mp4",
}
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"


@dataclasses.dataclass
class Args:
    model_path: str = "/fact_data/qiliu/dreamzero_weights/DreamZero-DROID"
    enable_dit_cache: bool = False
    num_chunks: int = 5
    warmup_chunks: int = 2
    profile_level: int = 1  # 1, 2, or 3
    output_dir: str = "./profile_output"
    no_compile: bool = False


# ── Reuse infrastructure from local_benchmark ──

def init_mesh():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    if world_size <= 2:
        mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("ip",))
    else:
        tp_size = world_size // 2
        mesh = init_device_mesh("cuda", mesh_shape=(2, tp_size), mesh_dim_names=("ip", "tp"))
    logger.info(f"Rank {rank}/{world_size} on cuda:{rank}, mesh={mesh}")
    return mesh


def load_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def build_obs(camera_frames: dict, frame_indices: list[int], prompt: str) -> dict:
    obs = {}
    for cam_key, all_frames in camera_frames.items():
        obs[cam_key] = all_frames[frame_indices]
    obs["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)
    obs["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)
    obs["annotation.language.action_text"] = prompt
    return obs


def broadcast_obs(obs: dict | None, rank: int):
    if rank == 0:
        serialized = pickle.dumps(obs)
        size = torch.tensor([len(serialized)], dtype=torch.int64, device="cuda")
        dist.broadcast(size, src=0)
        data = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data, src=0)
        return obs
    else:
        size = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(size, src=0)
        data = torch.zeros(size.item(), dtype=torch.uint8, device="cuda")
        dist.broadcast(data, src=0)
        return pickle.loads(data.cpu().numpy().tobytes())


def run_inference(policy, obs: dict, rank: int):
    from tianshou.data import Batch
    batch = Batch(obs=obs)
    dist.barrier()
    with torch.no_grad():
        result_batch, video_pred = policy.lazy_joint_forward_causal(batch)
    dist.barrier()
    return result_batch, video_pred


# ── Level 1: Stage-level profiling with DeviceTimer ──

class StageProfiler:
    """Monkey-patches action head methods to wrap them with DeviceTimer.

    Each patched method gets its own DeviceTimer instance to avoid issues
    with nested calls (e.g. encode_image internally calls vae.encode).
    """

    def __init__(self):
        self.records: list[dict] = []
        self._step_counter = 0
        self._timers: list[DeviceTimer] = []

    def _on_record(self, t: float, **metadata):
        metadata["duration_s"] = t
        self.records.append(metadata)

    def reset_step_counter(self):
        self._step_counter = 0

    def install(self, action_head):
        """Monkey-patch methods on the WANPolicyHead instance."""
        self._patch_method(action_head, "encode_prompt", "text_encoder")
        self._patch_method(action_head, "encode_image", "image_encoder")
        self._patch_method(action_head, "_run_diffusion_steps", "diffusion_step")
        # Patch vae.encode
        self._patch_method(action_head.vae, "encode", "vae_encode")
        # Patch DiT model forward
        self._patch_method(action_head.model, "__call__", "dit_forward")

    def _patch_method(self, obj, method_name: str, stage_name: str):
        original = getattr(obj, method_name)
        profiler = self
        # Each patched method gets its own timer so nested calls don't collide
        timer = DeviceTimer(reporter=self._on_record)
        self._timers.append(timer)

        @wraps(original)
        def wrapper(*args, **kwargs):
            step = profiler._step_counter
            with timer.wrap({"stage": stage_name, "step": step}):
                result = original(*args, **kwargs)
            if stage_name == "diffusion_step":
                profiler._step_counter += 1
            return result

        setattr(obj, method_name, wrapper)

    def flush(self):
        """Force-flush any remaining timer intervals by synchronizing CUDA."""
        torch.cuda.synchronize()
        for timer in self._timers:
            timer._report()

    def summarize(self) -> str:
        """Aggregate records and produce a summary table."""
        if not self.records:
            return "No profiling records collected."

        by_stage: dict[str, list[float]] = defaultdict(list)
        for rec in self.records:
            by_stage[rec["stage"]].append(rec["duration_s"])

        total_all = sum(d for durations in by_stage.values() for d in durations)

        lines = []
        lines.append(f"{'Stage':<25s} {'Calls':>6s} {'Avg(ms)':>10s} {'Total(ms)':>10s} {'%':>7s}")
        lines.append("-" * 62)
        for stage, durations in sorted(by_stage.items()):
            n = len(durations)
            total_ms = sum(durations) * 1000
            avg_ms = total_ms / n
            pct = (sum(durations) / total_all * 100) if total_all > 0 else 0
            lines.append(f"{stage:<25s} {n:>6d} {avg_ms:>10.2f} {total_ms:>10.2f} {pct:>6.1f}%")
        lines.append("-" * 62)
        lines.append(f"{'TOTAL':<25s} {'':<6s} {'':<10s} {total_all*1000:>10.2f} {'100.0%':>7s}")
        return "\n".join(lines)


# ── Level 2: NVTX hooks via PytHooks ──

def install_nvtx_hooks(action_head):
    """Register PytHooks on the DiT model, skipping duplicate module instances."""
    _nvtx_mod = _import_from_file(
        "sglang_nvtx_hooks", os.path.join(_SGLANG_UTILS, "nvtx_pytorch_hooks.py")
    )
    PytHooks = _nvtx_mod.PytHooks

    hooks = PytHooks()

    # Override register_hooks to skip duplicates instead of raising ValueError
    original_register = hooks.register_hooks

    def safe_register_hooks(network_model, module_prefix="top"):
        skip_types = (
            torch.nn.Identity,
            torch.nn.Dropout,
            torch.nn.Dropout1d,
            torch.nn.Dropout2d,
            torch.nn.Dropout3d,
        )
        for name, module in network_model.named_modules(prefix=module_prefix):
            if isinstance(module, skip_types):
                continue
            if module in hooks.module_to_name_map:
                continue  # skip duplicates instead of raising
            module.register_forward_pre_hook(hooks.module_fwd_pre_hook)
            module.register_forward_hook(hooks.module_fwd_hook)
            hooks.module_to_name_map[module] = name

    safe_register_hooks(action_head.model, module_prefix="dit")
    logger.info(f"NVTX hooks registered on {len(hooks.module_to_name_map)} modules")
    return hooks


# ── Main ──

def main(args: Args):
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    if not args.no_compile:
        torch._dynamo.config.recompile_limit = 800

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device_mesh = init_mesh()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    timeout_delta = datetime.timedelta(seconds=50000)
    dist.new_group(backend="gloo", timeout=timeout_delta)

    logger.info(f"Rank {rank}: loading model...")

    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=args.model_path,
        device="cuda",
        device_mesh=device_mesh,
    )
    logger.info(f"Rank {rank}: model loaded")

    action_head = policy.trained_model.action_head

    # ── Install profiling hooks ──
    stage_profiler = None
    if args.profile_level >= 1:
        stage_profiler = StageProfiler()
        stage_profiler.install(action_head)
        logger.info("Level 1: StageProfiler installed (DeviceTimer)")

    if args.profile_level == 2:
        if not args.no_compile:
            logger.warning("Level 2 NVTX hooks require --no-compile to avoid torch.compile conflicts")
        install_nvtx_hooks(action_head)
        logger.info("Level 2: NVTX hooks installed on DiT model")

    # ── Load video data (rank 0 only) ──
    if rank == 0:
        logger.info("Loading video frames...")
        camera_frames = {}
        for cam_key, fname in CAMERA_FILES.items():
            path = os.path.join(VIDEO_DIR, fname)
            camera_frames[cam_key] = load_video(path)
            logger.info(f"  {cam_key}: {camera_frames[cam_key].shape}")

        total_frames = min(v.shape[0] for v in camera_frames.values())

        chunks = []
        current = 23
        for _ in range(args.num_chunks):
            indices = [max(current + off, 0) for off in RELATIVE_OFFSETS]
            if indices[-1] >= total_frames:
                break
            chunks.append(indices)
            current += ACTION_HORIZON

        logger.info(f"Will run: 1 warmup-initial + {len(chunks)} chunks")

    # ── Warmup: initial frame ──
    if rank == 0:
        logger.info("=== Warmup: initial frame [0] ===")
        obs = build_obs(camera_frames, [0], PROMPT)
    else:
        obs = None

    obs = broadcast_obs(obs, rank)
    run_inference(policy, obs, rank)

    # Clear warmup records from profiler
    if stage_profiler is not None:
        stage_profiler.flush()
        stage_profiler.records.clear()
        stage_profiler.reset_step_counter()

    # ── Broadcast num_chunks ──
    num_chunks = len(chunks) if rank == 0 else 0
    nc_tensor = torch.tensor([num_chunks if rank == 0 else 0], dtype=torch.int32, device="cuda")
    dist.broadcast(nc_tensor, src=0)
    num_chunks = nc_tensor.item()

    # ── Benchmark loop ──
    times = []
    profiled_chunk_idx = args.warmup_chunks  # first post-warmup chunk for Level 3

    for i in range(num_chunks):
        if rank == 0:
            obs = build_obs(camera_frames, chunks[i], PROMPT)
        else:
            obs = None

        obs = broadcast_obs(obs, rank)

        if stage_profiler is not None:
            stage_profiler.reset_step_counter()

        # Level 3: wrap exactly one chunk with torch.profiler
        use_torch_profiler = (args.profile_level >= 3 and i == profiled_chunk_idx)

        if use_torch_profiler:
            os.makedirs(args.output_dir, exist_ok=True)
            trace_path = os.path.join(args.output_dir, f"trace_rank{rank}.json.gz")
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
            )
            prof.__enter__()

        t0 = time.time()
        result, video_pred = run_inference(policy, obs, rank)
        dt = time.time() - t0
        times.append(dt)

        if use_torch_profiler:
            prof.__exit__(None, None, None)
            prof.export_chrome_trace(trace_path)
            if rank == 0:
                logger.info(f"Chrome trace saved to {trace_path}")
                logger.info("Top 30 CUDA kernels by total time:")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

        if rank == 0:
            logger.info(f"  Chunk {i}: frames {chunks[i]}, time {dt:.2f}s")

    # ── Flush profiler ──
    if stage_profiler is not None:
        stage_profiler.flush()

    # ── Summary ──
    if rank == 0 and len(times) > 0:
        steady = times[args.warmup_chunks:] if len(times) > args.warmup_chunks else times
        avg = sum(steady) / len(steady) if steady else 0

        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append(f"Profile Level: {args.profile_level}")
        summary_lines.append(f"GPUs: {world_size}")
        summary_lines.append(f"DiT cache: {args.enable_dit_cache}")
        summary_lines.append(f"Total chunks: {len(times)}  (warmup: {args.warmup_chunks})")
        summary_lines.append(f"All times: {[f'{t:.2f}' for t in times]}")
        summary_lines.append(f"Steady-state avg (skip first {args.warmup_chunks}): {avg:.2f}s per chunk")
        summary_lines.append(f"Each chunk = {ACTION_HORIZON} action steps @ 15Hz = {ACTION_HORIZON/15:.1f}s of robot time")
        if avg > 0:
            action_fps = ACTION_HORIZON / avg
            summary_lines.append(f"Action throughput: {action_fps:.1f} action steps/sec")
            summary_lines.append(f"Real-time ratio: {(ACTION_HORIZON/15)/avg:.2f}x (>1 = real-time)")

        if stage_profiler is not None:
            summary_lines.append("")
            summary_lines.append("── DeviceTimer Stage Breakdown ──")
            summary_lines.append(stage_profiler.summarize())

        summary_lines.append("=" * 70)

        summary_text = "\n".join(summary_lines)
        print(summary_text)

        # Save to file
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "profile_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text + "\n")
        logger.info(f"Summary saved to {summary_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main(tyro.cli(Args))
