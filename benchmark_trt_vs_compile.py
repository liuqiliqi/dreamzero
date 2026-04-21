#!/usr/bin/env python3
"""Benchmark: PyTorch (no compile) vs torch.compile vs TensorRT on 2 GPUs.

Runs the full inference pipeline (KV prefill + diffusion loop) across multiple
chunks and reports steady-state latency for each mode.

Usage (2 GPUs, ip=2 no TP):
    # Mode 1: PyTorch baseline (no compile, no TRT)
    ENABLE_TENSORRT=true CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
        --standalone --nproc_per_node=2 benchmark_trt_vs_compile.py \
        --mode no-compile

    # Mode 2: torch.compile (reduce-overhead)
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
        --standalone --nproc_per_node=2 benchmark_trt_vs_compile.py \
        --mode compile

    # Mode 3: TensorRT engine
    ENABLE_TENSORRT=true LOAD_TRT_ENGINE=dit_droid.engine CUDA_VISIBLE_DEVICES=0,1 \
        python -m torch.distributed.run \
        --standalone --nproc_per_node=2 benchmark_trt_vs_compile.py \
        --mode trt
"""

import dataclasses
import datetime
import logging
import os
import pickle
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import tyro
from tianshou.data import Batch
from torch.distributed.device_mesh import init_device_mesh

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

logger = logging.getLogger(__name__)

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
    mode: str = "no-compile"  # no-compile | compile | trt
    num_chunks: int = 15
    enable_dit_cache: bool = False


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
    batch = Batch(obs=obs)
    dist.barrier()
    with torch.no_grad():
        result_batch, video_pred = policy.lazy_joint_forward_causal(batch)
    dist.barrier()
    return result_batch, video_pred


def main(args: Args):
    # ── Configure mode ──
    if args.mode == "trt":
        os.environ["ENABLE_TENSORRT"] = "true"
        # LOAD_TRT_ENGINE should be set externally
        if "LOAD_TRT_ENGINE" not in os.environ:
            os.environ["LOAD_TRT_ENGINE"] = "dit_droid.engine"
    elif args.mode == "no-compile":
        # Disable compile by setting ENABLE_TENSORRT=true (skips torch.compile)
        # but don't set LOAD_TRT_ENGINE (so no TRT engine is loaded)
        os.environ["ENABLE_TENSORRT"] = "true"
        os.environ.pop("LOAD_TRT_ENGINE", None)
    elif args.mode == "compile":
        # Normal mode: torch.compile enabled, no TRT
        os.environ.pop("ENABLE_TENSORRT", None)
        os.environ.pop("LOAD_TRT_ENGINE", None)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # ── Init distributed ──
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # 2 GPU: ip=2 (no TP)
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("ip",))
    logger.info(f"Rank {rank}/{world_size}, mode={args.mode}, mesh={mesh}")

    timeout_delta = datetime.timedelta(seconds=50000)
    dist.new_group(backend="gloo", timeout=timeout_delta)

    # ── Load model ──
    logger.info(f"Rank {rank}: loading model (mode={args.mode})...")
    t_load_start = time.time()
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=args.model_path,
        device="cuda",
        device_mesh=mesh,
    )
    t_load = time.time() - t_load_start
    logger.info(f"Rank {rank}: model loaded in {t_load:.1f}s")

    # ── Load video (rank 0 only) ──
    if rank == 0:
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

    # ── Warmup: initial frame ──
    if rank == 0:
        logger.info(f"=== Warmup: initial frame [0] (mode={args.mode}) ===")
        obs = build_obs(camera_frames, [0], PROMPT)
    else:
        obs = None

    obs = broadcast_obs(obs, rank)
    t0 = time.time()
    run_inference(policy, obs, rank)
    dt = time.time() - t0
    if rank == 0:
        logger.info(f"  Warmup: {dt:.2f}s")

    # ── Benchmark chunks ──
    times = []
    num_chunks = len(chunks) if rank == 0 else 0
    nc_tensor = torch.tensor([num_chunks if rank == 0 else 0], dtype=torch.int32, device="cuda")
    dist.broadcast(nc_tensor, src=0)
    num_chunks = nc_tensor.item()

    for i in range(num_chunks):
        if rank == 0:
            obs = build_obs(camera_frames, chunks[i], PROMPT)
        else:
            obs = None

        obs = broadcast_obs(obs, rank)
        t0 = time.time()
        run_inference(policy, obs, rank)
        dt = time.time() - t0
        times.append(dt)

        if rank == 0:
            logger.info(f"  Chunk {i}: {dt:.2f}s")

    # ── Summary ──
    if rank == 0 and len(times) > 0:
        skip = min(2, len(times) - 1)
        steady = times[skip:]
        avg = sum(steady) / len(steady)
        best = min(steady)
        worst = max(steady)
        action_fps = ACTION_HORIZON / avg
        rt_ratio = (ACTION_HORIZON / 15) / avg

        logger.info("=" * 70)
        logger.info(f"MODE: {args.mode}")
        logger.info(f"GPUs: {world_size}x H20 (ip={world_size}, no TP)")
        logger.info(f"Chunks: {len(times)} total, {len(steady)} steady-state (skip first {skip})")
        logger.info(f"All times: {[f'{t:.2f}' for t in times]}")
        logger.info(f"Steady-state:")
        logger.info(f"  Average: {avg:.3f}s per chunk")
        logger.info(f"  Best:    {best:.3f}s")
        logger.info(f"  Worst:   {worst:.3f}s")
        logger.info(f"  StdDev:  {np.std(steady):.3f}s")
        logger.info(f"Robot time per chunk: {ACTION_HORIZON/15:.1f}s (24 actions @ 15Hz)")
        logger.info(f"Action throughput: {action_fps:.1f} steps/sec")
        logger.info(f"Real-time ratio: {rt_ratio:.2f}x (>1 = real-time)")
        logger.info("=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main(tyro.cli(Args))
