#!/usr/bin/env python3
"""Local benchmark: load model on N GPUs, run inference on local video, no network.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
        --standalone --nproc_per_node=8 local_benchmark.py \
        --model-path /fact_data/qiliu/dreamzero_weights/DreamZero-DROID \
        --enable-dit-cache
"""

import dataclasses
import datetime
import logging
import os
import pickle
import time

import cv2
import imageio
import numpy as np
import torch
import torch.distributed as dist
import tyro
from einops import rearrange
from tianshou.data import Batch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from groot.vla.common.utils.misc.video_utils import _get_video_info_ffmpeg
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "debug_image")
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
    video_dir: str = DEFAULT_VIDEO_DIR
    enable_dit_cache: bool = False
    num_dit_steps: int = 0  # 0 = no caching. E.g., 5 for 5 DiT steps out of 16
    num_inference_steps: int = 0  # 0 = use model default (16)
    no_extrapolate: bool = False  # disable linear extrapolation for cached steps
    num_chunks: int | None = None  # None = run until the source video ends
    tome_ratio: float = 0.0  # Token Merging ratio (0.0 = disabled, 0.5 = merge 50%)
    no_compile: bool = False  # Disable torch.compile for fair comparison
    export_full_reconstruction: bool = True
    match_source_length: bool = True


def init_mesh() -> DeviceMesh:
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
    """Build observation dict in AR_droid format."""
    obs = {}
    for cam_key, all_frames in camera_frames.items():
        obs[cam_key] = all_frames[frame_indices]  # (T, H, W, 3)

    obs["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)
    obs["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)
    obs["annotation.language.action_text"] = prompt
    return obs


def build_chunks(total_frames: int, num_chunks: int | None) -> list[list[int]]:
    chunks = []
    current = 23
    while True:
        indices = [max(current + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        if num_chunks is not None and len(chunks) >= num_chunks:
            break
        current += ACTION_HORIZON
    return chunks


def resample_frames(frames: np.ndarray, target_num_frames: int) -> np.ndarray:
    if target_num_frames <= 0 or len(frames) == 0 or len(frames) == target_num_frames:
        return frames
    sample_positions = np.linspace(0, len(frames) - 1, target_num_frames)
    sample_indices = np.round(sample_positions).astype(int)
    return frames[sample_indices]


def broadcast_obs(obs: dict | None, rank: int):
    """Rank 0 broadcasts obs to all workers. Workers receive it."""
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
    """All ranks participate in distributed inference."""
    batch = Batch(obs=obs)
    dist.barrier()
    with torch.no_grad():
        result_batch, video_pred = policy.lazy_joint_forward_causal(batch)
    dist.barrier()
    return result_batch, video_pred


def main(args: Args):
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    if args.num_dit_steps > 0:
        os.environ["NUM_DIT_STEPS"] = str(args.num_dit_steps)
    if args.num_inference_steps > 0:
        os.environ["NUM_INFERENCE_STEPS"] = str(args.num_inference_steps)
    if args.no_extrapolate:
        os.environ["DIT_CACHE_EXTRAPOLATE"] = "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    if args.tome_ratio > 0:
        os.environ["TOME_RATIO"] = str(args.tome_ratio)
    if args.no_compile:
        os.environ["DISABLE_COMPILE"] = "true"
    torch._dynamo.config.recompile_limit = 800

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device_mesh = init_mesh()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    timeout_delta = datetime.timedelta(seconds=50000)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)

    logger.info(f"Rank {rank}: loading model...")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=args.model_path,
        device="cuda",
        device_mesh=device_mesh,
    )
    logger.info(f"Rank {rank}: model loaded")

    # Only rank 0 loads video data
    if rank == 0:
        logger.info(f"Loading video frames from {args.video_dir}...")
        camera_frames = {}
        source_video_path = None
        for cam_key, fname in CAMERA_FILES.items():
            path = os.path.join(args.video_dir, fname)
            if source_video_path is None:
                source_video_path = path
            camera_frames[cam_key] = load_video(path)
            logger.info(f"  {cam_key}: {camera_frames[cam_key].shape} ({path})")

        total_frames = min(v.shape[0] for v in camera_frames.values())
        source_video_info = _get_video_info_ffmpeg(source_video_path)
        source_fps = source_video_info["fps"]
        source_duration = total_frames / source_fps if source_fps > 0 else 0.0
        requested_num_chunks = None if args.export_full_reconstruction else args.num_chunks
        chunks = build_chunks(total_frames, requested_num_chunks)

        logger.info(
            f"Will run: 1 initial + {len(chunks)} chunks | "
            f"source fps={source_fps:.3f}, total_frames={total_frames}, duration={source_duration:.2f}s"
        )
    else:
        source_fps = None
        total_frames = None
        source_duration = None
        chunks = None
        source_video_info = None

    # ── Warmup: initial frame ──
    if rank == 0:
        logger.info("=== Warmup: initial frame [0] ===")
        obs = build_obs(camera_frames, [0], PROMPT)
    else:
        obs = None

    obs = broadcast_obs(obs, rank)
    t0 = time.time()
    result, warmup_video_pred = run_inference(policy, obs, rank)
    dt = time.time() - t0

    if rank == 0:
        action_dict = {}
        for k in dir(result.act):
            if k.startswith("action."):
                action_dict[k] = getattr(result.act, k)
        # Find action shape
        for k, v in action_dict.items():
            if hasattr(v, 'shape'):
                logger.info(f"  Warmup done: {dt:.1f}s, action key={k} shape={v.shape}")

    # ── Benchmark: subsequent chunks ──
    times = []
    video_preds = [warmup_video_pred] if rank == 0 else []

    num_chunks = len(chunks) if rank == 0 else 0
    # Broadcast num_chunks so workers know how many iterations
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
        result, video_pred = run_inference(policy, obs, rank)
        dt = time.time() - t0
        times.append(dt)

        if rank == 0:
            video_preds.append(video_pred)
            logger.info(f"  Chunk {i}: frames {chunks[i]}, time {dt:.2f}s")

    # ── Summary ──
    if rank == 0 and len(times) > 0:
        # Skip first 2 chunks (still warming up JIT)
        steady = times[2:] if len(times) > 2 else times
        avg = sum(steady) / len(steady)
        action_fps = ACTION_HORIZON / avg

        logger.info("=" * 60)
        logger.info(f"GPUs: {world_size}x H20")
        logger.info(f"DiT cache: {args.enable_dit_cache}")
        logger.info(f"ToMe ratio: {args.tome_ratio}")
        logger.info(f"Total chunks: {len(times)}")
        logger.info(f"All times: {[f'{t:.2f}' for t in times]}")
        logger.info(f"Steady-state avg (skip first 2): {avg:.2f}s per chunk")
        logger.info(f"Each chunk = {ACTION_HORIZON} action steps @ 15Hz = {ACTION_HORIZON/15:.1f}s of robot time")
        logger.info(f"Action throughput: {action_fps:.1f} action steps/sec")
        logger.info(f"Real-time ratio: {(ACTION_HORIZON/15)/avg:.2f}x (>1 = real-time)")
        logger.info("=" * 60)

        # Save video
        if video_preds:
            try:
                video_cat = torch.cat(video_preds, dim=2)
                frames = policy.trained_model.action_head.vae.decode(
                    video_cat,
                    tiled=policy.trained_model.action_head.tiled,
                    tile_size=(policy.trained_model.action_head.tile_size_height,
                               policy.trained_model.action_head.tile_size_width),
                    tile_stride=(policy.trained_model.action_head.tile_stride_height,
                                  policy.trained_model.action_head.tile_stride_width),
                )
                frames = rearrange(frames, "B C T H W -> B T H W C")
                frames = ((frames[0].float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                decoded_frame_count = len(frames)
                output_fps = source_fps
                if args.match_source_length:
                    frames = resample_frames(frames, total_frames)
                output_frame_count = len(frames)
                output_duration = output_frame_count / output_fps if output_fps > 0 else 0.0
                logger.info(
                    f"Decoded frames={decoded_frame_count}, output frames={output_frame_count}, "
                    f"source frames={total_frames}, source duration={source_duration:.2f}s, "
                    f"output duration={output_duration:.2f}s @ {output_fps:.3f}fps"
                )
                num_dit_steps = os.getenv("NUM_DIT_STEPS", "8")
                trt_tag = "_fp8trt" if os.getenv("LOAD_TRT_ENGINE") else ""
                recon_tag = "_fullrecon" if args.export_full_reconstruction else ""
                out_path = f"benchmark_{world_size}gpu_{num_dit_steps}steps{trt_tag}{recon_tag}_{int(round(output_fps))}fps.mp4"
                imageio.mimsave(out_path, list(frames), fps=output_fps, codec="libx264")
                logger.info(f"Saved predicted video to {out_path}")
            except Exception as e:
                logger.warning(f"Failed to save video: {e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main(tyro.cli(Args))
