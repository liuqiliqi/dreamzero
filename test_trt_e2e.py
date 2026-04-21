#!/usr/bin/env python3
"""End-to-end verification: TRT engine vs PyTorch model.

Loads both the TRT engine and PyTorch DiT model, runs them with identical
inputs, and compares the outputs.

Usage:
    ENABLE_TENSORRT=true CUDA_VISIBLE_DEVICES=0,1 python test_trt_e2e.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WEIGHTS_PATH = "/fact_data/qiliu/dreamzero_weights/DreamZero-DROID"
ENGINE_PATH = "dit_droid.engine"

# DROID shapes
KV_LEN = 2640  # 6 frames cached (3 blocks of 2 frames), per_frame_tokens=880


def create_inputs(device: torch.device, dtype=torch.bfloat16, seed=42):
    """Create deterministic dummy inputs."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=dtype).to(device)

    def randint(low, high, shape):
        return torch.randint(low, high, shape, generator=gen).to(device)

    return dict(
        x=randn(1, 16, 2, 44, 80),
        timestep=randint(100, 200, (1, 2)),
        context=randn(1, 512, 4096),
        kv_cache=randn(40, 2, 1, KV_LEN, 40, 128),
        y=randn(1, 20, 2, 44, 80),
        clip_feature=randn(1, 257, 1280),
        action=randn(1, 24, 32),
        timestep_action=randint(100, 200, (1, 24)),
        state=randn(1, 1, 64),
    )


def run_pytorch(inputs: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run PyTorch model and return (video_pred, action_pred)."""
    from safetensors.torch import load_file
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

    config_path = os.path.join(WEIGHTS_PATH, "config.json")
    with open(config_path) as f:
        vla_config = json.load(f)
    dit_cfg = vla_config["action_head_cfg"]["config"]["diffusion_model_cfg"]
    dit_kwargs = {k: v for k, v in dit_cfg.items() if not k.startswith("_")}

    logger.info("Instantiating CausalWanModel...")
    model = CausalWanModel(**dit_kwargs)

    # Load weights
    index_path = os.path.join(WEIGHTS_PATH, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    dit_prefix = "action_head.model."
    shard_files = set()
    for key, shard_file in index["weight_map"].items():
        if key.startswith(dit_prefix):
            shard_files.add(shard_file)

    state_dict = {}
    for shard_file in sorted(shard_files):
        shard_path = os.path.join(WEIGHTS_PATH, shard_file)
        logger.info("Loading shard: %s", shard_path)
        shard_data = load_file(shard_path)
        for key, value in shard_data.items():
            if key.startswith(dit_prefix):
                state_dict[key[len(dit_prefix):]] = value

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    logger.info("PyTorch model loaded (%d params)", sum(p.numel() for p in model.parameters()))

    # Move inputs to device
    pt_inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute seq_len from input shape
    F_dim = pt_inputs["x"].shape[2]
    H_lat, W_lat = pt_inputs["x"].shape[3], pt_inputs["x"].shape[4]
    per_frame_tokens = (H_lat // 2) * (W_lat // 2)
    seq_len = per_frame_tokens * F_dim
    kv_len = pt_inputs["kv_cache"].shape[3]
    current_start_frame = kv_len // per_frame_tokens

    # Unpack kv_cache
    kv_cache_list = [pt_inputs["kv_cache"][i] for i in range(40)]

    with torch.no_grad():
        # Warm up
        logger.info("PyTorch warm-up run...")
        _ = model._forward_inference(
            x=pt_inputs["x"],
            timestep=pt_inputs["timestep"],
            context=pt_inputs["context"],
            seq_len=seq_len,
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=pt_inputs["y"],
            clip_feature=pt_inputs["clip_feature"],
            action=pt_inputs["action"],
            timestep_action=pt_inputs["timestep_action"],
            state=pt_inputs["state"],
            current_start_frame=current_start_frame,
        )

        # Timed run
        torch.cuda.synchronize(device)
        t0 = time.time()
        video_pred, action_pred, _ = model._forward_inference(
            x=pt_inputs["x"],
            timestep=pt_inputs["timestep"],
            context=pt_inputs["context"],
            seq_len=seq_len,
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=pt_inputs["y"],
            clip_feature=pt_inputs["clip_feature"],
            action=pt_inputs["action"],
            timestep_action=pt_inputs["timestep_action"],
            state=pt_inputs["state"],
            current_start_frame=current_start_frame,
        )
        torch.cuda.synchronize(device)
        elapsed = time.time() - t0
        logger.info("PyTorch inference: %.3f s", elapsed)

    # Move results to CPU to free GPU
    video_pred_cpu = video_pred.cpu()
    action_pred_cpu = action_pred.cpu() if action_pred is not None else None

    # Free model
    del model, state_dict
    torch.cuda.empty_cache()

    return video_pred_cpu, action_pred_cpu


def run_trt(inputs: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TRT engine and return (video_pred, action_pred)."""
    from groot.vla.model.dreamzero.tensorrt_utils import load_tensorrt_engine

    device_id = device.index if device.index is not None else 0
    logger.info("Loading TRT engine from %s on device %d...", ENGINE_PATH, device_id)
    engine = load_tensorrt_engine(ENGINE_PATH, device=device_id)

    # Move inputs to device
    trt_inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warm up
    logger.info("TRT warm-up run...")
    _ = engine(
        trt_inputs["x"],
        trt_inputs["timestep"],
        context=trt_inputs["context"],
        kv_cache=trt_inputs["kv_cache"],
        y=trt_inputs["y"],
        clip_feature=trt_inputs["clip_feature"],
        action=trt_inputs["action"],
        timestep_action=trt_inputs["timestep_action"],
        state=trt_inputs["state"],
    )

    # Timed runs
    torch.cuda.synchronize(device)
    times = []
    for i in range(5):
        t0 = time.time()
        video_pred, action_pred = engine(
            trt_inputs["x"],
            trt_inputs["timestep"],
            context=trt_inputs["context"],
            kv_cache=trt_inputs["kv_cache"],
            y=trt_inputs["y"],
            clip_feature=trt_inputs["clip_feature"],
            action=trt_inputs["action"],
            timestep_action=trt_inputs["timestep_action"],
            state=trt_inputs["state"],
        )
        torch.cuda.synchronize(device)
        elapsed = time.time() - t0
        times.append(elapsed)
        logger.info("TRT inference run %d: %.3f s", i + 1, elapsed)

    logger.info("TRT average: %.3f s (min=%.3f, max=%.3f)",
                np.mean(times), min(times), max(times))

    video_pred_cpu = video_pred.cpu()
    action_pred_cpu = action_pred.cpu()

    del engine
    torch.cuda.empty_cache()

    return video_pred_cpu, action_pred_cpu


def compare(
    pt_video: torch.Tensor, pt_action: torch.Tensor | None,
    trt_video: torch.Tensor, trt_action: torch.Tensor,
) -> bool:
    """Compare PyTorch and TRT outputs."""
    # Cast to float32 for comparison
    pt_video_f = pt_video.float()
    trt_video_f = trt_video.float()

    video_abs_diff = (pt_video_f - trt_video_f).abs()
    video_max_diff = video_abs_diff.max().item()
    video_mean_diff = video_abs_diff.mean().item()
    video_rel_diff = (video_abs_diff / (pt_video_f.abs() + 1e-6)).mean().item()

    logger.info("Video noise pred comparison:")
    logger.info("  Shape: PT=%s, TRT=%s", pt_video.shape, trt_video.shape)
    logger.info("  Max abs diff: %.6f", video_max_diff)
    logger.info("  Mean abs diff: %.6f", video_mean_diff)
    logger.info("  Mean rel diff: %.6f", video_rel_diff)

    if pt_action is not None:
        pt_action_f = pt_action.float()
        trt_action_f = trt_action.float()
        action_abs_diff = (pt_action_f - trt_action_f).abs()
        action_max_diff = action_abs_diff.max().item()
        action_mean_diff = action_abs_diff.mean().item()

        logger.info("Action noise pred comparison:")
        logger.info("  Shape: PT=%s, TRT=%s", pt_action.shape, trt_action.shape)
        logger.info("  Max abs diff: %.6f", action_max_diff)
        logger.info("  Mean abs diff: %.6f", action_mean_diff)
    else:
        action_max_diff = 0
        logger.info("Action noise pred: PyTorch returned None, skipping comparison")

    # Pass/fail
    threshold = 0.01
    passed = video_max_diff < threshold and action_max_diff < threshold
    if passed:
        logger.info("PASSED: All diffs < %.4f", threshold)
    else:
        logger.warning("FAILED: Max diff exceeds %.4f (video=%.6f, action=%.6f)",
                       threshold, video_max_diff, action_max_diff)
    return passed


def main():
    if not os.path.exists(ENGINE_PATH):
        logger.error("TRT engine not found at %s", ENGINE_PATH)
        sys.exit(1)

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info("Available GPUs: %d", num_gpus)

    # TRT engine is device-specific (built on GPU 0), must run on same GPU.
    # Run PyTorch first, free memory, then run TRT on the same device.
    pt_device = torch.device("cuda:0")
    trt_device = torch.device("cuda:0")
    logger.info("Using GPU 0 for both (sequential: PyTorch → free → TRT)")

    # Create inputs on CPU (deterministic)
    logger.info("Creating deterministic inputs (kv_len=%d)...", KV_LEN)
    inputs = create_inputs(device=torch.device("cpu"), seed=42)
    for name, t in inputs.items():
        logger.info("  %s: shape=%s, dtype=%s", name, t.shape, t.dtype)

    # Run PyTorch
    logger.info("=" * 60)
    logger.info("Running PyTorch model...")
    pt_video, pt_action = run_pytorch(inputs, pt_device)
    logger.info("PyTorch video pred: shape=%s, range=[%.4f, %.4f]",
                pt_video.shape, pt_video.float().min(), pt_video.float().max())
    if pt_action is not None:
        logger.info("PyTorch action pred: shape=%s, range=[%.4f, %.4f]",
                    pt_action.shape, pt_action.float().min(), pt_action.float().max())

    # Run TRT
    logger.info("=" * 60)
    logger.info("Running TRT engine...")
    trt_video, trt_action = run_trt(inputs, trt_device)
    logger.info("TRT video pred: shape=%s, range=[%.4f, %.4f]",
                trt_video.shape, trt_video.float().min(), trt_video.float().max())
    logger.info("TRT action pred: shape=%s, range=[%.4f, %.4f]",
                trt_action.shape, trt_action.float().min(), trt_action.float().max())

    # Compare
    logger.info("=" * 60)
    logger.info("Comparing outputs...")
    passed = compare(pt_video, pt_action, trt_video, trt_action)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
