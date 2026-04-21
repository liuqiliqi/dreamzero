#!/usr/bin/env python3
"""End-to-end TP test with the real model, but with debug prints.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 test_tp_e2e.py
"""
import os
import sys
import time
import logging

os.environ["ATTENTION_BACKEND"] = "TE"

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, os.path.dirname(__file__))

from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel
from groot.vla.model.dreamzero.modules.wan2_1_submodule import rope_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [R%(process)d] %(message)s")
logger = logging.getLogger(__name__)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    tp_size = world_size
    tp_mesh = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))["tp"]

    logger.info(f"Rank {rank}/{world_size}, tp_size={tp_size}")

    # Create a small CausalWanModel matching the real architecture but with fewer layers
    torch.manual_seed(42)
    logger.info(f"[{rank}] Creating model...")
    model = CausalWanModel(
        model_type='i2v',
        patch_size=(1, 2, 2),
        frame_seqlen=220,  # smaller for test
        text_len=64,
        in_dim=36,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=2,  # only 2 layers for speed
        max_chunk_size=-1,
        qk_norm=True,
        cross_attn_norm=True,
        num_frame_per_block=1,
        num_action_per_block=24,
        num_state_per_block=1,
        diffusion_model_pretrained_path=None,
    ).to(device=device, dtype=torch.bfloat16).eval()

    logger.info(f"[{rank}] Model created. Applying TP...")
    dist.barrier()
    model.parallelize_tp(tp_mesh)
    logger.info(f"[{rank}] TP applied. Block0 self_attn num_heads={model.blocks[0].self_attn.num_heads}")
    logger.info(f"[{rank}] Block0 ffn[0] out_features={model.blocks[0].ffn[0].out_features}")
    logger.info(f"[{rank}] Block0 ffn[2] in_features={model.blocks[0].ffn[2].in_features}")

    # Test forward pass through _forward_blocks
    B = 1
    frame_seqlen = 220
    num_frames = 3
    seq_len = num_frames * frame_seqlen
    head_dim = model.blocks[0].self_attn.head_dim
    num_heads_local = model.blocks[0].self_attn.num_heads

    # Create dummy inputs
    x_video = torch.randn(B, seq_len, 5120, device=device, dtype=torch.bfloat16)
    e0 = torch.randn(B, 1, 6, 5120, device=device, dtype=torch.bfloat16)
    freqs = rope_params(1024, head_dim).to(device)[:seq_len].view(seq_len, 1, -1)
    freqs_action = rope_params(1024, head_dim).to(device)
    freqs_state = rope_params(1024, head_dim).to(device)
    context = torch.randn(B, 257 + 64, 5120, device=device, dtype=torch.bfloat16)

    # Create KV caches with correct local head count
    kv_cache_list = []
    for _ in range(model.num_layers):
        kv_cache_list.append(
            torch.zeros([2, B, 0, num_heads_local, head_dim], dtype=torch.bfloat16, device=device)
        )

    logger.info(f"[{rank}] Running forward block 0...")
    dist.barrier()

    with torch.no_grad():
        block = model.blocks[0]
        y, kv = block(
            x=x_video,
            e=e0,
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=None,
            context=context,
            kv_cache=kv_cache_list[0],
            is_tf=False,
        )

    logger.info(f"[{rank}] Block 0 output shape: {y.shape}, kv_cache: {kv.shape if kv is not None else None}")

    # Now test with the returned KV cache (simulates second diffusion step)
    logger.info(f"[{rank}] Running forward block 0 with KV cache (second step)...")
    dist.barrier()

    x_new = torch.randn(B, frame_seqlen, 5120, device=device, dtype=torch.bfloat16)
    freqs_new = rope_params(1024, head_dim).to(device)[:frame_seqlen].view(frame_seqlen, 1, -1)

    with torch.no_grad():
        y2, kv2 = block(
            x=x_new,
            e=e0,
            freqs=freqs_new,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=None,
            context=context,
            kv_cache=kv,
            is_tf=False,
            current_start_frame=num_frames,
        )

    logger.info(f"[{rank}] Block 0 step2 output: {y2.shape}, kv2: {kv2.shape if kv2 is not None else None}")

    dist.barrier()
    logger.info(f"[{rank}] ALL PASSED!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
