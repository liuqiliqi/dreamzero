#!/usr/bin/env python3
"""Multi-GPU distributed test for tensor parallelism.

Usage:
    # 2 GPU (tp=2, no CFG):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 test_tp_distributed.py

    # 4 GPU (ip=2, tp=2):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 test_tp_distributed.py
"""
import os
import sys
import copy
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, os.path.dirname(__file__))

from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanSelfAttention,
    CausalWanAttentionBlock,
    _shard_linear_column,
    _shard_linear_row,
)
from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
    WanT2VCrossAttention,
    WanI2VCrossAttention,
    rope_params,
)
from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule


def log(msg, rank=None):
    if rank is None or rank == 0:
        print(msg, flush=True)


def test_ffn_tp_distributed(tp_mesh, device):
    """Test FFN TP with real all-reduce."""
    rank = tp_mesh.get_local_rank()
    tp_size = tp_mesh.size()
    tp_group = tp_mesh.get_group()

    torch.manual_seed(42)
    dim, ffn_dim = 64, 256
    B, S = 2, 16

    # Create identical FFN on all ranks
    ffn_ref = nn.Sequential(
        nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
        nn.Linear(ffn_dim, dim)
    ).to(device).eval()
    x = torch.randn(B, S, dim, device=device)

    # Reference output (same on all ranks since same seed)
    with torch.no_grad():
        ref_out = ffn_ref(x)

    # Shard FFN
    ffn_tp = copy.deepcopy(ffn_ref)
    _shard_linear_column(ffn_tp[0], rank, tp_size)
    _shard_linear_row(ffn_tp[2], rank, tp_size)

    with torch.no_grad():
        tp_out = ffn_tp(x)
    dist.all_reduce(tp_out, group=tp_group)

    diff = (ref_out - tp_out).abs().max().item()
    log(f"  FFN TP={tp_size}: max diff = {diff:.2e} ✓" if diff < 1e-5 else f"  FFN TP={tp_size}: FAIL diff={diff}", rank)
    assert diff < 1e-5, f"Too large: {diff}"


def test_cross_attn_tp_distributed(tp_mesh, device):
    """Test WanT2VCrossAttention with real distributed TP."""
    rank = tp_mesh.get_local_rank()
    tp_size = tp_mesh.size()

    torch.manual_seed(42)
    dim, num_heads = 64, 8
    B, L1, L2 = 2, 16, 32

    ref = WanT2VCrossAttention(dim, num_heads, qk_norm=True).to(device).eval()
    x = torch.randn(B, L1, dim, device=device)
    context = torch.randn(B, L2, dim, device=device)

    with torch.no_grad():
        ref_out = ref(x, context, context_lens=None)

    # Apply TP
    model = copy.deepcopy(ref)
    model.parallelize_tp(tp_mesh)

    with torch.no_grad():
        tp_out = model(x, context, context_lens=None)

    diff = (ref_out - tp_out).abs().max().item()
    log(f"  WanT2VCrossAttention TP={tp_size}: max diff = {diff:.2e} ✓" if diff < 1e-4 else f"  WanT2VCrossAttention TP={tp_size}: FAIL diff={diff}", rank)
    assert diff < 1e-4, f"Too large: {diff}"


def test_i2v_cross_attn_tp_distributed(tp_mesh, device):
    """Test WanI2VCrossAttention with real distributed TP."""
    rank = tp_mesh.get_local_rank()
    tp_size = tp_mesh.size()

    torch.manual_seed(42)
    dim, num_heads = 64, 8
    B, L1 = 2, 16
    L2 = 257 + 32

    ref = WanI2VCrossAttention(dim, num_heads, qk_norm=True).to(device).eval()
    x = torch.randn(B, L1, dim, device=device)
    context = torch.randn(B, L2, dim, device=device)

    with torch.no_grad():
        ref_out = ref(x, context)

    model = copy.deepcopy(ref)
    model.parallelize_tp(tp_mesh)

    with torch.no_grad():
        tp_out = model(x, context)

    diff = (ref_out - tp_out).abs().max().item()
    log(f"  WanI2VCrossAttention TP={tp_size}: max diff = {diff:.2e} ✓" if diff < 1e-4 else f"  WanI2VCrossAttention TP={tp_size}: FAIL diff={diff}", rank)
    assert diff < 1e-4, f"Too large: {diff}"


def test_self_attn_tp_distributed(tp_mesh, device):
    """Test CausalWanSelfAttention with real distributed TP."""
    rank = tp_mesh.get_local_rank()
    tp_size = tp_mesh.size()

    torch.manual_seed(42)
    dim, num_heads = 64, 8
    head_dim = dim // num_heads
    frame_seqlen = 8
    num_frames = 4
    S = num_frames * frame_seqlen
    B = 2

    ref = CausalWanSelfAttention(
        dim=dim, num_heads=num_heads, frame_seqlen=frame_seqlen,
        qk_norm=True, num_action_per_block=4, num_state_per_block=1
    ).to(device).eval()

    x = torch.randn(B, S, dim, device=device)
    freqs = rope_params(64, head_dim).to(device)[:S].view(S, 1, -1)
    freqs_action = rope_params(64, head_dim).to(device)
    freqs_state = rope_params(64, head_dim).to(device)

    with torch.no_grad():
        ref_out, _ = ref(x, freqs=freqs, freqs_action=freqs_action,
                         freqs_state=freqs_state, action_register_length=None,
                         is_tf=False)

    model = copy.deepcopy(ref)
    model.parallelize_tp(tp_mesh)

    with torch.no_grad():
        tp_out, _ = model(x, freqs=freqs, freqs_action=freqs_action,
                          freqs_state=freqs_state, action_register_length=None,
                          is_tf=False)

    diff = (ref_out - tp_out).abs().max().item()
    log(f"  CausalWanSelfAttention TP={tp_size}: max diff = {diff:.2e} ✓" if diff < 1e-4 else f"  CausalWanSelfAttention TP={tp_size}: FAIL diff={diff}", rank)
    assert diff < 1e-4, f"Too large: {diff}"


def test_attention_block_tp_distributed(tp_mesh, device):
    """Test CausalWanAttentionBlock with real distributed TP (self-attn + cross-attn + FFN)."""
    rank = tp_mesh.get_local_rank()
    tp_size = tp_mesh.size()

    torch.manual_seed(42)
    dim, ffn_dim, num_heads = 64, 256, 8
    head_dim = dim // num_heads
    frame_seqlen = 8
    num_frames = 4
    S = num_frames * frame_seqlen
    B = 2

    ref = CausalWanAttentionBlock(
        cross_attn_type='i2v_cross_attn',
        dim=dim, ffn_dim=ffn_dim, num_heads=num_heads,
        frame_seqlen=frame_seqlen, qk_norm=True, cross_attn_norm=True,
        num_action_per_block=4, num_state_per_block=1
    ).to(device).eval()

    x = torch.randn(B, S, dim, device=device)
    e = torch.randn(B, 1, 6, dim, device=device)
    freqs = rope_params(64, head_dim).to(device)[:S].view(S, 1, -1)
    freqs_action = rope_params(64, head_dim).to(device)
    freqs_state = rope_params(64, head_dim).to(device)
    context = torch.randn(B, 257 + 32, dim, device=device)

    with torch.no_grad():
        ref_out, _ = ref(x, e=e, freqs=freqs, freqs_action=freqs_action,
                         freqs_state=freqs_state, action_register_length=None,
                         context=context, is_tf=False)

    model = copy.deepcopy(ref)
    model.parallelize_tp(tp_mesh)

    with torch.no_grad():
        tp_out, _ = model(x, e=e, freqs=freqs, freqs_action=freqs_action,
                          freqs_state=freqs_state, action_register_length=None,
                          context=context, is_tf=False)

    diff = (ref_out - tp_out).abs().max().item()
    log(f"  CausalWanAttentionBlock TP={tp_size}: max diff = {diff:.2e} ✓" if diff < 1e-3 else f"  CausalWanAttentionBlock TP={tp_size}: FAIL diff={diff}", rank)
    assert diff < 1e-3, f"Too large: {diff}"


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # Create TP-only mesh (all GPUs in one TP group)
    tp_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("tp",))["tp"]

    log(f"\n{'='*60}", rank)
    log(f"Distributed TP Test: {world_size} GPUs", rank)
    log(f"{'='*60}", rank)

    dist.barrier()
    test_ffn_tp_distributed(tp_mesh, device)
    dist.barrier()
    test_cross_attn_tp_distributed(tp_mesh, device)
    dist.barrier()
    test_i2v_cross_attn_tp_distributed(tp_mesh, device)
    dist.barrier()
    test_self_attn_tp_distributed(tp_mesh, device)
    dist.barrier()
    test_attention_block_tp_distributed(tp_mesh, device)
    dist.barrier()

    log(f"\n{'='*60}", rank)
    log("ALL DISTRIBUTED TESTS PASSED", rank)
    log(f"{'='*60}", rank)

    dist.destroy_process_group()
