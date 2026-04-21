#!/usr/bin/env python3
"""Test tensor parallelism correctness.

Single-process simulation: manually splits heads/weights and sums partial outputs
to verify they match the original (non-TP) output.
"""
import os
import sys
import copy
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanSelfAttention,
    _shard_linear_column,
    _shard_linear_row,
)
from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
    WanT2VCrossAttention,
    WanI2VCrossAttention,
    rope_params,
)
from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule
from groot.vla.model.dreamzero.modules.attention import flash_attention


def _shard_o_proj(model, num_heads, head_dim, dim, rank, tp_size):
    """Shard O-proj row-parallel and zero bias on non-zero ranks."""
    local_num_heads = num_heads // tp_size
    head_start = rank * local_num_heads
    head_end = head_start + local_num_heads
    w_o = model.o.weight.data
    w_o_heads = w_o.view(dim, num_heads, head_dim)
    w_o_local = w_o_heads[:, head_start:head_end, :].reshape(dim, local_num_heads * head_dim).contiguous()
    model.o.weight = nn.Parameter(w_o_local)
    model.o.in_features = local_num_heads * head_dim
    if rank != 0 and model.o.bias is not None:
        model.o.bias = nn.Parameter(torch.zeros_like(model.o.bias))


def test_ffn_tp(tp_size, device):
    """Test column-parallel + row-parallel FFN sums to original."""
    torch.manual_seed(42)
    dim, ffn_dim = 64, 256
    B, S = 2, 16

    ffn_ref = nn.Sequential(
        nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
        nn.Linear(ffn_dim, dim)
    ).to(device).eval()

    x = torch.randn(B, S, dim, device=device)
    with torch.no_grad():
        ref_out = ffn_ref(x)

    partial_outputs = []
    for rank in range(tp_size):
        ffn = copy.deepcopy(ffn_ref)
        _shard_linear_column(ffn[0], rank, tp_size)
        _shard_linear_row(ffn[2], rank, tp_size)
        with torch.no_grad():
            out = ffn(x)
        partial_outputs.append(out)

    tp_out = sum(partial_outputs)
    diff = (ref_out - tp_out).abs().max().item()
    print(f"  FFN TP={tp_size}: max diff = {diff:.2e}", end="")
    assert diff < 1e-5, f"Too large diff: {diff}"
    print(" ✓")


def test_cross_attention_tp(tp_size, device):
    """Test WanT2VCrossAttention with TP produces same output."""
    torch.manual_seed(42)
    dim, num_heads = 64, 8
    head_dim = dim // num_heads
    B, L1, L2 = 2, 16, 32

    ref = WanT2VCrossAttention(dim, num_heads, qk_norm=True).to(device).eval()
    x = torch.randn(B, L1, dim, device=device)
    context = torch.randn(B, L2, dim, device=device)

    with torch.no_grad():
        ref_out = ref(x, context, context_lens=None)

    # Manually simulate TP: compute Q/K/V with full weights, slice heads,
    # pass through sharded O-proj, sum
    partial_outputs = []
    for rank in range(tp_size):
        local_n = num_heads // tp_size
        hs = rank * local_n
        he = hs + local_n

        with torch.no_grad():
            q = ref.norm_q(ref.q(x)).view(B, -1, num_heads, head_dim)[:, :, hs:he, :].contiguous()
            k = ref.norm_k(ref.k(context)).view(B, -1, num_heads, head_dim)[:, :, hs:he, :].contiguous()
            v = ref.v(context).view(B, -1, num_heads, head_dim)[:, :, hs:he, :].contiguous()
            attn_out = flash_attention(q, k, v, k_lens=None)
            attn_flat = attn_out.flatten(2)  # (B, L, local_n * head_dim)

            # Sharded O-proj
            w_o = ref.o.weight.data  # (dim, dim)
            w_o_heads = w_o.view(dim, num_heads, head_dim)
            w_o_local = w_o_heads[:, hs:he, :].reshape(dim, local_n * head_dim)
            out = torch.nn.functional.linear(attn_flat, w_o_local)
            if rank == 0 and ref.o.bias is not None:
                out = out + ref.o.bias
        partial_outputs.append(out)

    tp_out = sum(partial_outputs)
    diff = (ref_out - tp_out).abs().max().item()
    print(f"  WanT2VCrossAttention TP={tp_size}: max diff = {diff:.2e}", end="")
    assert diff < 1e-4, f"Too large diff: {diff}"
    print(" ✓")


def test_i2v_cross_attention_tp(tp_size, device):
    """Test WanI2VCrossAttention with TP produces same output."""
    torch.manual_seed(42)
    dim, num_heads = 64, 8
    head_dim = dim // num_heads
    B, L1 = 2, 16
    L2 = 257 + 32  # 257 img + 32 text

    ref = WanI2VCrossAttention(dim, num_heads, qk_norm=True).to(device).eval()
    x = torch.randn(B, L1, dim, device=device)
    context = torch.randn(B, L2, dim, device=device)

    with torch.no_grad():
        ref_out = ref(x, context)

    context_img = context[:, :257]
    context_text = context[:, 257:]

    partial_outputs = []
    for rank in range(tp_size):
        local_n = num_heads // tp_size
        hs = rank * local_n
        he = hs + local_n

        with torch.no_grad():
            q = ref.norm_q(ref.q(x)).view(B, -1, num_heads, head_dim)[:, :, hs:he, :].contiguous()
            k = ref.norm_k(ref.k(context_text)).view(B, -1, num_heads, head_dim)[:, :, hs:he, :].contiguous()
            v = ref.v(context_text).view(B, -1, num_heads, head_dim)[:, :, hs:he, :].contiguous()
            text_out = flash_attention(q, k, v, k_lens=None)

            k_img = ref.norm_k_img(ref.k_img(context_img)).view(B, -1, num_heads, head_dim)[:, :, hs:he, :].contiguous()
            v_img = ref.v_img(context_img).view(B, -1, num_heads, head_dim)[:, :, hs:he, :].contiguous()
            img_out = flash_attention(q, k_img, v_img, k_lens=None)

            combined = text_out.flatten(2) + img_out.flatten(2)

            w_o = ref.o.weight.data.view(dim, num_heads, head_dim)
            w_o_local = w_o[:, hs:he, :].reshape(dim, local_n * head_dim)
            out = torch.nn.functional.linear(combined, w_o_local)
            if rank == 0 and ref.o.bias is not None:
                out = out + ref.o.bias
        partial_outputs.append(out)

    tp_out = sum(partial_outputs)
    diff = (ref_out - tp_out).abs().max().item()
    print(f"  WanI2VCrossAttention TP={tp_size}: max diff = {diff:.2e}", end="")
    assert diff < 1e-4, f"Too large diff: {diff}"
    print(" ✓")


def test_self_attention_tp(tp_size, device):
    """Test CausalWanSelfAttention with TP produces same output."""
    torch.manual_seed(42)
    dim, num_heads = 64, 8
    head_dim = dim // num_heads
    B = 2
    frame_seqlen = 8
    num_frames = 4  # 4 frames of frame_seqlen tokens each
    S = num_frames * frame_seqlen

    ref = CausalWanSelfAttention(
        dim=dim, num_heads=num_heads, frame_seqlen=frame_seqlen,
        qk_norm=True, num_action_per_block=4, num_state_per_block=1
    ).to(device).eval()

    x = torch.randn(B, S, dim, device=device)

    # Create freqs as the model does: concat 3 components into (F*H*W, 1, D)
    # For simplicity, treat grid as (num_frames, 1, frame_seqlen//1) but actually
    # freqs needs shape (seq_len, 1, head_dim) for rope_action_apply
    freqs = rope_params(64, head_dim).to(device)[:S].view(S, 1, -1)
    freqs_action = rope_params(64, head_dim).to(device)
    freqs_state = rope_params(64, head_dim).to(device)

    with torch.no_grad():
        ref_out, _ = ref(x, freqs=freqs, freqs_action=freqs_action,
                         freqs_state=freqs_state, action_register_length=None,
                         is_tf=False)

    # Simulate TP: each rank gets local heads, sharded O-proj, sum outputs
    partial_outputs = []
    for rank in range(tp_size):
        model = copy.deepcopy(ref)
        local_num_heads = num_heads // tp_size

        _shard_o_proj(model, num_heads, head_dim, dim, rank, tp_size)

        model._full_num_heads = num_heads
        model.num_heads = local_num_heads
        model.tp_size = tp_size
        model.tp_rank = rank
        model.tp_group = None  # will skip all_reduce since group=None triggers error

        model.attn = AttentionModule(num_heads=local_num_heads, head_dim=head_dim)
        model.causal_attn = AttentionModule(num_heads=local_num_heads, head_dim=head_dim, causal=True)

        # Monkey-patch to skip all_reduce in test
        original_forward = model.forward
        def make_patched_forward(m):
            def patched_forward(*args, **kwargs):
                old_tp = m.tp_size
                m.tp_size = 1  # temporarily disable all_reduce
                result = original_forward(*args, **kwargs)
                m.tp_size = old_tp
                return result
            return patched_forward
        # Instead: just set tp_size=1 for all_reduce skip, keep tp_size for head slicing
        # Actually we need tp_size>1 for the qkv_fn head slicing but tp_size=1 for all_reduce skip
        # Simplest: override o forward

        # Better approach: just override the all_reduce check
        import unittest.mock as mock
        import torch.distributed as dist_mod

        with torch.no_grad(), mock.patch.object(dist_mod, 'all_reduce', lambda *a, **kw: None):
            out, _ = model(x, freqs=freqs, freqs_action=freqs_action,
                           freqs_state=freqs_state, action_register_length=None,
                           is_tf=False)
        partial_outputs.append(out)

    tp_out = sum(partial_outputs)
    diff = (ref_out - tp_out).abs().max().item()
    print(f"  CausalWanSelfAttention TP={tp_size}: max diff = {diff:.2e}", end="")
    assert diff < 1e-4, f"Too large diff: {diff}"
    print(" ✓")


if __name__ == "__main__":
    device = "cuda:0"
    print("=" * 60)
    print("Tensor Parallelism Correctness Tests (single-process)")
    print("=" * 60)

    for tp in [2, 4]:
        print(f"\n--- TP size = {tp} ---")
        test_ffn_tp(tp, device)
        test_cross_attention_tp(tp, device)
        test_i2v_cross_attention_tp(tp, device)
        test_self_attention_tp(tp, device)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
