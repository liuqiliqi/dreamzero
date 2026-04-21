"""Token Merging (ToMe) utilities for DreamZero DiT — KV-only merge.

Only merges K and V for attention acceleration; Q stays at full resolution.
This avoids spatial information loss that occurs when merging the full hidden
state (which causes mosaic / blur artifacts in diffusion outputs).

The merge pattern is deterministic (spatial-only, no metric dependency) so
that KV-cache entries remain consistent across diffusion steps.

Reference: Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023.
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class MergeInfo:
    """Precomputed merge indices — all tensors live on the same device."""
    surviving_indices: torch.Tensor   # [M_total] positions that survive
    merge_src_global: torch.Tensor    # [r * num_frames] src positions to merge
    merge_dst_merged: torch.Tensor    # [r * num_frames] dst positions in merged space
    unmerge_map: torch.Tensor         # [N] original_pos -> merged_pos  (unused for KV-only)
    M_total: int                      # number of surviving tokens
    counts: torch.Tensor              # [M_total] how many tokens merged into each position


# ---------------------------------------------------------------------------
# Deterministic merge indices (spatial checkerboard, no metric dependency)
# ---------------------------------------------------------------------------

@torch.compiler.disable
def compute_merge_indices_deterministic(
    N: int,
    r: int,
    frame_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> MergeInfo:
    """Fixed spatial merge pattern — independent of token values.

    Uses a checkerboard partition (dst = even parity, src = odd parity).
    For each src, picks the nearest dst by Manhattan distance.
    Always merges the first *r* src positions (sorted by raster order).

    Because the pattern does not depend on token content, the same MergeInfo
    can be reused across diffusion steps, keeping KV-cache layout consistent.
    """
    if frame_size <= 0:
        raise ValueError(f"frame_size must be positive, got {frame_size}")
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    if N % frame_size != 0:
        raise ValueError(f"N={N} not divisible by frame_size={frame_size}")

    num_frames = N // frame_size

    frame_h = int(math.isqrt(frame_size))
    while frame_h > 1 and frame_size % frame_h != 0:
        frame_h -= 1
    frame_w = frame_size // frame_h

    rows = torch.arange(frame_h, device=device)
    cols = torch.arange(frame_w, device=device)
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    checker = (grid_rows + grid_cols) % 2 == 0

    dst_local = (grid_rows[checker] * frame_w + grid_cols[checker]).reshape(-1)
    src_local = (grid_rows[~checker] * frame_w + grid_cols[~checker]).reshape(-1)
    dst_local = torch.sort(dst_local).values
    src_local = torch.sort(src_local).values

    n_dst = dst_local.numel()
    n_src = src_local.numel()
    r = min(r, n_src)

    if r <= 0:
        surviving = torch.arange(N, device=device)
        return MergeInfo(
            surviving_indices=surviving,
            merge_src_global=torch.zeros(0, dtype=torch.long, device=device),
            merge_dst_merged=torch.zeros(0, dtype=torch.long, device=device),
            unmerge_map=surviving.clone(),
            M_total=N,
            counts=torch.ones(N, device=device, dtype=dtype),
        )

    # nearest dst for each src (Manhattan distance)
    sr = src_local // frame_w
    sc = src_local % frame_w
    dr = dst_local // frame_w
    dc = dst_local % frame_w
    dist = (sr.unsqueeze(1) - dr.unsqueeze(0)).abs() + \
           (sc.unsqueeze(1) - dc.unsqueeze(0)).abs()
    nearest_dst_idx = dist.argmin(dim=1)  # [n_src] -> index into dst_local

    src_merge_idx = torch.arange(r, device=device)
    src_survive_idx = torch.arange(r, n_src, device=device)
    target_dst_local_idx = nearest_dst_idx[src_merge_idx]

    M_per_frame = n_dst + (n_src - r)
    M_total = M_per_frame * num_frames

    surviving_parts = []
    unmerge_map = torch.empty(N, dtype=torch.long, device=device)
    merge_src_all = []
    merge_dst_all = []

    offset = 0
    for f in range(num_frames):
        fb = f * frame_size
        dst_g = dst_local + fb
        src_surv_g = src_local[src_survive_idx] + fb
        frame_surv = torch.cat([dst_g, src_surv_g]).sort()[0]
        surviving_parts.append(frame_surv)

        unmerge_map[frame_surv] = torch.arange(offset, offset + M_per_frame, device=device)

        ms_g = src_local[src_merge_idx] + fb
        td_g = dst_local[target_dst_local_idx] + fb
        unmerge_map[ms_g] = unmerge_map[td_g]

        merge_src_all.append(ms_g)
        merge_dst_all.append(unmerge_map[td_g])
        offset += M_per_frame

    surviving_indices = torch.cat(surviving_parts)
    merge_src_global = torch.cat(merge_src_all)
    merge_dst_merged = torch.cat(merge_dst_all)

    counts = torch.ones(M_total, device=device, dtype=dtype)
    if merge_src_global.numel() > 0:
        counts.scatter_add_(
            0, merge_dst_merged,
            torch.ones_like(merge_dst_merged, dtype=dtype),
        )

    return MergeInfo(
        surviving_indices=surviving_indices,
        merge_src_global=merge_src_global,
        merge_dst_merged=merge_dst_merged,
        unmerge_map=unmerge_map,
        M_total=M_total,
        counts=counts,
    )


# ---------------------------------------------------------------------------
# Merge / gather helpers  (used on K and V only, never on the full hidden state)
# ---------------------------------------------------------------------------

def merge_kv(x: torch.Tensor, info: MergeInfo) -> torch.Tensor:
    """Merge tokens via weighted averaging — applied to K or V only.

    Args:
        x: [B, N, ...] full-resolution K or V.
        info: MergeInfo (deterministic).

    Returns:
        [B, M_total, ...] merged tensor.
    """
    rest = x.shape[2:]
    merged = x[:, info.surviving_indices].clone()

    if info.merge_src_global.numel() > 0:
        src = x[:, info.merge_src_global]
        idx = info.merge_dst_merged.view(1, -1, *([1] * len(rest)))
        idx = idx.expand(x.shape[0], -1, *rest)
        merged.scatter_add_(1, idx, src)
        c = info.counts.view(1, info.M_total, *([1] * len(rest)))
        merged = merged / c

    return merged


# ---------------------------------------------------------------------------
# Legacy / compat (kept so test_tome.py still imports without error)
# ---------------------------------------------------------------------------

def merge_tokens(x, info):
    """Alias kept for backward compat with tests."""
    return merge_kv(x, info)

def unmerge_tokens(x, info):
    """Naive unmerge — only used in tests, never in the real forward path."""
    return x[:, info.unmerge_map]

def bipartite_soft_matching(metric, r, frame_size):
    """Legacy test wrapper."""
    from groot.vla.model.dreamzero.modules.tome_utils import compute_merge_indices_deterministic
    B, N, C = metric.shape
    n_src = frame_size - frame_size // 2
    r = min(r, n_src)
    if r <= 0:
        identity = lambda x: x
        return identity, identity, torch.arange(N, device=metric.device)
    info = compute_merge_indices_deterministic(N, r, frame_size, metric.device, metric.dtype)
    return (
        lambda x: merge_kv(x, info),
        lambda x: unmerge_tokens(x, info),
        info.surviving_indices,
    )
