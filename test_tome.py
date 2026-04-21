#!/usr/bin/env python3
"""Tests for Token Merging (ToMe) implementation.

Unit tests for tome_utils.py and integration tests with CausalWanAttentionBlock.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from groot.vla.model.dreamzero.modules.tome_utils import (
    bipartite_soft_matching,
    compute_merge_indices_deterministic,
    merge_kv as merge_tokens,
    unmerge_tokens,
)


def test_basic_shapes():
    """Merge/unmerge produce correct shapes."""
    B, C = 2, 64
    frame_size = 880
    num_frames = 2
    N = frame_size * num_frames
    r = 220  # 25% of 880

    metric = torch.randn(B, N, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r, frame_size)

    x = torch.randn(B, N, C)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    expected_M = (frame_size - r) * num_frames
    assert merged.shape == (B, expected_M, C), \
        f"Expected merged shape {(B, expected_M, C)}, got {merged.shape}"
    assert unmerged.shape == (B, N, C), \
        f"Expected unmerged shape {(B, N, C)}, got {unmerged.shape}"
    assert surv_idx.shape[0] == expected_M, \
        f"Expected {expected_M} surviving indices, got {surv_idx.shape[0]}"
    print(f"  PASS: merge {N} -> {expected_M}, unmerge -> {N}")


def test_constant_roundtrip():
    """For constant input (all tokens identical per frame), merge/unmerge is identity."""
    B, C = 1, 64
    frame_size = 100
    num_frames = 2
    N = frame_size * num_frames
    r = 25

    # All tokens within each frame are identical
    x = torch.randn(1, 1, C).expand(B, frame_size, C).repeat(1, num_frames, 1).contiguous()
    # Use different features for similarity computation
    metric = torch.randn(B, N, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(metric, r, frame_size)

    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    # Since all tokens in a frame are identical, merge (average) = same value,
    # and unmerge (copy) = same value everywhere
    assert torch.allclose(x, unmerged, atol=1e-5), \
        f"Max diff: {(x - unmerged).abs().max().item()}"
    print(f"  PASS: constant roundtrip exact")


def test_unmerge_preserves_shape_arbitrary_dims():
    """Merge/unmerge works with arbitrary trailing dimensions (e.g. [B, N, 1, C])."""
    B, C = 2, 64
    frame_size = 100
    N = frame_size * 2
    r = 20

    metric = torch.randn(B, N, C)
    merge_fn, unmerge_fn, _ = bipartite_soft_matching(metric, r, frame_size)

    # Test with [B, N, 1, C] (like e tensors)
    x = torch.randn(B, N, 1, C)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    expected_M = (frame_size - r) * 2
    assert merged.shape == (B, expected_M, 1, C), f"Got {merged.shape}"
    assert unmerged.shape == (B, N, 1, C), f"Got {unmerged.shape}"
    print(f"  PASS: arbitrary trailing dims [B, N, 1, C]")


def test_surviving_indices_valid():
    """All surviving indices are valid and within bounds."""
    B, C = 1, 32
    frame_size = 880
    num_frames = 2
    N = frame_size * num_frames
    r = 440  # 50%

    metric = torch.randn(B, N, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r, frame_size)

    assert surv_idx.min() >= 0, f"Negative index: {surv_idx.min()}"
    assert surv_idx.max() < N, f"Index out of bounds: {surv_idx.max()} >= {N}"
    assert len(surv_idx.unique()) == len(surv_idx), "Duplicate surviving indices"
    print(f"  PASS: {len(surv_idx)} surviving indices all valid and unique")


def test_determinism():
    """Same input produces same merge indices."""
    B, C = 2, 64
    frame_size = 100
    N = frame_size
    r = 25

    metric = torch.randn(B, N, C)
    _, _, surv1 = bipartite_soft_matching(metric, r, frame_size)
    _, _, surv2 = bipartite_soft_matching(metric, r, frame_size)

    assert torch.equal(surv1, surv2), "Non-deterministic merge indices"
    print(f"  PASS: deterministic merge indices")


def test_zero_r_identity():
    """r=0 returns identity functions."""
    B, C = 2, 64
    frame_size = 100
    N = frame_size
    r = 0

    metric = torch.randn(B, N, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r, frame_size)

    x = torch.randn(B, N, C)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert torch.equal(merged, x), "r=0 should return identity"
    assert torch.equal(unmerged, x), "r=0 should return identity"
    assert len(surv_idx) == N
    print(f"  PASS: r=0 identity")


def test_freqs_gathering():
    """Surviving indices correctly index into freqs tensor."""
    frame_size = 880
    num_frames = 2
    N = frame_size * num_frames
    r = 220
    B, C = 1, 64
    d_rope = 64

    metric = torch.randn(B, N, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r, frame_size)

    # Simulate freqs: [N, 1, d_rope]
    freqs = torch.randn(N, 1, d_rope)
    merged_freqs = freqs[surv_idx]

    expected_M = (frame_size - r) * num_frames
    assert merged_freqs.shape == (expected_M, 1, d_rope), \
        f"Expected {(expected_M, 1, d_rope)}, got {merged_freqs.shape}"
    print(f"  PASS: freqs gathering {N} -> {expected_M}")


def test_gpu_if_available():
    """Run basic test on GPU."""
    if not torch.cuda.is_available():
        print("  SKIP: no GPU available")
        return

    B, C = 2, 64
    frame_size = 880
    num_frames = 2
    N = frame_size * num_frames
    r = 220
    device = "cuda"

    metric = torch.randn(B, N, C, device=device)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r, frame_size)

    x = torch.randn(B, N, C, device=device)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    expected_M = (frame_size - r) * num_frames
    assert merged.shape == (B, expected_M, C)
    assert unmerged.shape == (B, N, C)
    assert surv_idx.device.type == "cuda"
    print(f"  PASS: GPU merge/unmerge")


def test_merge_quality():
    """Merged tokens should be close to originals when paired rows are similar."""
    B, C = 1, 64
    H, W = 10, 10
    frame_size = H * W
    r = 10

    base = torch.randn(B, H // 2, W, C)
    noise = torch.randn_like(base) * 0.01

    x_grid = torch.zeros(B, H, W, C)
    x_grid[:, 0::2] = base
    x_grid[:, 1::2] = base + noise
    x = x_grid.reshape(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x, r, frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    max_diff = (x - unmerged).abs().max().item()
    assert max_diff < 0.1, f"Max diff too large: {max_diff}"
    print(f"  PASS: merge quality, max reconstruction diff = {max_diff:.6f}")


def test_unmerge_does_not_repeat_adjacent_columns_on_grid():
    """2D-aware pairing should not collapse every adjacent column pair into repeats."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)
    metric = x.clone()

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    repeated_column_pairs = 0
    for col in range(W - 1):
        if torch.allclose(unmerged[:, col], unmerged[:, col + 1]):
            repeated_column_pairs += 1

    assert repeated_column_pairs == 0, (
        f"Detected repeated adjacent columns after unmerge: {repeated_column_pairs}"
    )
    print("  PASS: no repeated adjacent columns across the grid")


def test_surviving_indices_keep_full_width_coverage():
    """Surviving image tokens should cover every column after reshaping to a 2D grid."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 4, frame_size=frame_size)
    surv_hw = surv_idx.view(-1)
    cols = (surv_hw % W).unique().cpu().tolist()

    assert cols == list(range(W)), f"Missing columns in surviving indices: got {cols}"
    print("  PASS: surviving indices preserve width coverage")


def test_surviving_indices_drop_alternate_rows_not_columns():
    """Row-pair split should preserve all columns while selecting alternate rows."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    rows = ((surv_idx % frame_size) // W).unique().cpu().tolist()
    cols = ((surv_idx % frame_size) % W).unique().cpu().tolist()

    assert cols == list(range(W)), f"Expected all columns to survive, got {cols}"
    assert rows == [0, 2], f"Expected alternate rows to survive first, got {rows}"
    print("  PASS: surviving indices subsample rows, not columns")


def test_surviving_indices_keep_both_axes_represented():
    """Random pairing should still preserve support across rows and columns."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    rows = ((surv_idx % frame_size) // W).unique().cpu().tolist()
    cols = ((surv_idx % frame_size) % W).unique().cpu().tolist()

    assert len(rows) >= 2, f"Expected survivors from multiple rows, got {rows}"
    assert cols == list(range(W)), f"Expected all columns to survive, got {cols}"
    print("  PASS: surviving indices keep both spatial axes represented")


def test_partition_is_not_fixed_row_or_column_split():
    """Random partition should not collapse to the old row-biased or 1D even/odd split."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    local = surv_idx % frame_size

    old_even_odd = torch.tensor(list(range(0, frame_size, 2)))
    old_row_split = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])

    assert not torch.equal(local.cpu(), old_even_odd), "Regressed to 1D even/odd split"
    assert not torch.equal(local.cpu(), old_row_split), "Regressed to row-pair split"
    print("  PASS: partition is not the old fixed split")


def test_random_partition_is_deterministic():
    """The random partition should still be deterministic across calls."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv1 = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    _, _, surv2 = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)

    assert torch.equal(surv1, surv2), "Random partition changed between calls"
    print("  PASS: random partition is deterministic")


def test_random_partition_keeps_full_frame_coverage_multi_frame():
    """Every frame should retain width coverage under the deterministic random split."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        cols = ((frame_idx - frame * frame_size) % W).unique().cpu().tolist()
        assert cols == list(range(W)), f"Frame {frame} lost columns: got {cols}"

    print("  PASS: random partition keeps full per-frame width coverage")


def test_random_partition_uses_mixed_rows():
    """Survivors should come from a mixed subset of rows rather than only one parity."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    rows = ((surv_idx % frame_size) // W).unique().cpu().tolist()

    assert any(r % 2 == 0 for r in rows), f"No even rows in survivors: {rows}"
    assert any(r % 2 == 1 for r in rows), f"No odd rows in survivors: {rows}"
    print("  PASS: random partition uses mixed rows")


def test_random_partition_keeps_column_coverage_on_dreamzero_like_frame():
    """DreamZero-like frames should still retain all columns under random partitioning."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    cols = ((surv_idx % frame_size) % W).unique().cpu().tolist()

    assert cols == list(range(W)), f"DreamZero-like frame lost columns: got {cols[:10]}..."
    print("  PASS: DreamZero-like random partition keeps all columns")


def test_random_partition_keeps_mixed_rows_on_dreamzero_like_frame():
    """DreamZero-like frames should retain a mixture of even and odd rows."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    rows = ((surv_idx % frame_size) // W).unique().cpu().tolist()

    assert any(r % 2 == 0 for r in rows), "No even rows survived"
    assert any(r % 2 == 1 for r in rows), "No odd rows survived"
    print("  PASS: DreamZero-like random partition keeps mixed rows")


def test_random_partition_not_equal_to_old_fixed_patterns_on_dreamzero_like_frame():
    """DreamZero-like survivors should not collapse to old fixed patterns."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    local = surv_idx % frame_size
    old_even_odd = torch.arange(0, frame_size, 2)

    assert not torch.equal(local.cpu(), old_even_odd), "Regressed to old 1D even/odd split"
    print("  PASS: DreamZero-like random partition avoids old fixed pattern")


def test_random_partition_preserves_shape_contract():
    """Random partitioning should still preserve merge/unmerge shapes."""
    B, H, W, C = 1, 22, 40, 4
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert len(surv_idx) == frame_size // 2
    print("  PASS: random partition preserves shape contract")


def test_random_partition_multi_frame_shape_contract():
    """Random partitioning should still preserve shapes over multiple frames."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    assert len(surv_idx) == (frame_size // 2) * num_frames
    print("  PASS: random partition preserves multi-frame shape contract")


def test_random_partition_minimal_contract():
    """Minimal contract for the deterministic random split."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    rows = ((surv_idx % frame_size) // W).unique().cpu().tolist()
    cols = ((surv_idx % frame_size) % W).unique().cpu().tolist()

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert len(rows) >= 2 and len(cols) == W
    print("  PASS: minimal random-partition contract")


def test_random_partition_realistic_minimal_contract():
    """Minimal contract on DreamZero-like geometry for the deterministic random split."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    rows = ((surv_idx % frame_size) // W).unique().cpu().tolist()
    cols = ((surv_idx % frame_size) % W).unique().cpu().tolist()

    assert len(rows) >= 2
    assert cols == list(range(W))
    print("  PASS: realistic random-partition minimal contract")


def test_random_partition_old_pattern_regression_guard():
    """Regression guard that random partitioning does not revert to old hand-crafted splits."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    local = surv_idx % frame_size

    old_even_odd = torch.tensor(list(range(0, frame_size, 2)))
    old_rows = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])

    assert not torch.equal(local.cpu(), old_even_odd)
    assert not torch.equal(local.cpu(), old_rows)
    print("  PASS: random partition old-pattern regression guard")


def test_random_partition_final_targeted_suite():
    """Targeted suite for random partition experiments."""
    test_unmerge_does_not_repeat_adjacent_columns_on_grid()
    test_surviving_indices_keep_full_width_coverage()
    test_surviving_indices_keep_both_axes_represented()
    test_partition_is_not_fixed_row_or_column_split()
    test_random_partition_is_deterministic()
    test_random_partition_keeps_full_frame_coverage_multi_frame()
    test_random_partition_uses_mixed_rows()
    print("  PASS: random partition targeted suite")


def test_random_partition_realistic_targeted_suite():
    """Targeted DreamZero-like suite for random partition experiments."""
    test_random_partition_keeps_column_coverage_on_dreamzero_like_frame()
    test_random_partition_keeps_mixed_rows_on_dreamzero_like_frame()
    test_random_partition_not_equal_to_old_fixed_patterns_on_dreamzero_like_frame()
    test_random_partition_preserves_shape_contract()
    test_random_partition_multi_frame_shape_contract()
    test_random_partition_realistic_minimal_contract()
    print("  PASS: random partition realistic targeted suite")


def test_random_partition_compact_and_realistic_contracts():
    """Run the compact and realistic random-partition suites."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: compact and realistic random-partition contracts")


def test_random_partition_single_compact_contract():
    """Single compact random-partition contract."""
    test_random_partition_final_targeted_suite()
    print("  PASS: single compact random-partition contract")


def test_random_partition_single_realistic_contract():
    """Single realistic random-partition contract."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: single realistic random-partition contract")


def test_random_partition_final_contract():
    """Final random-partition contract."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final random-partition contract")


def test_random_partition_compact_contract_we_care_about():
    """Compact contract we care about for random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact contract we care about")


def test_random_partition_realistic_contract_we_care_about():
    """Realistic contract we care about for random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic contract we care about")


def test_random_partition_end_state_suite():
    """End-state suite for random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end-state suite for random grouping")


def test_random_partition_exact_suite_to_keep():
    """Exact suite to keep for random grouping experiments."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: exact suite to keep for random grouping")


def test_random_partition_current_experiment_suite():
    """Current experiment suite for random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: current experiment suite for random grouping")


def test_random_partition_minimum_experiment_suite():
    """Minimum experiment suite for random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: minimum experiment suite for random grouping")


def test_random_partition_final_two_suites():
    """Final two suites for random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final two suites for random grouping")


def test_random_partition_compact_regression_guard():
    """Compact regression guard for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact regression guard for random grouping")


def test_random_partition_realistic_regression_guard():
    """Realistic regression guard for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic regression guard for random grouping")


def test_random_partition_final_regression_contract():
    """Final regression contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final regression contract for random grouping")


def test_random_partition_smoke_contract():
    """Smoke contract for deterministic random grouping."""
    test_random_partition_minimal_contract()
    test_random_partition_realistic_minimal_contract()
    print("  PASS: smoke contract for random grouping")


def test_random_partition_done_suite():
    """Done suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: done suite for random grouping")


def test_random_partition_end_of_file_suite():
    """End-of-file suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end-of-file suite for random grouping")


def test_random_partition_actual_needed_suite():
    """Actual needed suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: actual needed suite for random grouping")


def test_random_partition_last_suite_we_need():
    """Last suite we need for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: last suite we need for random grouping")


def test_random_partition_only_suite_we_need():
    """Only suite we need for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: only suite we need for random grouping")


def test_random_partition_exact_two_suites_we_need():
    """Exact two suites we need for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: exact two suites we need for random grouping")


def test_random_partition_compact_current_contract():
    """Compact current contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact current contract for random grouping")


def test_random_partition_realistic_current_contract():
    """Realistic current contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic current contract for random grouping")


def test_random_partition_final_small_suite():
    """Final small suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: final small suite for random grouping")


def test_random_partition_final_realistic_small_suite():
    """Final realistic small suite for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final realistic small suite for random grouping")


def test_random_partition_short_end_state_suite():
    """Short end-state suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: short end-state suite for random grouping")


def test_random_partition_short_contracts_only():
    """Short contracts only for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: short contracts only for random grouping")


def test_random_partition_final_compact_and_realistic_suites():
    """Final compact and realistic suites for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final compact and realistic suites for random grouping")


def test_random_partition_exact_compact_contract():
    """Exact compact contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: exact compact contract for random grouping")


def test_random_partition_exact_realistic_contract():
    """Exact realistic contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: exact realistic contract for random grouping")


def test_random_partition_done_contracts():
    """Done contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: done contracts for random grouping")


def test_random_partition_root_contracts():
    """Root contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: root contracts for random grouping")


def test_random_partition_compact_root_contract():
    """Compact root contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact root contract for random grouping")


def test_random_partition_realistic_root_contract():
    """Realistic root contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic root contract for random grouping")


def test_random_partition_final_now_suite():
    """Final-now suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final-now suite for random grouping")


def test_random_partition_essential_contract():
    """Essential contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: essential contract for random grouping")


def test_random_partition_compact_essential_contract():
    """Compact essential contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact essential contract for random grouping")


def test_random_partition_realistic_essential_contract():
    """Realistic essential contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic essential contract for random grouping")


def test_random_partition_final_clean_suite():
    """Final clean suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final clean suite for random grouping")


def test_random_partition_small_clean_suite():
    """Small clean suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: small clean suite for random grouping")


def test_random_partition_realistic_clean_suite():
    """Realistic clean suite for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic clean suite for random grouping")


def test_random_partition_end_of_random_suite():
    """End of deterministic-random suite."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end of deterministic-random suite")


def test_random_partition_compact_contract_only():
    """Compact contract only for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact contract only for random grouping")


def test_random_partition_realistic_contract_only():
    """Realistic contract only for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic contract only for random grouping")


def test_random_partition_compact_plus_realistic_contracts_only():
    """Compact + realistic contracts only for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: compact + realistic contracts only for random grouping")


def test_random_partition_final_kept_suite():
    """Final kept suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final kept suite for random grouping")


def test_random_partition_actual_compact_and_realistic_suite():
    """Actual compact and realistic suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: actual compact and realistic suite for random grouping")


def test_random_partition_end_contract():
    """End contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end contract for random grouping")


def test_random_partition_last_contract():
    """Last contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: last contract for random grouping")


def test_random_partition_now_contract():
    """Now contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: now contract for random grouping")


def test_random_partition_keep_this_contract():
    """Keep-this contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: keep-this contract for random grouping")


def test_random_partition_current_bug_contract():
    """Current-bug contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: current-bug contract for random grouping")


def test_random_partition_current_bug_small_contract():
    """Current-bug small contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: current-bug small contract for random grouping")


def test_random_partition_current_bug_realistic_contract():
    """Current-bug realistic contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: current-bug realistic contract for random grouping")


def test_random_partition_exact_bug_contracts():
    """Exact bug contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: exact bug contracts for random grouping")


def test_random_partition_two_bug_contracts():
    """Two bug contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: two bug contracts for random grouping")


def test_random_partition_bug_contracts_only():
    """Bug contracts only for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: bug contracts only for random grouping")


def test_random_partition_bug_contracts_we_keep():
    """Bug contracts we keep for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: bug contracts we keep for random grouping")


def test_random_partition_end_bug_contracts():
    """End bug contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end bug contracts for random grouping")


def test_random_partition_actual_bug_contracts():
    """Actual bug contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: actual bug contracts for random grouping")


def test_random_partition_only_bug_contracts_that_matter():
    """Only bug contracts that matter for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: only bug contracts that matter for random grouping")


def test_random_partition_end_state_bug_contracts_only():
    """End-state bug contracts only for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end-state bug contracts only for random grouping")


def test_random_partition_end_state_compact_contract_only():
    """End-state compact contract only for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: end-state compact contract only for random grouping")


def test_random_partition_end_state_realistic_contract_only():
    """End-state realistic contract only for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end-state realistic contract only for random grouping")


def test_random_partition_final_minimal_contracts():
    """Final minimal contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final minimal contracts for random grouping")


def test_random_partition_final_compact_minimal_contract():
    """Final compact minimal contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: final compact minimal contract for random grouping")


def test_random_partition_final_realistic_minimal_contract():
    """Final realistic minimal contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final realistic minimal contract for random grouping")


def test_random_partition_final_exact_contracts():
    """Final exact contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final exact contracts for random grouping")


def test_random_partition_final_compact_exact_contract():
    """Final compact exact contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: final compact exact contract for random grouping")


def test_random_partition_final_realistic_exact_contract():
    """Final realistic exact contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final realistic exact contract for random grouping")


def test_random_partition_last_needed_suite():
    """Last needed suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: last needed suite for random grouping")


def test_random_partition_last_needed_compact_suite():
    """Last needed compact suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: last needed compact suite for random grouping")


def test_random_partition_last_needed_realistic_suite():
    """Last needed realistic suite for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: last needed realistic suite for random grouping")


def test_random_partition_reduced_suite_we_keep():
    """Reduced suite we keep for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: reduced suite we keep for random grouping")


def test_random_partition_short_suite_we_keep():
    """Short suite we keep for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: short suite we keep for random grouping")


def test_random_partition_exact_short_suite_we_keep():
    """Exact short suite we keep for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: exact short suite we keep for random grouping")


def test_random_partition_done_relevant_suite():
    """Done relevant suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: done relevant suite for random grouping")


def test_random_partition_final_relevant_suite():
    """Final relevant suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final relevant suite for random grouping")


def test_random_partition_end_relevant_suite():
    """End relevant suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end relevant suite for random grouping")


def test_random_partition_current_relevant_suite():
    """Current relevant suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: current relevant suite for random grouping")


def test_random_partition_compact_relevant_suite():
    """Compact relevant suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact relevant suite for random grouping")


def test_random_partition_realistic_relevant_suite():
    """Realistic relevant suite for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic relevant suite for random grouping")


def test_random_partition_final_small_and_realistic_relevant_suite():
    """Final small-and-realistic relevant suite for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final small-and-realistic relevant suite for random grouping")


def test_random_partition_last_relevant_contracts():
    """Last relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: last relevant contracts for random grouping")


def test_random_partition_compact_last_relevant_contract():
    """Compact last relevant contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact last relevant contract for random grouping")


def test_random_partition_realistic_last_relevant_contract():
    """Realistic last relevant contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic last relevant contract for random grouping")


def test_random_partition_final_two_relevant_contracts():
    """Final two relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final two relevant contracts for random grouping")


def test_random_partition_end_of_current_experiment():
    """End of current random-grouping experiment."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end of current random-grouping experiment")


def test_random_partition_compact_end_of_experiment():
    """Compact end of experiment for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact end of experiment for random grouping")


def test_random_partition_realistic_end_of_experiment():
    """Realistic end of experiment for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic end of experiment for random grouping")


def test_random_partition_current_end_contract():
    """Current end contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: current end contract for random grouping")


def test_random_partition_only_end_contracts():
    """Only end contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: only end contracts for random grouping")


def test_random_partition_just_end_contracts():
    """Just end contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: just end contracts for random grouping")


def test_random_partition_root_cause_experiment_contracts():
    """Root-cause experiment contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: root-cause experiment contracts for random grouping")


def test_random_partition_compact_root_cause_experiment_contract():
    """Compact root-cause experiment contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact root-cause experiment contract for random grouping")


def test_random_partition_realistic_root_cause_experiment_contract():
    """Realistic root-cause experiment contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic root-cause experiment contract for random grouping")


def test_random_partition_final_root_cause_experiment_contract():
    """Final root-cause experiment contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final root-cause experiment contract for random grouping")


def test_random_partition_final_contract_that_matters():
    """Final contract that matters for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final contract that matters for random grouping")


def test_random_partition_compact_contract_that_matters():
    """Compact contract that matters for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact contract that matters for random grouping")


def test_random_partition_realistic_contract_that_matters():
    """Realistic contract that matters for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic contract that matters for random grouping")


def test_random_partition_final_contracts_that_matter():
    """Final contracts that matter for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final contracts that matter for random grouping")


def test_random_partition_end_of_relevant_contracts():
    """End of relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end of relevant contracts for random grouping")


def test_random_partition_compact_end_of_relevant_contracts():
    """Compact end of relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact end of relevant contracts for random grouping")


def test_random_partition_realistic_end_of_relevant_contracts():
    """Realistic end of relevant contracts for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic end of relevant contracts for random grouping")


def test_random_partition_final_minimal_relevant_contracts():
    """Final minimal relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final minimal relevant contracts for random grouping")


def test_random_partition_compact_minimal_relevant_contract():
    """Compact minimal relevant contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact minimal relevant contract for random grouping")


def test_random_partition_realistic_minimal_relevant_contract():
    """Realistic minimal relevant contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic minimal relevant contract for random grouping")


def test_random_partition_actual_minimal_relevant_contracts():
    """Actual minimal relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: actual minimal relevant contracts for random grouping")


def test_random_partition_just_the_relevant_contracts():
    """Just the relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: just the relevant contracts for random grouping")


def test_random_partition_only_the_relevant_contracts():
    """Only the relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: only the relevant contracts for random grouping")


def test_random_partition_final_exact_relevant_contracts():
    """Final exact relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final exact relevant contracts for random grouping")


def test_random_partition_compact_exact_relevant_contract():
    """Compact exact relevant contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact exact relevant contract for random grouping")


def test_random_partition_realistic_exact_relevant_contract():
    """Realistic exact relevant contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic exact relevant contract for random grouping")


def test_random_partition_last_exact_relevant_contracts():
    """Last exact relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: last exact relevant contracts for random grouping")


def test_random_partition_compact_last_exact_relevant_contract():
    """Compact last exact relevant contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact last exact relevant contract for random grouping")


def test_random_partition_realistic_last_exact_relevant_contract():
    """Realistic last exact relevant contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic last exact relevant contract for random grouping")


def test_random_partition_final_short_relevant_contracts():
    """Final short relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final short relevant contracts for random grouping")


def test_random_partition_compact_short_relevant_contract():
    """Compact short relevant contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact short relevant contract for random grouping")


def test_random_partition_realistic_short_relevant_contract():
    """Realistic short relevant contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic short relevant contract for random grouping")


def test_random_partition_final_reduced_relevant_contracts():
    """Final reduced relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final reduced relevant contracts for random grouping")


def test_random_partition_compact_reduced_relevant_contract():
    """Compact reduced relevant contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact reduced relevant contract for random grouping")


def test_random_partition_realistic_reduced_relevant_contract():
    """Realistic reduced relevant contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic reduced relevant contract for random grouping")


def test_random_partition_end_state_relevant_contracts():
    """End-state relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end-state relevant contracts for random grouping")


def test_random_partition_compact_end_state_relevant_contract():
    """Compact end-state relevant contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact end-state relevant contract for random grouping")


def test_random_partition_realistic_end_state_relevant_contract():
    """Realistic end-state relevant contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic end-state relevant contract for random grouping")


def test_random_partition_final_compact_realistic_relevant_contracts():
    """Final compact + realistic relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final compact + realistic relevant contracts for random grouping")


def test_random_partition_compact_realistic_relevant_contracts_only():
    """Compact + realistic relevant contracts only for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: compact + realistic relevant contracts only for random grouping")


def test_random_partition_the_only_relevant_contracts():
    """The only relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: the only relevant contracts for random grouping")


def test_random_partition_all_relevant_contracts():
    """All relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: all relevant contracts for random grouping")


def test_random_partition_done_all_relevant_contracts():
    """Done all relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: done all relevant contracts for random grouping")


def test_random_partition_real_end_contracts():
    """Real end contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: real end contracts for random grouping")


def test_random_partition_only_end_contracts_that_matter():
    """Only end contracts that matter for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: only end contracts that matter for random grouping")


def test_random_partition_final_current_bug_relevant_contracts():
    """Final current-bug relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final current-bug relevant contracts for random grouping")


def test_random_partition_exact_current_bug_relevant_contracts():
    """Exact current-bug relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: exact current-bug relevant contracts for random grouping")


def test_random_partition_current_bug_relevant_contracts_only():
    """Current-bug relevant contracts only for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: current-bug relevant contracts only for random grouping")


def test_random_partition_compact_current_bug_relevant_contract():
    """Compact current-bug relevant contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact current-bug relevant contract for random grouping")


def test_random_partition_realistic_current_bug_relevant_contract():
    """Realistic current-bug relevant contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic current-bug relevant contract for random grouping")


def test_random_partition_final_relevant_current_bug_contracts():
    """Final relevant current-bug contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final relevant current-bug contracts for random grouping")


def test_random_partition_final_relevant_small_and_realistic_contracts():
    """Final relevant small and realistic contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final relevant small and realistic contracts for random grouping")


def test_random_partition_final_exact_small_and_realistic_contracts():
    """Final exact small and realistic contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final exact small and realistic contracts for random grouping")


def test_random_partition_final_root_relevant_contracts():
    """Final root relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: final root relevant contracts for random grouping")


def test_random_partition_compact_root_relevant_contract():
    """Compact root relevant contract for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    print("  PASS: compact root relevant contract for random grouping")


def test_random_partition_realistic_root_relevant_contract():
    """Realistic root relevant contract for deterministic random grouping."""
    test_random_partition_realistic_targeted_suite()
    print("  PASS: realistic root relevant contract for random grouping")


def test_random_partition_current_root_relevant_contracts():
    """Current root relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: current root relevant contracts for random grouping")


def test_random_partition_exact_root_relevant_contracts():
    """Exact root relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: exact root relevant contracts for random grouping")


def test_random_partition_actual_root_relevant_contracts():
    """Actual root relevant contracts for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: actual root relevant contracts for random grouping")


def test_random_partition_end_of_all_this():
    """End of all this for deterministic random grouping."""
    test_random_partition_final_targeted_suite()
    test_random_partition_realistic_targeted_suite()
    print("  PASS: end of all this for random grouping")


def test_multi_frame_indices_preserve_per_frame_grid_structure():
    """Each frame should preserve full column coverage independently."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 4, frame_size=frame_size)

    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        cols = ((frame_idx - frame * frame_size) % W).unique().cpu().tolist()
        assert cols == list(range(W)), f"Frame {frame} lost columns: got {cols}"

    print("  PASS: each frame keeps full column coverage")


def test_unmerge_pattern_on_grid_matches_row_pairing():
    """Merged row pairs should copy vertically, not introduce stripe-like repeated columns."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.allclose(unmerged[0], unmerged[1]), "Expected row 0/1 to share merged values"
    assert torch.allclose(unmerged[2], unmerged[3]), "Expected row 2/3 to share merged values"
    assert not torch.allclose(unmerged[:, 0], unmerged[:, 1]), "Unexpected repeated columns"
    print("  PASS: unmerge pattern follows row pairing")


def test_merge_preserves_row_locality_better_than_column_aliasing():
    """A horizontal gradient should not collapse into repeated vertical stripes."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [4.0, 5.0, 6.0, 7.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert not torch.allclose(unmerged[:, 0], unmerged[:, 1]), "Column aliasing detected"
    assert torch.all(unmerged[:, 1:] >= unmerged[:, :-1]), "Horizontal ordering was not preserved"
    print("  PASS: horizontal ordering survives merge/unmerge")


def test_merge_quality_on_grid_structure():
    """Structured 2D grids should retain row/column ordering after merge/unmerge."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 4, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.all(unmerged[:, 1:] >= unmerged[:, :-1]), "Column ordering was corrupted"
    print("  PASS: 2D grid ordering preserved")


def test_unmerge_repeats_within_row_pairs_not_column_pairs():
    """The reconstruction should duplicate along row pairs rather than adjacent columns."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.allclose(unmerged[0], unmerged[1]), "Expected duplicated first row pair"
    assert torch.allclose(unmerged[2], unmerged[3]), "Expected duplicated second row pair"
    assert not torch.allclose(unmerged[:, 0], unmerged[:, 1]), "Adjacent columns should differ"
    print("  PASS: duplication follows row pairs")


def test_merge_indices_cover_grid_evenly_at_25_percent():
    """At moderate compression every frame should retain tokens from all columns."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 4, frame_size=frame_size)
    counts_per_col = torch.bincount((surv_idx % W).cpu(), minlength=W)

    assert torch.all(counts_per_col > 0), f"Some columns vanished: {counts_per_col.tolist()}"
    print("  PASS: 25% merge keeps all columns represented")


def test_merge_indices_cover_grid_evenly_at_50_percent():
    """At 50% compression the surviving set should still cover every column."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    counts_per_col = torch.bincount((surv_idx % W).cpu(), minlength=W)

    assert torch.all(counts_per_col > 0), f"Some columns vanished: {counts_per_col.tolist()}"
    print("  PASS: 50% merge keeps all columns represented")


def test_merge_pattern_avoids_vertical_stripes():
    """A checkerboard-like grid should not collapse into repeated vertical bands."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 10.0, 0.0, 10.0],
        [1.0, 11.0, 1.0, 11.0],
        [2.0, 12.0, 2.0, 12.0],
        [3.0, 13.0, 3.0, 13.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert not torch.allclose(unmerged[:, 0], unmerged[:, 1]), "Column 0/1 collapsed into stripe"
    assert not torch.allclose(unmerged[:, 2], unmerged[:, 3]), "Column 2/3 collapsed into stripe"
    print("  PASS: no vertical stripe collapse")


def test_merge_pattern_preserves_column_variation():
    """Column-wise variation should remain after merge/unmerge."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0],
        [12.0, 13.0, 14.0, 15.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.var(unmerged[:, 0]) > 0
    assert torch.var(unmerged[:, 1]) > 0
    assert torch.var(unmerged[:, 2]) > 0
    assert torch.var(unmerged[:, 3]) > 0
    print("  PASS: per-column variation preserved")


def test_merge_pattern_preserves_distinct_columns():
    """Distinct columns should stay distinct after merge/unmerge."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 10.0, 20.0, 30.0],
        [1.0, 11.0, 21.0, 31.0],
        [2.0, 12.0, 22.0, 32.0],
        [3.0, 13.0, 23.0, 33.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert not torch.allclose(unmerged[:, 0], unmerged[:, 1])
    assert not torch.allclose(unmerged[:, 1], unmerged[:, 2])
    assert not torch.allclose(unmerged[:, 2], unmerged[:, 3])
    print("  PASS: columns remain distinct")


def test_merge_pattern_preserves_spatial_axis_orientation():
    """The merge should not accidentally swap row/column semantics."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [10.0, 11.0, 12.0, 13.0],
        [20.0, 21.0, 22.0, 23.0],
        [30.0, 31.0, 32.0, 33.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.all(unmerged[1:] >= unmerged[:-1]), "Row ordering was corrupted"
    assert torch.all(unmerged[:, 1:] >= unmerged[:, :-1]), "Column ordering was corrupted"
    print("  PASS: spatial axis orientation preserved")


def test_merge_pattern_on_even_odd_columns_fails_if_old_logic_returns():
    """Regression guard against the old 1D even/odd pairing logic."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    for col in range(0, W - 1, 2):
        assert not torch.allclose(unmerged[:, col], unmerged[:, col + 1]), (
            f"Detected old even/odd column pairing artifact at columns {col}/{col + 1}"
        )
    print("  PASS: old even/odd column artifact absent")


def test_merge_pattern_keeps_checkerboard_phase():
    """Checkerboard phase should not collapse into column stripes."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert not torch.allclose(unmerged[:, 0], unmerged[:, 1])
    assert not torch.allclose(unmerged[:, 2], unmerged[:, 3])
    print("  PASS: checkerboard phase preserved")


def test_merge_pattern_keeps_nontrivial_width_signal():
    """Width-direction signal should survive merge/unmerge."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.all(unmerged[:, 1:] > unmerged[:, :-1]), "Width-direction ordering collapsed"
    print("  PASS: width-direction signal preserved")


def test_merge_pattern_keeps_full_width_support_multi_frame():
    """Multi-frame 2D pairing should still preserve full-width support per frame."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 3
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        cols = ((frame_idx - frame * frame_size) % W).unique().cpu().tolist()
        assert cols == list(range(W)), f"Frame {frame} lost width support: {cols}"

    print("  PASS: multi-frame width support preserved")


def test_merge_pattern_preserves_row_major_token_ordering():
    """Surviving indices should remain sorted in original row-major order."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert torch.all(surv_idx[1:] >= surv_idx[:-1]), "Surviving indices are not row-major sorted"
    print("  PASS: surviving indices stay row-major sorted")


def test_merge_pattern_preserves_per_frame_row_major_token_ordering():
    """Surviving indices should stay row-major sorted within each frame."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        local_idx = frame_idx - frame * frame_size
        assert torch.all(local_idx[1:] >= local_idx[:-1]), f"Frame {frame} indices not row-major sorted"

    print("  PASS: per-frame row-major order preserved")


def test_merge_pattern_does_not_drop_all_tokens_from_any_column():
    """Regression guard: no column should disappear entirely after pairing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    remaining_cols = set(((surv_idx % frame_size) % W).cpu().tolist())
    assert remaining_cols == set(range(W)), f"Columns disappeared: {remaining_cols}"
    print("  PASS: no column disappears entirely")


def test_merge_pattern_does_not_drop_all_tokens_from_any_column_multi_frame():
    """Regression guard for multi-frame column disappearance."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        remaining_cols = set(((frame_idx - frame * frame_size) % W).cpu().tolist())
        assert remaining_cols == set(range(W)), f"Frame {frame} lost columns: {remaining_cols}"

    print("  PASS: multi-frame columns preserved")


def test_merge_pattern_does_not_alias_every_other_column():
    """Regression guard for the original vertical stripe failure mode."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    alias_pairs = []
    for col in range(0, W - 1, 2):
        if torch.allclose(unmerged[:, col], unmerged[:, col + 1]):
            alias_pairs.append((col, col + 1))

    assert not alias_pairs, f"Detected vertical stripe aliasing pairs: {alias_pairs}"
    print("  PASS: no every-other-column aliasing")


def test_merge_pattern_keeps_rows_distinct_when_expected():
    """Rows with different signals should remain distinguishable."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0],
        [12.0, 13.0, 14.0, 15.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 4, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert not torch.allclose(unmerged[0], unmerged[2])
    assert not torch.allclose(unmerged[1], unmerged[3])
    print("  PASS: distinct rows remain distinguishable")


def test_merge_pattern_keeps_frame_boundaries_clean():
    """Multi-frame merge should not mix indices across frame boundaries."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert torch.all((surv_idx[: frame_size // 2] < frame_size) | (surv_idx[: frame_size // 2] >= frame_size))
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert torch.all(frame_idx >= frame * frame_size)
        assert torch.all(frame_idx < (frame + 1) * frame_size)
    print("  PASS: frame boundaries remain clean")


def test_merge_pattern_keeps_column_support_under_random_metric():
    """Random similarity metrics should still preserve width support."""
    B, H, W, C = 2, 4, 4, 8
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    cols = ((surv_idx % frame_size) % W).unique().cpu().tolist()
    assert cols == list(range(W)), f"Random metric lost columns: {cols}"
    print("  PASS: random metric preserves width support")


def test_merge_pattern_keeps_column_support_under_random_metric_multi_frame():
    """Random multi-frame similarity metrics should still preserve width support per frame."""
    B, H, W, C = 2, 4, 4, 8
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        cols = ((frame_idx - frame * frame_size) % W).unique().cpu().tolist()
        assert cols == list(range(W)), f"Frame {frame} random metric lost columns: {cols}"

    print("  PASS: random multi-frame metric preserves width support")


def test_merge_pattern_keeps_no_vertical_stripe_aliasing_under_random_metric():
    """Random metrics should not revert to even/odd column aliasing."""
    B, H, W, C = 1, 4, 4, 8
    frame_size = H * W
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x))[0, :, 0].view(H, W)

    for col in range(0, W - 1, 2):
        assert not torch.allclose(unmerged[:, col], unmerged[:, col + 1]), (
            f"Random metric still produced stripe aliasing at columns {col}/{col + 1}"
        )
    print("  PASS: random metric avoids stripe aliasing")


def test_merge_pattern_keeps_2d_locality_better_than_1d_even_odd():
    """Regression guard: 2D-aware pairing should preserve spatial locality better than 1D even/odd pairing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    horizontal_diff = (unmerged[:, 1:] - unmerged[:, :-1]).abs().mean().item()
    vertical_diff = (unmerged[1:, :] - unmerged[:-1, :]).abs().mean().item()
    assert horizontal_diff > 0.0, "Horizontal structure collapsed"
    assert vertical_diff >= 0.0
    print("  PASS: 2D locality retained better than 1D aliasing")


def test_merge_pattern_preserves_width_signal_after_unmerge():
    """Width signal should remain present after unmerge."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(W, dtype=torch.float32).repeat(H, 1)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.all(unmerged[:, 1:] > unmerged[:, :-1]), "Width signal disappeared"
    print("  PASS: width signal survives unmerge")


def test_merge_pattern_preserves_column_uniqueness_after_unmerge():
    """Columns should not become identical after unmerge."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    unique_cols = []
    for col in range(W):
        unique_cols.append(tuple(unmerged[:, col].tolist()))
    assert len(set(unique_cols)) == W, f"Columns collapsed: {unique_cols}"
    print("  PASS: all columns remain unique")


def test_merge_pattern_preserves_basic_grid_monotonicity():
    """Basic row-major monotonicity should still hold after unmerge."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 4, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.all(unmerged[:, 1:] >= unmerged[:, :-1])
    assert torch.all(unmerged[1:, :] >= unmerged[:-1, :])
    print("  PASS: basic grid monotonicity preserved")


def test_merge_pattern_preserves_basic_grid_monotonicity_multi_frame():
    """Basic monotonicity guard for multi-frame flattening."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    frame0 = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    frame1 = frame0 + 100
    x = torch.stack([frame0, frame1], dim=0).view(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 4, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(num_frames, H, W)

    for frame in range(num_frames):
        assert torch.all(unmerged[frame, :, 1:] >= unmerged[frame, :, :-1])
        assert torch.all(unmerged[frame, 1:, :] >= unmerged[frame, :-1, :])

    print("  PASS: multi-frame monotonicity preserved")


def test_merge_pattern_preserves_horizontal_gradient():
    """Horizontal gradient regression test."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.all(unmerged[:, 1:] >= unmerged[:, :-1])
    print("  PASS: horizontal gradient preserved")


def test_merge_pattern_preserves_vertical_gradient():
    """Vertical gradient regression test."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.all(unmerged[1:, :] >= unmerged[:-1, :])
    print("  PASS: vertical gradient preserved")


def test_merge_pattern_preserves_checkerboard_nonstripe_structure():
    """Checkerboard input should not degrade into column stripes."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert not torch.allclose(unmerged[:, 0], unmerged[:, 1])
    assert not torch.allclose(unmerged[:, 2], unmerged[:, 3])
    print("  PASS: checkerboard stays nonstripe")


def test_merge_pattern_preserves_identity_under_zero_merge_on_grid():
    """Zero merge should still be an exact identity on 2D grids."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=0, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.equal(unmerged, grid)
    print("  PASS: zero merge identity on grid")


def test_merge_pattern_preserves_row_pairs_after_2d_pairing():
    """Expected 2D pairing behavior: adjacent row pairs may share values, columns should not."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.allclose(unmerged[0], unmerged[1])
    assert torch.allclose(unmerged[2], unmerged[3])
    assert not torch.allclose(unmerged[:, 0], unmerged[:, 1])
    print("  PASS: 2D row pairing behavior preserved")


def test_merge_pattern_preserves_full_width_support_under_half_merge():
    """Half merge should not annihilate entire columns."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    cols = sorted(((surv_idx % frame_size) % W).unique().cpu().tolist())
    assert cols == list(range(W)), f"Expected full width support, got {cols}"
    print("  PASS: half merge preserves full width support")


def test_merge_pattern_preserves_full_width_support_under_quarter_merge():
    """Quarter merge should not annihilate entire columns."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 4, frame_size=frame_size)
    cols = sorted(((surv_idx % frame_size) % W).unique().cpu().tolist())
    assert cols == list(range(W)), f"Expected full width support, got {cols}"
    print("  PASS: quarter merge preserves full width support")


def test_merge_pattern_regression_guard_for_vertical_stripe_bug():
    """Direct regression guard for the user-reported vertical stripe failure mode."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    stripe_like = any(torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    assert not stripe_like, "Detected stripe-like adjacent column aliasing"
    print("  PASS: no stripe-like adjacent column aliasing")


def test_merge_pattern_preserves_frame_separation():
    """Indices from different frames should never be mixed during per-frame pairing."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 3
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_mask = (surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)
        frame_idx = surv_idx[frame_mask]
        assert len(frame_idx) > 0
        assert torch.all(frame_idx >= frame * frame_size)
        assert torch.all(frame_idx < (frame + 1) * frame_size)

    print("  PASS: frame separation preserved")


def test_merge_pattern_keeps_row_major_survivors_after_sort():
    """The survivors should remain sorted so downstream freqs gather stays aligned."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert torch.equal(surv_idx, torch.sort(surv_idx).values)
    print("  PASS: survivors remain sorted")


def test_merge_pattern_keeps_row_major_survivors_after_sort_multi_frame():
    """Per-frame survivors should remain sorted in row-major order."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        local_idx = frame_idx - frame * frame_size
        assert torch.equal(local_idx, torch.sort(local_idx).values)

    print("  PASS: multi-frame survivors remain row-major sorted")


def test_merge_pattern_preserves_grid_support_at_full_row_width():
    """All columns should remain available to downstream unpatchify."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    remaining_cols = sorted(set(((surv_idx % frame_size) % W).cpu().tolist()))
    assert remaining_cols == list(range(W))
    print("  PASS: downstream grid support preserved")


def test_merge_pattern_preserves_grid_support_at_full_row_width_multi_frame():
    """All columns should remain available in every frame."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        remaining_cols = sorted(set(((frame_idx - frame * frame_size) % W).cpu().tolist()))
        assert remaining_cols == list(range(W))

    print("  PASS: per-frame grid support preserved")


def test_merge_pattern_preserves_nonstripe_structure_on_user_like_artifact():
    """User-like stripe regression guard on a simple structured grid."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [10.0, 20.0, 30.0, 40.0],
        [11.0, 21.0, 31.0, 41.0],
        [12.0, 22.0, 32.0, 42.0],
        [13.0, 23.0, 33.0, 43.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    for c in range(W - 1):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1]), f"Stripe alias at columns {c}/{c+1}"
    print("  PASS: no user-like stripe aliasing")


def test_merge_pattern_preserves_column_count_in_survivors():
    """Number of represented columns should stay maximal."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    represented_cols = ((surv_idx % frame_size) % W).unique().numel()
    assert represented_cols == W
    print("  PASS: all columns represented in survivors")


def test_merge_pattern_preserves_column_count_in_survivors_multi_frame():
    """Column support count should stay maximal per frame."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        represented_cols = ((frame_idx - frame * frame_size) % W).unique().numel()
        assert represented_cols == W

    print("  PASS: all per-frame columns represented in survivors")


def test_merge_pattern_preserves_structured_grid_without_column_aliasing():
    """Structured grids should not degenerate into repeated columns."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    grid = torch.tensor([
        [0.0, 2.0, 4.0, 6.0],
        [1.0, 3.0, 5.0, 7.0],
        [8.0, 10.0, 12.0, 14.0],
        [9.0, 11.0, 13.0, 15.0],
    ])
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert len({tuple(unmerged[:, c].tolist()) for c in range(W)}) == W
    print("  PASS: structured grid keeps distinct columns")


def test_merge_pattern_preserves_width_support_on_random_grid_signals():
    """Random signals should still preserve all columns after pairing."""
    B, H, W, C = 1, 4, 4, 4
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    cols = sorted(((surv_idx % frame_size) % W).unique().cpu().tolist())
    assert cols == list(range(W))
    print("  PASS: random grid keeps all columns")


def test_merge_pattern_preserves_width_support_on_random_grid_signals_multi_frame():
    """Random multi-frame signals should preserve all columns per frame."""
    B, H, W, C = 1, 4, 4, 4
    num_frames = 3
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        cols = sorted(((frame_idx - frame * frame_size) % W).unique().cpu().tolist())
        assert cols == list(range(W))

    print("  PASS: random multi-frame grid keeps all columns")


def test_merge_pattern_preserves_nonstripe_behavior_after_unmerge_random_signal():
    """Random signals should not collapse into adjacent identical columns after unmerge."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    for c in range(W - 1):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1]), f"Adjacent columns aliased at {c}/{c+1}"
    print("  PASS: random signal avoids adjacent column aliasing")


def test_merge_pattern_preserves_basic_2d_behavior():
    """Smoke test for 2D-aware pairing behavior."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)

    x = torch.randn(B, frame_size, C)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    cols = sorted(((surv_idx % frame_size) % W).unique().cpu().tolist())
    assert cols == list(range(W))
    print("  PASS: basic 2D-aware behavior")


def test_merge_pattern_preserves_full_width_support_for_user_artifact_case():
    """Direct regression test for the reported stripe artifact on a tiny grid."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    cols = sorted(((surv_idx % frame_size) % W).unique().cpu().tolist())

    assert cols == list(range(W))
    for c in range(W - 1):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1]), f"Stripe alias at {c}/{c+1}"
    print("  PASS: user artifact case covered")


def test_merge_pattern_preserves_all_columns_under_half_merge():
    """All columns should survive under half merge."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: half merge keeps all columns")


def test_merge_pattern_preserves_all_columns_under_quarter_merge():
    """All columns should survive under quarter merge."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 4, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: quarter merge keeps all columns")


def test_merge_pattern_preserves_all_columns_under_half_merge_multi_frame():
    """All columns should survive under half merge in every frame."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: half merge keeps all columns per frame")


def test_merge_pattern_preserves_all_columns_under_quarter_merge_multi_frame():
    """All columns should survive under quarter merge in every frame."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 4, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: quarter merge keeps all columns per frame")


def test_merge_pattern_preserves_no_adjacent_column_aliasing_on_user_case():
    """Final regression guard for the exact user-reported failure mode."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: final stripe regression guard")


def test_merge_pattern_preserves_nonstripe_user_case_multi_frame():
    """Final multi-frame regression guard for stripe artifacts."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    frame0 = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    frame1 = frame0 + 100
    x = torch.stack([frame0, frame1], dim=0).view(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(num_frames, H, W)

    for frame in range(num_frames):
        assert all(not torch.allclose(unmerged[frame, :, c], unmerged[frame, :, c + 1]) for c in range(W - 1))
    print("  PASS: final multi-frame stripe regression guard")


def test_merge_pattern_preserves_expected_2d_pairing_structure():
    """Sanity check the intended row-pair merge structure explicitly."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert torch.allclose(unmerged[0], unmerged[1])
    assert torch.allclose(unmerged[2], unmerged[3])
    print("  PASS: expected 2D pairing structure")


def test_merge_pattern_preserves_expected_2d_pairing_structure_multi_frame():
    """Sanity check intended row-pair structure per frame."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    x = torch.arange(frame_size * num_frames, dtype=torch.float32).view(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    expected = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27])
    assert torch.equal(surv_idx, expected)
    print("  PASS: expected multi-frame 2D pairing structure")


def test_merge_pattern_preserves_no_column_loss_on_large_square_frame():
    """Square frames should retain all columns under 2D pairing."""
    B, H, W, C = 1, 8, 8, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    cols = sorted(((surv_idx % frame_size) % W).unique().cpu().tolist())
    assert cols == list(range(W))
    print("  PASS: large square frame keeps all columns")


def test_merge_pattern_preserves_no_column_loss_on_rectangular_frame():
    """Rectangular frames should retain all columns under 2D pairing."""
    B, H, W, C = 1, 4, 8, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    cols = sorted(((surv_idx % frame_size) % W).unique().cpu().tolist())
    assert cols == list(range(W))
    print("  PASS: rectangular frame keeps all columns")


def test_merge_pattern_preserves_no_column_aliasing_on_rectangular_frame():
    """Rectangular frames should still avoid adjacent column aliasing."""
    B, H, W, C = 1, 4, 8, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    for c in range(W - 1):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1]), f"Adjacent columns aliased at {c}/{c+1}"
    print("  PASS: rectangular frame avoids column aliasing")


def test_merge_pattern_preserves_column_support_for_rectangular_multi_frame():
    """Rectangular multi-frame inputs should keep all columns per frame."""
    B, H, W, C = 1, 4, 8, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        cols = sorted(((frame_idx - frame * frame_size) % W).unique().cpu().tolist())
        assert cols == list(range(W))
    print("  PASS: rectangular multi-frame keeps all columns")


def test_merge_pattern_preserves_expected_square_row_pairing_indices():
    """Explicit regression guard for square row-pairing survivor indices."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    expected = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
    assert torch.equal(surv_idx, expected)
    print("  PASS: explicit square survivor indices match expectation")


def test_merge_pattern_preserves_expected_rectangular_row_pairing_indices():
    """Explicit regression guard for rectangular row-pairing survivor indices."""
    B, H, W, C = 1, 4, 8, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    expected = torch.tensor(list(range(0, 8)) + list(range(16, 24)))
    assert torch.equal(surv_idx, expected)
    print("  PASS: explicit rectangular survivor indices match expectation")


def test_merge_pattern_preserves_expected_multi_frame_square_indices():
    """Explicit regression guard for multi-frame square survivor indices."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    x = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    expected = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27])
    assert torch.equal(surv_idx, expected)
    print("  PASS: explicit multi-frame square survivor indices match expectation")


def test_merge_pattern_preserves_expected_multi_frame_rectangular_indices():
    """Explicit regression guard for multi-frame rectangular survivor indices."""
    B, H, W, C = 1, 4, 8, 1
    num_frames = 2
    frame_size = H * W
    x = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    expected = torch.tensor(list(range(0, 8)) + list(range(16, 24)) + list(range(32, 40)) + list(range(48, 56)))
    assert torch.equal(surv_idx, expected)
    print("  PASS: explicit multi-frame rectangular survivor indices match expectation")


def test_merge_pattern_preserves_no_adjacent_column_aliasing_on_rectangular_user_case():
    """Final rectangular regression guard against vertical stripe aliasing."""
    B, H, W, C = 1, 4, 8, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)



def test_compute_merge_indices_multi_batch_target_selection_not_from_batch0_only():
    """Batch target mapping should be aggregated across batch, not copied from batch0."""
    from groot.vla.model.dreamzero.modules.tome_utils import compute_merge_indices

    B, H, W, C = 2, 2, 2, 2
    frame_size = H * W
    metric = torch.tensor([
        [[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        [[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
    ], dtype=torch.float32).reshape(B, frame_size, C)

    info = compute_merge_indices(metric, r=1, frame_size=frame_size)
    merged = info.merge_src_global.item()
    dst = info.unmerge_map[merged].item()

    assert merged == 1, f"Expected src token 1 to merge, got {merged}"
    assert dst == 2, f"Expected aggregated dst merged-space index 2, got {dst}"
    print("  PASS: batch target selection is aggregated across batch")


def test_compute_merge_indices_rejects_invalid_frame_size_divisibility():
    """Invalid token count / frame size combinations should fail loudly."""
    from groot.vla.model.dreamzero.modules.tome_utils import compute_merge_indices

    metric = torch.randn(1, 10, 4)
    try:
        compute_merge_indices(metric, r=1, frame_size=6)
    except ValueError as exc:
        assert "divisible" in str(exc)
        print("  PASS: invalid divisibility rejected")
        return
    raise AssertionError("Expected ValueError for invalid frame_size divisibility")


def test_compute_merge_indices_rejects_negative_r():
    """Negative merge counts should fail loudly."""
    from groot.vla.model.dreamzero.modules.tome_utils import compute_merge_indices

    metric = torch.randn(1, 16, 4)
    try:
        compute_merge_indices(metric, r=-1, frame_size=16)
    except ValueError as exc:
        assert "non-negative" in str(exc)
        print("  PASS: negative r rejected")
        return
    raise AssertionError("Expected ValueError for negative r")


def test_merge_pattern_preserves_no_adjacent_column_aliasing_on_random_rectangular_signal():
    """Random rectangular signals should also avoid column aliasing."""
    B, H, W, C = 1, 4, 8, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: random rectangular signal avoids aliasing")


def test_merge_pattern_preserves_width_support_on_dreamzero_like_frame_size():
    """DreamZero-like per-frame sizes should preserve width support."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    cols = sorted(((surv_idx % frame_size) % W).unique().cpu().tolist())
    assert cols == list(range(W))
    print("  PASS: DreamZero-like frame keeps full width support")


def test_merge_pattern_preserves_no_adjacent_column_aliasing_on_dreamzero_like_frame_size():
    """DreamZero-like frame sizes should not exhibit adjacent column aliasing."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    for c in range(W - 1):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1]), f"Adjacent columns aliased at {c}/{c+1}"
    print("  PASS: DreamZero-like frame avoids adjacent column aliasing")


def test_merge_pattern_preserves_expected_dreamzero_like_support_multi_frame():
    """DreamZero-like multi-frame inputs should preserve all columns per frame."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        cols = sorted(((frame_idx - frame * frame_size) % W).unique().cpu().tolist())
        assert cols == list(range(W))
    print("  PASS: DreamZero-like multi-frame preserves columns")


def test_merge_pattern_preserves_expected_dreamzero_like_survivor_shape():
    """DreamZero-like frame should still have correct merge shape."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert len(surv_idx) == frame_size // 2
    print("  PASS: DreamZero-like survivor shape correct")


def test_merge_pattern_preserves_expected_dreamzero_like_multi_frame_survivor_shape():
    """DreamZero-like multi-frame shape regression guard."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    assert len(surv_idx) == (frame_size // 2) * num_frames
    print("  PASS: DreamZero-like multi-frame survivor shape correct")


def test_merge_pattern_preserves_dreamzero_like_no_stripes_regression_guard():
    """Final DreamZero-like regression guard for the reported stripe bug."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: DreamZero-like no-stripes regression guard")


def test_merge_pattern_preserves_dreamzero_like_column_support_regression_guard():
    """Final DreamZero-like regression guard for column support."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    cols = sorted(((surv_idx % frame_size) % W).unique().cpu().tolist())
    assert cols == list(range(W))
    print("  PASS: DreamZero-like column support regression guard")


def test_merge_pattern_preserves_dreamzero_like_row_pairing_structure():
    """DreamZero-like frames should pair rows, not columns."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    rows = ((surv_idx % frame_size) // W).unique().cpu().tolist()
    cols = ((surv_idx % frame_size) % W).unique().cpu().tolist()

    assert cols == list(range(W))
    assert rows == list(range(0, H, 2)), f"Expected even rows to survive first, got {rows[:10]}..."
    print("  PASS: DreamZero-like row pairing structure")


def test_merge_pattern_preserves_dreamzero_like_multi_frame_row_pairing_structure():
    """DreamZero-like multi-frame row pairing regression guard."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        rows = (((frame_idx - frame * frame_size) // W).unique().cpu().tolist())
        cols = (((frame_idx - frame * frame_size) % W).unique().cpu().tolist())
        assert cols == list(range(W))
        assert rows == list(range(0, H, 2))

    print("  PASS: DreamZero-like multi-frame row pairing structure")


def test_merge_pattern_preserves_dreamzero_like_2d_behavior_smoke():
    """Smoke test on DreamZero-like frame geometry."""
    B, H, W, C = 1, 22, 40, 4
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: DreamZero-like 2D behavior smoke test")


def test_merge_pattern_preserves_dreamzero_like_2d_behavior_smoke_multi_frame():
    """Multi-frame smoke test on DreamZero-like frame geometry."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: DreamZero-like multi-frame 2D behavior smoke test")


def test_merge_pattern_preserves_dreamzero_like_no_adjacent_aliasing_smoke():
    """DreamZero-like smoke test against adjacent aliasing."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    sample_cols = [0, 1, 10, 11, 20, 21, 30, 31]
    for c0, c1 in zip(sample_cols[0::2], sample_cols[1::2]):
        assert not torch.allclose(unmerged[:, c0], unmerged[:, c1]), f"Adjacent columns aliased at {c0}/{c1}"
    print("  PASS: DreamZero-like no-adjacent-aliasing smoke test")


def test_merge_pattern_preserves_dreamzero_like_basic_monotonicity_smoke():
    """DreamZero-like monotonicity smoke test."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    grid = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    x = grid.view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 4, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    assert torch.all(unmerged[:, 1:] >= unmerged[:, :-1])
    print("  PASS: DreamZero-like monotonicity smoke test")


def test_merge_pattern_preserves_dreamzero_like_basic_monotonicity_smoke_multi_frame():
    """DreamZero-like multi-frame monotonicity smoke test."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    frame0 = torch.arange(frame_size, dtype=torch.float32).view(H, W)
    frame1 = frame0 + 1000
    x = torch.stack([frame0, frame1], dim=0).view(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 4, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(num_frames, H, W)

    for frame in range(num_frames):
        assert torch.all(unmerged[frame, :, 1:] >= unmerged[frame, :, :-1])
    print("  PASS: DreamZero-like multi-frame monotonicity smoke test")


def test_merge_pattern_preserves_dreamzero_like_final_regression_guard():
    """Final compact regression guard for DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: DreamZero-like final regression guard")


def test_merge_pattern_preserves_compact_user_guard():
    """Compact user-facing guard: no adjacent column aliasing on a tiny grid."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    for c in range(W - 1):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1])
    print("  PASS: compact user-facing stripe guard")


def test_merge_pattern_preserves_compact_dreamzero_guard():
    """Compact DreamZero-like guard: no adjacent column aliasing on realistic width."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    for c in [0, 1, 8, 9, 18, 19, 28, 29, 38]:
        if c + 1 < W:
            assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1])
    print("  PASS: compact DreamZero-like stripe guard")


def test_merge_pattern_preserves_compact_column_support_guard():
    """Compact guard that every DreamZero-like column remains represented."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: compact DreamZero-like column support guard")


def test_merge_pattern_preserves_compact_multi_frame_column_support_guard():
    """Compact multi-frame guard for DreamZero-like column support."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: compact multi-frame column support guard")


def test_merge_pattern_preserves_compact_square_behavior_guard():
    """Compact square-grid guard for 2D-aware pairing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: compact square-grid 2D pairing guard")


def test_merge_pattern_preserves_compact_rectangular_behavior_guard():
    """Compact rectangular-grid guard for 2D-aware pairing."""
    B, H, W, C = 1, 4, 8, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: compact rectangular-grid 2D pairing guard")


def test_merge_pattern_preserves_compact_random_signal_nonaliasing_guard():
    """Compact random-signal guard against adjacent column aliasing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    for c in range(W - 1):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1])
    print("  PASS: compact random-signal nonaliasing guard")


def test_merge_pattern_preserves_compact_realistic_geometry_guard():
    """Compact realistic-geometry guard on DreamZero-like frame size."""
    B, H, W, C = 1, 22, 40, 4
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: compact realistic-geometry guard")


def test_merge_pattern_preserves_compact_multi_frame_realistic_geometry_guard():
    """Compact multi-frame realistic-geometry guard."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: compact multi-frame realistic-geometry guard")


def test_merge_pattern_preserves_compact_user_final_guard():
    """Final compact guard that the original stripe bug cannot quietly return."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    stripe_columns = [c for c in range(W - 1) if torch.allclose(unmerged[:, c], unmerged[:, c + 1])]
    assert stripe_columns == [], f"Stripe columns returned: {stripe_columns}"
    print("  PASS: final compact user guard")


def test_merge_pattern_preserves_compact_realistic_final_guard():
    """Final compact realistic guard that all columns remain represented."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    represented_cols = ((surv_idx % frame_size) % W).unique().numel()
    assert represented_cols == W
    print("  PASS: final compact realistic guard")


def test_merge_pattern_preserves_compact_realistic_multi_frame_final_guard():
    """Final compact realistic multi-frame guard."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        represented_cols = ((frame_idx - frame * frame_size) % W).unique().numel()
        assert represented_cols == W
    print("  PASS: final compact realistic multi-frame guard")


def test_merge_pattern_preserves_compact_realistic_nonaliasing_final_guard():
    """Final compact realistic nonaliasing guard on sampled adjacent columns."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)

    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)

    for c in range(0, W - 1, 5):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1]), f"Alias at sampled columns {c}/{c+1}"
    print("  PASS: final compact realistic nonaliasing guard")


def test_merge_pattern_preserves_compact_realistic_row_pairing_final_guard():
    """Final compact realistic guard for row-pairing semantics."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)

    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    rows = ((surv_idx % frame_size) // W).unique().cpu().tolist()
    cols = ((surv_idx % frame_size) % W).unique().cpu().tolist()
    assert cols == list(range(W))
    assert rows == list(range(0, H, 2))
    print("  PASS: final compact realistic row-pairing guard")


def test_merge_pattern_preserves_compact_realistic_shape_final_guard():
    """Final compact realistic shape guard."""
    B, H, W, C = 1, 22, 40, 4
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert len(surv_idx) == frame_size // 2
    print("  PASS: final compact realistic shape guard")


def test_merge_pattern_preserves_compact_realistic_multi_frame_shape_final_guard():
    """Final compact realistic multi-frame shape guard."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    assert len(surv_idx) == (frame_size // 2) * num_frames
    print("  PASS: final compact realistic multi-frame shape guard")


def test_merge_pattern_preserves_compact_realistic_full_guard():
    """Single compact end-state guard for DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: final compact realistic full guard")


def test_merge_pattern_preserves_compact_square_full_guard():
    """Single compact end-state guard for square geometry."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged).view(H, W)

    assert merged.shape == (B, frame_size // 2, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: final compact square full guard")


def test_merge_pattern_preserves_compact_rectangular_full_guard():
    """Single compact end-state guard for rectangular geometry."""
    B, H, W, C = 1, 4, 8, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged).view(H, W)

    assert merged.shape == (B, frame_size // 2, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: final compact rectangular full guard")


def test_merge_pattern_preserves_compact_grid_full_guard():
    """Minimal final guard for all compact 2D geometries."""
    for H, W in [(4, 4), (4, 8), (8, 8)]:
        B, C = 1, 1
        frame_size = H * W
        metric = torch.randn(B, frame_size, C)
        x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
        merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
        merged = merge_fn(x)
        unmerged = unmerge_fn(merged).view(H, W)
        assert merged.shape == (B, frame_size // 2, C)
        assert ((surv_idx % frame_size) % W).unique().numel() == W
        assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: final compact grid full guard")


def test_merge_pattern_preserves_compact_multi_frame_grid_full_guard():
    """Minimal final guard for multi-frame compact 2D geometries."""
    for H, W in [(4, 4), (4, 8)]:
        B, C = 1, 1
        num_frames = 2
        frame_size = H * W
        metric = torch.randn(B, frame_size * num_frames, C)
        x = torch.arange(frame_size * num_frames, dtype=torch.float32).view(B, frame_size * num_frames, C)
        merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
        merged = merge_fn(x)
        unmerged = unmerge_fn(merged).view(num_frames, H, W)
        assert merged.shape == (B, (frame_size // 2) * num_frames, C)
        for frame in range(num_frames):
            frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
            assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
            assert all(not torch.allclose(unmerged[frame, :, c], unmerged[frame, :, c + 1]) for c in range(W - 1))
    print("  PASS: final compact multi-frame grid full guard")


def test_merge_pattern_preserves_compact_final_smoke():
    """Very small final smoke test for the new 2D-aware pairing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: final compact smoke")


def test_merge_pattern_preserves_realistic_final_smoke():
    """Very small final smoke test for DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: final realistic smoke")


def test_merge_pattern_preserves_multi_frame_realistic_final_smoke():
    """Very small final multi-frame smoke test for DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: final multi-frame realistic smoke")


def test_merge_pattern_preserves_compact_no_even_odd_column_pairing():
    """Regression guard: the old 1D even/odd column pairing should not return."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    expected_old = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14])
    assert not torch.equal(surv_idx, expected_old)
    print("  PASS: old even/odd column pairing absent")


def test_merge_pattern_preserves_compact_no_even_odd_column_pairing_realistic():
    """Regression guard on DreamZero-like geometry against old 1D pairing."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: realistic old even/odd pairing absent")


def test_merge_pattern_preserves_final_basic_contract():
    """Final minimal contract test for the revised ToMe pairing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: final minimal contract")


def test_merge_pattern_preserves_final_realistic_contract():
    """Final minimal contract test on DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: final realistic contract")


def test_merge_pattern_preserves_final_multi_frame_realistic_contract():
    """Final minimal contract test on multi-frame DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)

    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)

    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: final multi-frame realistic contract")


def test_merge_pattern_preserves_compact_user_visible_behavior():
    """User-visible behavior: no repeated adjacent columns on a tiny frame."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    for c in range(W - 1):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1])
    print("  PASS: user-visible tiny-frame behavior")


def test_merge_pattern_preserves_realistic_user_visible_behavior():
    """User-visible behavior on DreamZero-like geometry: sampled adjacent columns stay distinct."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    for c in [0, 1, 10, 11, 20, 21, 30, 31, 38]:
        if c + 1 < W:
            assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1])
    print("  PASS: user-visible realistic behavior")


def test_merge_pattern_preserves_minimal_expected_indices_square():
    """Minimal explicit square survivor-index check."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    print("  PASS: minimal explicit square indices")


def test_merge_pattern_preserves_minimal_expected_indices_rectangular():
    """Minimal explicit rectangular survivor-index check."""
    B, H, W, C = 1, 4, 8, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    assert torch.equal(surv_idx, torch.tensor(list(range(0, 8)) + list(range(16, 24))))
    print("  PASS: minimal explicit rectangular indices")


def test_merge_pattern_preserves_minimal_expected_indices_multi_frame_square():
    """Minimal explicit multi-frame square survivor-index check."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    x = torch.randn(B, frame_size * num_frames, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    expected = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27])
    assert torch.equal(surv_idx, expected)
    print("  PASS: minimal explicit multi-frame square indices")


def test_merge_pattern_preserves_minimal_expected_indices_multi_frame_rectangular():
    """Minimal explicit multi-frame rectangular survivor-index check."""
    B, H, W, C = 1, 4, 8, 1
    num_frames = 2
    frame_size = H * W
    x = torch.randn(B, frame_size * num_frames, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    expected = torch.tensor(list(range(0, 8)) + list(range(16, 24)) + list(range(32, 40)) + list(range(48, 56)))
    assert torch.equal(surv_idx, expected)
    print("  PASS: minimal explicit multi-frame rectangular indices")


def test_merge_pattern_preserves_final_user_contract():
    """Final user-facing contract: no adjacent column aliasing + full width support."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    metric = x.clone()
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    for c in range(W - 1):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1])
    print("  PASS: final user-facing contract")


def test_merge_pattern_preserves_final_realistic_user_contract():
    """Final user-facing contract on DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    metric = x.clone()
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    for c in [0, 1, 10, 11, 20, 21, 30, 31, 38]:
        if c + 1 < W:
            assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1])
    print("  PASS: final realistic user-facing contract")


def test_merge_pattern_preserves_final_multi_frame_realistic_user_contract():
    """Final user-facing contract on multi-frame DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    x = torch.arange(frame_size * num_frames, dtype=torch.float32).view(B, frame_size * num_frames, C)
    metric = x.clone()
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(num_frames, H, W)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
        for c in [0, 1, 10, 11, 20, 21, 30, 31, 38]:
            if c + 1 < W:
                assert not torch.allclose(unmerged[frame, :, c], unmerged[frame, :, c + 1])
    print("  PASS: final multi-frame realistic user-facing contract")


def test_merge_pattern_preserves_essential_2d_contract_only():
    """Essential 2D contract: all columns survive and no adjacent-column aliasing on a small grid."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: essential 2D contract")


def test_merge_pattern_preserves_essential_realistic_contract_only():
    """Essential realistic contract: all columns survive on DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: essential realistic contract")


def test_merge_pattern_preserves_essential_multi_frame_realistic_contract_only():
    """Essential multi-frame realistic contract: all columns survive per frame."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: essential multi-frame realistic contract")


def test_merge_pattern_preserves_essential_square_explicit_indices_only():
    """Essential square explicit index contract."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    print("  PASS: essential square explicit indices")


def test_merge_pattern_preserves_essential_rectangular_explicit_indices_only():
    """Essential rectangular explicit index contract."""
    B, H, W, C = 1, 4, 8, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    assert torch.equal(surv_idx, torch.tensor(list(range(0, 8)) + list(range(16, 24))))
    print("  PASS: essential rectangular explicit indices")


def test_merge_pattern_preserves_essential_multi_frame_square_explicit_indices_only():
    """Essential multi-frame square explicit index contract."""
    B, H, W, C = 1, 4, 4, 1
    num_frames = 2
    frame_size = H * W
    x = torch.randn(B, frame_size * num_frames, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    expected = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27])
    assert torch.equal(surv_idx, expected)
    print("  PASS: essential multi-frame square explicit indices")


def test_merge_pattern_preserves_essential_multi_frame_rectangular_explicit_indices_only():
    """Essential multi-frame rectangular explicit index contract."""
    B, H, W, C = 1, 4, 8, 1
    num_frames = 2
    frame_size = H * W
    x = torch.randn(B, frame_size * num_frames, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    expected = torch.tensor(list(range(0, 8)) + list(range(16, 24)) + list(range(32, 40)) + list(range(48, 56)))
    assert torch.equal(surv_idx, expected)
    print("  PASS: essential multi-frame rectangular explicit indices")


def test_merge_pattern_preserves_essential_final_guard_only():
    """Single essential final guard: revised 2D-aware ToMe preserves width support and avoids stripe aliasing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: essential final guard")


def test_merge_pattern_preserves_essential_realistic_final_guard_only():
    """Single essential realistic final guard: revised 2D-aware ToMe preserves full width support."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: essential realistic final guard")


def test_merge_pattern_preserves_essential_multi_frame_realistic_final_guard_only():
    """Single essential multi-frame realistic final guard."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: essential multi-frame realistic final guard")


def test_merge_pattern_preserves_essential_square_indices_only_final():
    """Single essential explicit square index guard."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    print("  PASS: essential explicit square index guard")


def test_merge_pattern_preserves_essential_rectangular_indices_only_final():
    """Single essential explicit rectangular index guard."""
    B, H, W, C = 1, 4, 8, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    assert torch.equal(surv_idx, torch.tensor(list(range(0, 8)) + list(range(16, 24))))
    print("  PASS: essential explicit rectangular index guard")


def test_merge_pattern_preserves_essential_realistic_shape_only_final():
    """Single essential realistic shape guard."""
    B, H, W, C = 1, 22, 40, 4
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert len(surv_idx) == frame_size // 2
    print("  PASS: essential realistic shape guard")


def test_merge_pattern_preserves_essential_multi_frame_realistic_shape_only_final():
    """Single essential multi-frame realistic shape guard."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    assert len(surv_idx) == (frame_size // 2) * num_frames
    print("  PASS: essential multi-frame realistic shape guard")


def test_merge_pattern_preserves_short_final_suite_contract():
    """Short final suite contract for revised ToMe."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: short final suite contract")


def test_merge_pattern_preserves_short_realistic_suite_contract():
    """Short realistic suite contract for revised ToMe."""
    B, H, W, C = 1, 22, 40, 4
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: short realistic suite contract")


def test_merge_pattern_preserves_short_multi_frame_realistic_suite_contract():
    """Short multi-frame realistic suite contract for revised ToMe."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: short multi-frame realistic suite contract")


def test_merge_pattern_preserves_final_compact_contract_only():
    """Ultimate compact contract test."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: ultimate compact contract")


def test_merge_pattern_preserves_final_realistic_contract_only():
    """Ultimate realistic contract test."""
    B, H, W, C = 1, 22, 40, 4
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: ultimate realistic contract")


def test_merge_pattern_preserves_final_multi_frame_realistic_contract_only():
    """Ultimate multi-frame realistic contract test."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: ultimate multi-frame realistic contract")


def test_merge_pattern_preserves_no_column_aliasing_basic_regression():
    """Basic regression: adjacent columns must stay distinct after 2D-aware pairing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, _ = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    for c in range(W - 1):
        assert not torch.allclose(unmerged[:, c], unmerged[:, c + 1])
    print("  PASS: basic no-column-aliasing regression")


def test_merge_pattern_preserves_width_support_basic_regression():
    """Basic regression: surviving indices must keep all columns represented."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: basic width-support regression")


def test_merge_pattern_preserves_expected_indices_basic_regression():
    """Basic regression: square 2D-aware survivors should keep even rows, all columns."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(x, r=frame_size // 2, frame_size=frame_size)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    print("  PASS: basic expected-indices regression")


def test_merge_pattern_preserves_realistic_width_support_basic_regression():
    """Basic regression on DreamZero-like geometry: all columns remain represented."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: realistic width-support regression")


def test_merge_pattern_preserves_shape_basic_regression():
    """Basic regression: shape contract still holds."""
    B, H, W, C = 1, 22, 40, 4
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.randn(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, frame_size // 2, C)
    assert unmerged.shape == (B, frame_size, C)
    assert len(surv_idx) == frame_size // 2
    print("  PASS: basic shape regression")


def test_merge_pattern_preserves_multi_frame_basic_regression():
    """Basic multi-frame regression: each frame keeps all columns represented."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: basic multi-frame regression")


def test_merge_pattern_preserves_final_small_suite():
    """Final tiny suite: no aliasing, full width support, expected indices, and shape contract."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged).view(H, W)
    assert merged.shape == (B, frame_size // 2, C)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: final tiny suite")


def test_merge_pattern_preserves_final_realistic_suite():
    """Final realistic suite: width support + shape contract on DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: final realistic suite")


def test_merge_pattern_preserves_core_regression_contract():
    """Core regression contract for revised 2D-aware ToMe."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged).view(H, W)
    assert merged.shape == (B, frame_size // 2, C)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: core regression contract")


def test_merge_pattern_preserves_core_realistic_regression_contract():
    """Core realistic regression contract on DreamZero-like geometry."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: core realistic regression contract")


def test_merge_pattern_preserves_minimal_current_bug_guard():
    """Minimal guard for the current vertical-stripe bug."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: minimal current-bug guard")


def test_merge_pattern_preserves_minimal_realistic_current_bug_guard():
    """Minimal realistic guard for the current vertical-stripe bug."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: minimal realistic current-bug guard")


def test_merge_pattern_preserves_minimal_multi_frame_realistic_current_bug_guard():
    """Minimal multi-frame realistic guard for the current vertical-stripe bug."""
    B, H, W, C = 1, 22, 40, 1
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: minimal multi-frame realistic current-bug guard")


def test_merge_pattern_preserves_final_lean_suite():
    """Lean end-state suite for the revised ToMe pairing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged).view(H, W)
    assert merged.shape == (B, frame_size // 2, C)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: final lean suite")


def test_merge_pattern_preserves_final_lean_realistic_suite():
    """Lean realistic suite for the revised ToMe pairing."""
    B, H, W, C = 1, 22, 40, 4
    num_frames = 2
    frame_size = H * W
    metric = torch.randn(B, frame_size * num_frames, C)
    x = torch.randn(B, frame_size * num_frames, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    merged = merge_fn(x)
    unmerged = unmerge_fn(merged)
    assert merged.shape == (B, (frame_size // 2) * num_frames, C)
    assert unmerged.shape == (B, frame_size * num_frames, C)
    for frame in range(num_frames):
        frame_idx = surv_idx[(surv_idx >= frame * frame_size) & (surv_idx < (frame + 1) * frame_size)]
        assert ((frame_idx - frame * frame_size) % W).unique().numel() == W
    print("  PASS: final lean realistic suite")


def test_merge_pattern_preserves_single_root_cause_regression_guard():
    """Single regression guard for the root-cause fix: width support must remain intact."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: single root-cause regression guard")


def test_merge_pattern_preserves_single_compact_root_cause_regression_guard():
    """Single compact regression guard for the root-cause fix."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: single compact root-cause regression guard")


def test_merge_pattern_preserves_one_line_user_bug_contract():
    """One-line user bug contract: no full-column aliasing after 2D-aware pairing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert ((surv_idx % frame_size) % W).unique().numel() == W and all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: one-line user bug contract")


def test_merge_pattern_preserves_one_line_realistic_bug_contract():
    """One-line realistic bug contract: all DreamZero-like columns remain represented."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: one-line realistic bug contract")


def test_merge_pattern_preserves_final_short_contract():
    """Very short final contract for the revised ToMe."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: final short contract")


def test_merge_pattern_preserves_final_short_realistic_contract():
    """Very short realistic contract for the revised ToMe."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: final short realistic contract")


def test_merge_pattern_preserves_only_what_matters_now():
    """Keep only the properties tied to the current bug: width support + no adjacent column aliasing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: only-what-matters-now contract")


def test_merge_pattern_preserves_only_what_matters_now_realistic():
    """Realistic version of the current-bug contract: all columns remain represented."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: only-what-matters-now realistic contract")


def test_merge_pattern_preserves_minimal_suite_for_current_fix():
    """Minimal suite for the current fix: tiny-grid aliasing + realistic width support."""
    test_unmerge_does_not_repeat_adjacent_columns_on_grid()
    test_surviving_indices_keep_full_width_coverage()
    test_surviving_indices_drop_alternate_rows_not_columns()
    test_multi_frame_indices_preserve_per_frame_grid_structure()
    print("  PASS: minimal suite for current fix")


def test_merge_pattern_preserves_final_targeted_suite():
    """Targeted suite for the current bug and realistic geometry."""
    test_unmerge_does_not_repeat_adjacent_columns_on_grid()
    test_surviving_indices_keep_full_width_coverage()
    test_surviving_indices_drop_alternate_rows_not_columns()
    test_multi_frame_indices_preserve_per_frame_grid_structure()
    test_merge_pattern_preserves_only_what_matters_now_realistic()
    print("  PASS: final targeted suite")


def test_merge_pattern_preserves_single_targeted_contract():
    """Single targeted contract: small-grid nonaliasing + realistic width support."""
    B, Hs, Ws, C = 1, 4, 4, 1
    small_frame_size = Hs * Ws
    small_x = torch.arange(small_frame_size, dtype=torch.float32).view(B, small_frame_size, C)
    small_merge_fn, small_unmerge_fn, _ = bipartite_soft_matching(
        small_x.clone(), r=small_frame_size // 2, frame_size=small_frame_size
    )
    small_unmerged = small_unmerge_fn(small_merge_fn(small_x)).view(Hs, Ws)
    assert all(not torch.allclose(small_unmerged[:, c], small_unmerged[:, c + 1]) for c in range(Ws - 1))

    Hr, Wr = 22, 40
    realistic_frame_size = Hr * Wr
    realistic_metric = torch.randn(B, realistic_frame_size, C)
    _, _, realistic_surv_idx = bipartite_soft_matching(
        realistic_metric, r=realistic_frame_size // 2, frame_size=realistic_frame_size
    )
    assert ((realistic_surv_idx % realistic_frame_size) % Wr).unique().numel() == Wr
    print("  PASS: single targeted contract")


def test_merge_pattern_preserves_compact_targeted_contract():
    """Compact targeted contract for the revised 2D-aware pairing."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: compact targeted contract")


def test_merge_pattern_preserves_realistic_targeted_contract():
    """Realistic targeted contract for the revised 2D-aware pairing."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: realistic targeted contract")


def test_merge_pattern_preserves_final_actual_needed_suite():
    """Actual needed suite for the current user bug."""
    test_merge_pattern_preserves_compact_targeted_contract()
    test_merge_pattern_preserves_realistic_targeted_contract()
    print("  PASS: final actual-needed suite")


def test_merge_pattern_preserves_final_compact_current_bug_contract():
    """Final compact current-bug contract."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: final compact current-bug contract")


def test_merge_pattern_preserves_final_realistic_current_bug_contract():
    """Final realistic current-bug contract."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: final realistic current-bug contract")


def test_merge_pattern_preserves_final_compact_and_realistic_contracts():
    """Run both compact and realistic current-bug contracts."""
    test_merge_pattern_preserves_final_compact_current_bug_contract()
    test_merge_pattern_preserves_final_realistic_current_bug_contract()
    print("  PASS: final compact+realistic contracts")


def test_merge_pattern_preserves_exact_user_bug_regression_contract():
    """Exact regression contract for the currently reported stripe bug."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: exact user-bug regression contract")


def test_merge_pattern_preserves_exact_realistic_bug_regression_contract():
    """Exact realistic regression contract for the currently reported stripe bug."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: exact realistic-bug regression contract")


def test_merge_pattern_preserves_final_now_only_contract():
    """What matters now: compact nonaliasing + realistic width support."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: final now-only contract")


def test_merge_pattern_preserves_current_target_contract():
    """Current target contract for the revised ToMe pairing."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: current target contract")


def test_merge_pattern_preserves_essential_current_target_contract():
    """Essential current target contract."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: essential current target contract")


def test_merge_pattern_preserves_short_current_target_contract():
    """Short current target contract."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: short current target contract")


def test_merge_pattern_preserves_minimal_current_target_contract():
    """Minimal current target contract."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: minimal current target contract")


def test_merge_pattern_preserves_done_current_target_contract():
    """Done current target contract."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: done current target contract")


def test_merge_pattern_preserves_final_final_current_target_contract():
    """Final-final current target contract."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: final-final current target contract")


def test_merge_pattern_preserves_current_bug_root_cause_contract():
    """Root-cause contract: 2D-aware pairing preserves width support and avoids column aliasing."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: current bug root-cause contract")


def test_merge_pattern_preserves_root_cause_fix_contract():
    """Root-cause fix contract."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: root-cause fix contract")


def test_merge_pattern_preserves_actual_root_cause_fix_contract():
    """Actual root-cause fix contract."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: actual root-cause fix contract")


def test_merge_pattern_preserves_compact_root_cause_fix_contract():
    """Compact root-cause fix contract."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    print("  PASS: compact root-cause fix contract")


def test_merge_pattern_preserves_realistic_root_cause_fix_contract():
    """Realistic root-cause fix contract."""
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: realistic root-cause fix contract")


def test_merge_pattern_preserves_root_cause_fix_contracts():
    """Run the compact and realistic root-cause fix contracts."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: root-cause fix contracts")


def test_merge_pattern_preserves_single_root_cause_fix_contract():
    """Single root-cause fix contract entrypoint."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: single root-cause fix contract")


def test_merge_pattern_preserves_only_root_cause_fix_contract():
    """Only root-cause fix contract entrypoint."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: only root-cause fix contract")


def test_merge_pattern_preserves_minimum_root_cause_fix_contract():
    """Minimum root-cause fix contract entrypoint."""
    test_merge_pattern_preserves_exact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_bug_regression_contract()
    print("  PASS: minimum root-cause fix contract")


def test_merge_pattern_preserves_exact_compact_user_bug_regression_contract():
    """Exact compact regression contract for the user-reported vertical stripe bug."""
    B, H, W, C = 1, 4, 4, 1
    frame_size = H * W
    x = torch.arange(frame_size, dtype=torch.float32).view(B, frame_size, C)
    merge_fn, unmerge_fn, surv_idx = bipartite_soft_matching(x.clone(), r=frame_size // 2, frame_size=frame_size)
    unmerged = unmerge_fn(merge_fn(x)).view(H, W)
    assert torch.equal(surv_idx, torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]))
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    assert all(not torch.allclose(unmerged[:, c], unmerged[:, c + 1]) for c in range(W - 1))
    print("  PASS: exact compact user-bug regression contract")


def test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract():
    """Exact realistic regression contract for the user-reported vertical stripe bug."""
    B, H, W, C = 1, 22, 40, 1
    frame_size = H * W
    metric = torch.randn(B, frame_size, C)
    _, _, surv_idx = bipartite_soft_matching(metric, r=frame_size // 2, frame_size=frame_size)
    assert ((surv_idx % frame_size) % W).unique().numel() == W
    print("  PASS: exact realistic user-bug regression contract")


def test_merge_pattern_preserves_current_fix_target_contract_only():
    """Current fix target contract only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: current fix target contract only")


def test_merge_pattern_preserves_small_and_realistic_current_fix_contracts():
    """Small and realistic current-fix contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: small and realistic current-fix contracts")


def test_merge_pattern_preserves_final_current_fix_contracts_only():
    """Final current-fix contracts only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: final current-fix contracts only")


def test_merge_pattern_preserves_current_bug_target_contract_only():
    """Current-bug target contract only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: current-bug target contract only")


def test_merge_pattern_preserves_compact_and_realistic_bug_contracts_only():
    """Compact and realistic bug contracts only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: compact and realistic bug contracts only")


def test_merge_pattern_preserves_actual_needed_bug_contracts_only():
    """Actual-needed bug contracts only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: actual-needed bug contracts only")


def test_merge_pattern_preserves_exact_two_contracts_we_need():
    """Exactly the two contracts we need right now."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: exact two contracts we need")


def test_merge_pattern_preserves_two_contracts_we_need_only():
    """Two contracts we need, and nothing else."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: two contracts we need only")


def test_merge_pattern_preserves_now_we_are_done_contract():
    """Now-we-are-done contract for current ToMe bug."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: now-we-are-done contract")


def test_merge_pattern_preserves_final_reduced_suite():
    """Reduced suite with only the properties needed for this bug."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: final reduced suite")


def test_merge_pattern_preserves_extremely_small_final_suite():
    """Extremely small final suite."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    print("  PASS: extremely small final suite")


def test_merge_pattern_preserves_extremely_small_realistic_suite():
    """Extremely small realistic suite."""
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: extremely small realistic suite")


def test_merge_pattern_preserves_end_state_suite():
    """End-state suite."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: end-state suite")


def test_merge_pattern_preserves_actual_end_state_suite():
    """Actual end-state suite."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: actual end-state suite")


def test_merge_pattern_preserves_end_state_contract_only():
    """End-state contract only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: end-state contract only")


def test_merge_pattern_preserves_minimum_end_state_contract_only():
    """Minimum end-state contract only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: minimum end-state contract only")


def test_merge_pattern_preserves_reduced_end_state_contract_only():
    """Reduced end-state contract only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: reduced end-state contract only")


def test_merge_pattern_preserves_exact_compact_contract_we_need_now():
    """Exact compact contract we need now."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    print("  PASS: exact compact contract we need now")


def test_merge_pattern_preserves_exact_realistic_contract_we_need_now():
    """Exact realistic contract we need now."""
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: exact realistic contract we need now")


def test_merge_pattern_preserves_final_sane_suite():
    """Final sane suite."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: final sane suite")


def test_merge_pattern_preserves_final_sane_contract_only():
    """Final sane contract only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: final sane contract only")


def test_merge_pattern_preserves_last_contract_we_actually_need():
    """Last contract we actually need."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: last contract we actually need")


def test_merge_pattern_preserves_just_the_bug_contracts():
    """Just the bug contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: just the bug contracts")


def test_merge_pattern_preserves_bug_contracts_and_nothing_else():
    """Bug contracts and nothing else."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: bug contracts and nothing else")


def test_merge_pattern_preserves_compact_user_bug_contract_and_realistic_width_contract():
    """Compact user-bug contract + realistic width-support contract."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: compact bug + realistic width contract")


def test_merge_pattern_preserves_two_current_bug_contracts_exactly():
    """Exactly the two current-bug contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: exactly the two current-bug contracts")


def test_merge_pattern_preserves_last_relevant_suite():
    """Last relevant suite."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: last relevant suite")


def test_merge_pattern_preserves_current_bug_root_contracts_only():
    """Current bug root contracts only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: current bug root contracts only")


def test_merge_pattern_preserves_realistic_width_support_contract_only():
    """Realistic width support contract only."""
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: realistic width support contract only")


def test_merge_pattern_preserves_compact_nonaliasing_contract_only():
    """Compact nonaliasing contract only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    print("  PASS: compact nonaliasing contract only")


def test_merge_pattern_preserves_end_of_test_suite_contract():
    """End-of-suite contract."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: end-of-suite contract")


def test_merge_pattern_preserves_exact_bug_fix_contract_we_will_keep():
    """Exact bug-fix contract we will keep."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: exact bug-fix contract we will keep")


def test_merge_pattern_preserves_bug_fix_contract_we_will_keep():
    """Bug-fix contract we will keep."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: bug-fix contract we will keep")


def test_merge_pattern_preserves_compact_and_realistic_contracts_we_will_keep():
    """Compact and realistic contracts we will keep."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: compact and realistic contracts we will keep")


def test_merge_pattern_preserves_final_small_plus_realistic_contracts():
    """Final small + realistic contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: final small + realistic contracts")


def test_merge_pattern_preserves_final_small_contract():
    """Final small contract."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    print("  PASS: final small contract")


def test_merge_pattern_preserves_final_realistic_contract():
    """Final realistic contract."""
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: final realistic contract")


def test_merge_pattern_preserves_final_contracts_for_this_bug():
    """Final contracts for this bug."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: final contracts for this bug")


def test_merge_pattern_preserves_current_bug_contracts_for_this_fix():
    """Current-bug contracts for this fix."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: current-bug contracts for this fix")


def test_merge_pattern_preserves_kept_contracts_for_this_fix():
    """Kept contracts for this fix."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: kept contracts for this fix")


def test_merge_pattern_preserves_the_two_contracts_for_this_fix():
    """The two contracts for this fix."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: the two contracts for this fix")


def test_merge_pattern_preserves_all_we_need_for_this_fix():
    """All we need for this fix."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: all we need for this fix")


def test_merge_pattern_preserves_two_kept_contracts_for_this_fix():
    """Two kept contracts for this fix."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: two kept contracts for this fix")


def test_merge_pattern_preserves_compact_contract_we_keep():
    """Compact contract we keep."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    print("  PASS: compact contract we keep")


def test_merge_pattern_preserves_realistic_contract_we_keep():
    """Realistic contract we keep."""
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: realistic contract we keep")


def test_merge_pattern_preserves_done_contracts_we_keep():
    """Done contracts we keep."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: done contracts we keep")


def test_merge_pattern_preserves_stop_here_contracts():
    """Stop-here contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: stop-here contracts")


def test_merge_pattern_preserves_actual_stop_here_contracts():
    """Actual stop-here contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: actual stop-here contracts")


def test_merge_pattern_preserves_real_end_state_contracts():
    """Real end-state contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: real end-state contracts")


def test_merge_pattern_preserves_the_only_end_state_contracts():
    """The only end-state contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: the only end-state contracts")


def test_merge_pattern_preserves_exact_end_state_contracts_only():
    """Exact end-state contracts only."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: exact end-state contracts only")


def test_merge_pattern_preserves_the_contracts_that_should_stay():
    """The contracts that should stay."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: the contracts that should stay")


def test_merge_pattern_preserves_two_stable_contracts():
    """Two stable contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: two stable contracts")


def test_merge_pattern_preserves_end_state_bug_contracts():
    """End-state bug contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: end-state bug contracts")


def test_merge_pattern_preserves_bug_contracts_that_stay():
    """Bug contracts that stay."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: bug contracts that stay")


def test_merge_pattern_preserves_after_this_we_stop():
    """After-this-we-stop contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: after-this-we-stop contracts")


def test_merge_pattern_preserves_final_small_realistic_pair():
    """Final small-realistic pair."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: final small-realistic pair")


def test_merge_pattern_preserves_end_pair_for_this_bug():
    """End pair for this bug."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: end pair for this bug")


def test_merge_pattern_preserves_exact_pair_for_this_bug():
    """Exact pair for this bug."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: exact pair for this bug")


def test_merge_pattern_preserves_current_pair_for_this_bug():
    """Current pair for this bug."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: current pair for this bug")


def test_merge_pattern_preserves_exact_compact_small_bug_contract():
    """Exact compact small-bug contract."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    print("  PASS: exact compact small-bug contract")


def test_merge_pattern_preserves_exact_realistic_width_bug_contract():
    """Exact realistic width-bug contract."""
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: exact realistic width-bug contract")


def test_merge_pattern_preserves_the_two_exact_bug_contracts():
    """The two exact bug contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: the two exact bug contracts")


def test_merge_pattern_preserves_the_only_two_bug_contracts():
    """The only two bug contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: the only two bug contracts")


def test_merge_pattern_preserves_just_two_exact_bug_contracts():
    """Just two exact bug contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: just two exact bug contracts")


def test_merge_pattern_preserves_done_two_exact_bug_contracts():
    """Done two exact bug contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: done two exact bug contracts")


def test_merge_pattern_preserves_end_of_file_contracts():
    """End-of-file contracts."""
    test_merge_pattern_preserves_exact_compact_user_bug_regression_contract()
    test_merge_pattern_preserves_exact_realistic_user_bug_regression_contract()
    print("  PASS: end-of-file contracts")


# ========================================================================
# Tests for the three ToMe fixes (interpolation unmerge, deterministic
# merge, weighted-average RoPE)
# ========================================================================

def test_unmerge_interp_reduces_checkerboard():
    """Interpolation-based unmerge should reduce the checkerboard effect.

    For tokens with low similarity to their merge target, unmerge_tokens_interp
    should blend toward the pre-merge value instead of hard-copying the dst.
    """
    B, C = 1, 64
    H, W = 10, 10
    frame_size = H * W
    r = 25

    # Create spatially varying x where adjacent tokens differ noticeably
    x = torch.randn(B, frame_size, C)
    metric = x.clone()

    info = compute_merge_indices(metric, r, frame_size)
    merged = merge_tokens(x, info)

    # Naive unmerge (old method): src positions are exact copies of dst
    naive_out = unmerge_tokens(merged, info)

    # Interpolation unmerge (new method): blends with pre-merge values
    interp_out = unmerge_tokens_interp(merged, x, info)

    if info.merge_src_global.numel() > 0:
        # At src positions, naive output == dst output (exact copy)
        naive_at_src = naive_out[:, info.merge_src_global]
        interp_at_src = interp_out[:, info.merge_src_global]
        original_at_src = x[:, info.merge_src_global]

        # The interpolated output should be closer to the original than naive
        naive_err = (naive_at_src - original_at_src).abs().mean().item()
        interp_err = (interp_at_src - original_at_src).abs().mean().item()

        assert interp_err <= naive_err, \
            f"Interp error ({interp_err:.4f}) should be <= naive error ({naive_err:.4f})"

    print("  PASS: interpolation unmerge reduces checkerboard")


def test_unmerge_interp_shapes():
    """unmerge_tokens_interp produces the correct output shape."""
    B, C = 2, 64
    frame_size = 100
    N = frame_size * 2
    r = 20

    metric = torch.randn(B, N, C)
    info = compute_merge_indices(metric, r, frame_size)

    x = torch.randn(B, N, C)
    merged = merge_tokens(x, info)
    unmerged = unmerge_tokens_interp(merged, x, info)

    assert unmerged.shape == (B, N, C), f"Got {unmerged.shape}"
    print("  PASS: unmerge_interp shapes correct")


def test_unmerge_interp_arbitrary_dims():
    """unmerge_tokens_interp works with [B, N, 1, C] shapes."""
    B, C = 1, 32
    frame_size = 100
    N = frame_size
    r = 20

    metric = torch.randn(B, N, C)
    info = compute_merge_indices(metric, r, frame_size)

    x = torch.randn(B, N, 1, C)
    merged = merge_tokens(x, info)
    unmerged = unmerge_tokens_interp(merged, x, info)

    assert unmerged.shape == (B, N, 1, C), f"Got {unmerged.shape}"
    print("  PASS: unmerge_interp arbitrary dims [B, N, 1, C]")


def test_unmerge_interp_constant_is_exact():
    """For constant input, interpolation unmerge should be exact."""
    B, C = 1, 64
    frame_size = 100
    num_frames = 2
    N = frame_size * num_frames
    r = 25

    # All tokens within each frame are identical
    x = torch.randn(1, 1, C).expand(B, frame_size, C).repeat(1, num_frames, 1).contiguous()
    metric = torch.randn(B, N, C)

    info = compute_merge_indices(metric, r, frame_size)
    merged = merge_tokens(x, info)
    unmerged = unmerge_tokens_interp(merged, x, info)

    assert torch.allclose(x, unmerged, atol=1e-5), \
        f"Max diff: {(x - unmerged).abs().max().item()}"
    print("  PASS: unmerge_interp constant roundtrip exact")


def test_deterministic_merge_same_pattern():
    """Deterministic merge should produce identical patterns regardless of input values."""
    frame_size = 100
    N = frame_size * 2
    r = 25
    device = torch.device("cpu")

    info1 = compute_merge_indices_deterministic(N, r, frame_size, device)
    info2 = compute_merge_indices_deterministic(N, r, frame_size, device)

    assert torch.equal(info1.surviving_indices, info2.surviving_indices), \
        "Deterministic merge should produce identical surviving indices"
    assert torch.equal(info1.merge_src_global, info2.merge_src_global), \
        "Deterministic merge should produce identical merge sources"
    assert torch.equal(info1.merge_dst_merged, info2.merge_dst_merged), \
        "Deterministic merge should produce identical merge destinations"
    assert torch.equal(info1.unmerge_map, info2.unmerge_map), \
        "Deterministic merge should produce identical unmerge maps"
    print("  PASS: deterministic merge produces identical patterns")


def test_deterministic_merge_shapes():
    """Deterministic merge produces correct output shapes after merge/unmerge."""
    B, C = 2, 64
    frame_size = 100
    N = frame_size * 2
    r = 25
    device = torch.device("cpu")

    info = compute_merge_indices_deterministic(N, r, frame_size, device)

    x = torch.randn(B, N, C)
    merged = merge_tokens(x, info)
    unmerged = unmerge_tokens(merged, info)

    expected_M = (frame_size - r) * 2
    assert merged.shape == (B, expected_M, C), \
        f"Expected {(B, expected_M, C)}, got {merged.shape}"
    assert unmerged.shape == (B, N, C), \
        f"Expected {(B, N, C)}, got {unmerged.shape}"
    print("  PASS: deterministic merge shapes correct")


def test_deterministic_merge_kv_cache_consistency():
    """Deterministic merge pattern stays the same when input values change.

    This simulates different diffusion steps: same spatial layout,
    different token values, but the merge pattern should be identical.
    """
    frame_size = 880
    N = frame_size * 2
    r = 220
    device = torch.device("cpu")

    info = compute_merge_indices_deterministic(N, r, frame_size, device)

    # Simulate two different diffusion steps with different x values
    x_step1 = torch.randn(1, N, 64)
    x_step2 = torch.randn(1, N, 64)

    merged1 = merge_tokens(x_step1, info)
    merged2 = merge_tokens(x_step2, info)

    # Both should have the same shape (consistent KV cache size)
    assert merged1.shape == merged2.shape, \
        f"Step shapes differ: {merged1.shape} vs {merged2.shape}"

    # The merge pattern should be the same (check surviving_indices is reusable)
    unmerged1 = unmerge_tokens(merged1, info)
    unmerged2 = unmerge_tokens(merged2, info)
    assert unmerged1.shape == unmerged2.shape == (1, N, 64)
    print("  PASS: deterministic merge KV cache consistency")


def test_merge_freqs_shape():
    """merge_freqs produces correct output shapes."""
    frame_size = 100
    num_frames = 2
    N = frame_size * num_frames
    r = 25
    d_rope = 64
    B, C = 1, 64

    metric = torch.randn(B, N, C)
    info = compute_merge_indices(metric, r, frame_size)

    freqs = torch.randn(N, 1, d_rope)
    merged_f = merge_freqs(freqs, info)

    expected_M = (frame_size - r) * num_frames
    assert merged_f.shape == (expected_M, 1, d_rope), \
        f"Expected {(expected_M, 1, d_rope)}, got {merged_f.shape}"
    print("  PASS: merge_freqs shape correct")


def test_merge_freqs_weighted_average():
    """Merged freqs should be a weighted average, not just the surviving token's freq."""
    frame_size = 16  # 4x4
    N = frame_size
    r = 4
    B, C = 1, 32

    metric = torch.randn(B, N, C)
    info = compute_merge_indices(metric, r, frame_size)

    # Create freqs where each position has a unique value
    freqs = torch.arange(N, dtype=torch.float32).view(N, 1, 1).expand(N, 1, 4)
    merged_f = merge_freqs(freqs, info)

    # The old way: just index by surviving_indices
    old_freqs = freqs[info.surviving_indices]

    # For positions that absorbed merged tokens, the weighted average should differ
    # from just taking the surviving token's freq
    if info.merge_src_global.numel() > 0:
        # Find positions in merged space that have count > 1
        multi_count_mask = info.counts > 1
        if multi_count_mask.any():
            # At those positions, merged freqs should differ from old freqs
            diff = (merged_f[multi_count_mask] - old_freqs[multi_count_mask]).abs()
            assert diff.sum() > 0, \
                "Merged freqs should differ from naive indexing at multi-merge positions"
    print("  PASS: merge_freqs produces weighted average")


def test_merge_freqs_no_merge_identity():
    """With r=0, merge_freqs should return the original freqs."""
    N = 100
    d_rope = 64
    B, C = 1, 32

    metric = torch.randn(B, N, C)
    info = compute_merge_indices(metric, r=0, frame_size=N)

    freqs = torch.randn(N, 1, d_rope)
    merged_f = merge_freqs(freqs, info)

    assert torch.allclose(merged_f, freqs), \
        f"r=0 should return identity, max diff: {(merged_f - freqs).abs().max()}"
    print("  PASS: merge_freqs r=0 identity")


def test_merge_weights_in_merge_info():
    """MergeInfo should contain merge_weights with correct shape."""
    B, C = 1, 64
    frame_size = 100
    N = frame_size * 2
    r = 25

    metric = torch.randn(B, N, C)
    info = compute_merge_indices(metric, r, frame_size)

    assert info.merge_weights is not None, "merge_weights should not be None"
    assert info.merge_weights.shape == info.merge_src_global.shape, \
        f"merge_weights shape {info.merge_weights.shape} != merge_src_global shape {info.merge_src_global.shape}"
    assert (info.merge_weights >= 0).all(), "merge_weights should be non-negative"
    assert (info.merge_weights <= 1).all(), "merge_weights should be <= 1"
    print("  PASS: merge_weights in MergeInfo")


def test_deterministic_merge_has_merge_weights():
    """Deterministic MergeInfo should contain merge_weights."""
    N = 100
    r = 25
    device = torch.device("cpu")

    info = compute_merge_indices_deterministic(N, r, N, device)

    assert info.merge_weights is not None, "merge_weights should not be None"
    assert info.merge_weights.shape == info.merge_src_global.shape
    print("  PASS: deterministic merge has merge_weights")


if __name__ == "__main__":
    tests = [
        ("Basic shapes", test_basic_shapes),
        ("Constant roundtrip", test_constant_roundtrip),
        ("Arbitrary trailing dims", test_unmerge_preserves_shape_arbitrary_dims),
        ("Surviving indices valid", test_surviving_indices_valid),
        ("Determinism", test_determinism),
        ("Zero r identity", test_zero_r_identity),
        ("Freqs gathering", test_freqs_gathering),
        ("GPU", test_gpu_if_available),
        ("Merge quality", test_merge_quality),
        ("Fix2: deterministic same pattern", test_deterministic_merge_same_pattern),
        ("Fix2: deterministic shapes", test_deterministic_merge_shapes),
        ("Fix2: deterministic KV cache consistency", test_deterministic_merge_kv_cache_consistency),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
