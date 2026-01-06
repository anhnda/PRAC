"""
FAST INT4 Weight-Only Quantization on GQA Attention with ReFlip Strategy

Vectorized version of quantize_qkv.py for 10-100x speedup.

Key optimizations:
1. Vectorized attention score computation for all heads
2. Batch processing of moderate dimension selection
3. Vectorized flip impact calculation (removes hidden_dim loop)
4. Optimized cumulative sum operations
5. Batch dequantization

Usage:
    python fast_quantize_qkv.py [--critical-dim-pct 0.15] [--max-flip-pct 0.05]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils_qkv import (
    find_knee_point,
    compute_dynamic_outlier_threshold,
    quantize_weight_groupwise_int4,
    compute_quantization_error
)

sns.set_style("whitegrid")


# Import the heuristic flip function (not vectorized - already optimized)
from quantize_qkv import quantize_weight_groupwise_int4_with_flip


def quantize_qkv_reflip_fast(Wq, Wk, X, Q_orig_all, Q_heuristic_all,
                              Wq_heuristic, Wk_heuristic, K_heuristic,
                              Wq_int_heuristic, Wq_scales_heuristic, Wq_zp_heuristic,
                              critical_dim_pct=0.1, knee_tolerance=0.0,
                              group_size=128, max_flip_pct=0.1,
                              correction_scale=1.0, debug=False):
    """
    FAST vectorized ReFlip implementation.

    Same functionality as quantize_qkv_reflip but 10-100x faster through vectorization.
    """
    num_heads = Wq.shape[0]
    head_dim = Wq.shape[1]
    hidden_dim = Wq.shape[2]

    # Start with heuristic-quantized weights
    Wq_int_reflip = Wq_int_heuristic.copy()

    # Compute K_orig for reference
    K_orig = X @ Wk.T  # [head_dim]

    # ========== VECTORIZED: Compute attention score errors for ALL heads at once ==========
    # Instead of: for head_idx in range(num_heads): score = Q @ K
    # Vectorized: scores = Q_all @ K  [num_heads] = [num_heads, head_dim] @ [head_dim]

    scores_orig = Q_orig_all @ K_orig  # [num_heads]
    scores_heuristic = Q_heuristic_all @ K_heuristic  # [num_heads]
    score_errors = scores_orig - scores_heuristic  # [num_heads]

    if debug:
        for head_idx in range(num_heads):
            print(f"\nHead {head_idx}:")
            print(f"  Score original:    {scores_orig[head_idx]:.6f}")
            print(f"  Score heuristic:   {scores_heuristic[head_idx]:.6f}")
            print(f"  Score error:       {score_errors[head_idx]:.6f}")

    # ========== Identify moderate dimensions (keep loop - knee detection not easily vectorizable) ==========
    all_moderate_dims = []
    all_dim_corrections = []

    for head_idx in range(num_heads):
        Q_orig = Q_orig_all[head_idx]
        score_error = score_errors[head_idx]

        # Sort by |Q_orig| magnitude (descending)
        sorted_indices_desc = np.argsort(np.abs(Q_orig))[::-1]
        sorted_magnitudes = np.abs(Q_orig[sorted_indices_desc])

        # Apply Kneedle to find knee point
        first_half = sorted_magnitudes[:head_dim // 2]
        knee_idx = find_knee_point(first_half[::-1], tolerance_offset=knee_tolerance)
        knee_idx = len(first_half) - knee_idx - 1

        # Select moderate dimensions
        num_moderate = max(int(critical_dim_pct * head_dim), 1)
        moderate_start = knee_idx
        moderate_end = min(moderate_start + num_moderate, head_dim)
        moderate_indices = sorted_indices_desc[moderate_start:moderate_end]

        all_moderate_dims.append(moderate_indices)

        # Redistribute scalar error proportionally
        if len(moderate_indices) > 0:
            Q_moderate = Q_orig[moderate_indices]
            Q_moderate_abs = np.abs(Q_moderate)
            Q_moderate_sum = Q_moderate_abs.sum()

            if Q_moderate_sum > 1e-10:
                dim_corrections = score_error * (Q_moderate_abs / Q_moderate_sum)
            else:
                dim_corrections = np.full(len(moderate_indices), score_error / len(moderate_indices))
        else:
            dim_corrections = np.array([])

        all_dim_corrections.append(dim_corrections)

        if debug:
            print(f"  Moderate dims: {len(moderate_indices)} (from {moderate_start} to {moderate_end})")
            print(f"  Sum of corrections: {dim_corrections.sum():.6f} (should ≈ {score_error:.6f})")

    # ========== VECTORIZED: Apply integer flips ==========
    # Prepare expanded scales/zp for ALL heads and dims at once
    n_groups = hidden_dim // group_size
    total_flips = 0

    for head_idx in range(num_heads):
        moderate_indices = all_moderate_dims[head_idx]
        dim_corrections = all_dim_corrections[head_idx]
        score_error = score_errors[head_idx]

        if len(moderate_indices) == 0:
            continue

        # Batch process all moderate dimensions for this head
        num_moderate = len(moderate_indices)

        # VECTORIZED: Expand scales/zp for all moderate dimensions
        scales_batch = Wq_scales_heuristic[head_idx, moderate_indices, :]  # [num_moderate, n_groups]
        zp_batch = Wq_zp_heuristic[head_idx, moderate_indices, :]  # [num_moderate, n_groups]

        # Expand to full dimension: [num_moderate, hidden_dim]
        scales_expanded = np.repeat(scales_batch, group_size, axis=1)[:, :hidden_dim]
        zp_expanded = np.repeat(zp_batch, group_size, axis=1)[:, :hidden_dim]

        # VECTORIZED: Compute current Q values for all moderate dimensions
        W_current_batch = (Wq_int_reflip[head_idx, moderate_indices, :] - zp_expanded) * scales_expanded
        Q_current_batch = W_current_batch @ X  # [num_moderate]

        # VECTORIZED: Compute targets for all moderate dimensions
        K_values = K_orig[moderate_indices]  # [num_moderate]

        # Skip dimensions with K too small
        valid_K_mask = np.abs(K_values) >= 1e-10
        if not valid_K_mask.any():
            continue

        delta_Q_targets = np.where(valid_K_mask, dim_corrections / K_values, 0.0)  # [num_moderate]
        Q_targets = Q_current_batch + delta_Q_targets  # [num_moderate]
        error_currents = Q_current_batch - Q_targets  # [num_moderate]

        # Determine flip directions for all dimensions
        delta_score_needed = score_error
        flip_directions = np.where(
            delta_score_needed > 0,
            np.where(K_values > 0, 1, -1),
            np.where(K_values > 0, -1, 1)
        )  # [num_moderate]

        # Process each moderate dimension (still need loop for greedy selection)
        for i, dim_idx in enumerate(moderate_indices):
            if not valid_K_mask[i] or abs(dim_corrections[i]) < 1e-10:
                continue

            flip_direction = flip_directions[i]
            scales_exp = scales_expanded[i]
            zp_exp = zp_expanded[i]
            error_current = error_currents[i]

            # VECTORIZED: Calculate impact of ALL flips at once
            current_qvals = Wq_int_reflip[head_idx, dim_idx, :]  # [hidden_dim]
            new_qvals = current_qvals + flip_direction  # [hidden_dim]

            # Vectorized validity check
            valid_flips = (new_qvals >= 0) & (new_qvals <= 15)  # [hidden_dim]

            # Vectorized flip impacts
            flip_impacts = flip_direction * scales_exp * X  # [hidden_dim]

            # Filter to beneficial flips
            target_sign = -np.sign(error_current)
            beneficial_flips = (np.sign(flip_impacts) == target_sign) & valid_flips

            if not beneficial_flips.any():
                continue

            # Sort beneficial flips
            flip_scores = np.abs(flip_impacts) * beneficial_flips
            sorted_indices = np.argsort(-flip_scores)  # Descending

            # VECTORIZED: Greedy selection with cumsum
            beneficial_sorted = beneficial_flips[sorted_indices]
            impacts_sorted = flip_impacts[sorted_indices]

            # Build cumsum only for beneficial flips
            impacts_beneficial = impacts_sorted * beneficial_sorted
            cumsum_impacts = np.concatenate([[0], np.cumsum(impacts_beneficial)])

            # Find optimal K
            residuals = np.abs(error_current + cumsum_impacts)
            best_k = np.argmin(residuals)

            # Cap at max_flip_pct
            max_flips = int(hidden_dim * max_flip_pct)
            best_k = min(best_k, max_flips)

            # Apply optimal flips
            if best_k > 0:
                flip_indices = sorted_indices[:best_k][beneficial_sorted[:best_k]]
                Wq_int_reflip[head_idx, dim_idx, flip_indices] += flip_direction
                total_flips += len(flip_indices)

            if debug:
                Q_after = Q_current_batch[i] + cumsum_impacts[best_k]
                error_after = Q_after - Q_targets[i]
                print(f"    Dim {dim_idx}: Q_current={Q_current_batch[i]:.6f}, "
                      f"Q_target={Q_targets[i]:.6f}, "
                      f"error_reduction={abs(error_current) - abs(error_after):.6f}, "
                      f"flips={len(flip_indices) if best_k > 0 else 0}")

    # ========== VECTORIZED: Batch dequantize all modified weights ==========
    Wq_reflip = np.zeros_like(Wq_int_reflip, dtype=np.float32)

    # Vectorized dequantization for all heads and dimensions
    for head_idx in range(num_heads):
        # Expand scales/zp for entire head: [head_dim, hidden_dim]
        scales_expanded = np.repeat(Wq_scales_heuristic[head_idx], group_size, axis=1)[:, :hidden_dim]
        zp_expanded = np.repeat(Wq_zp_heuristic[head_idx], group_size, axis=1)[:, :hidden_dim]

        # Vectorized dequantization: W = (W_int - zp) * scale
        Wq_reflip[head_idx] = (Wq_int_reflip[head_idx] - zp_expanded) * scales_expanded

    # Return results
    Wq_quant_reflip = Wq_reflip
    Wq_scales = Wq_scales_heuristic
    Wq_zp = Wq_zp_heuristic
    Wq_int = Wq_int_reflip

    Wk_quant_reflip = Wk_heuristic
    Wk_scales = np.ones((head_dim, hidden_dim // group_size))
    Wk_zp = np.zeros((head_dim, hidden_dim // group_size))
    Wk_int = Wk_heuristic

    reflip_stats = {
        'moderate_dims_per_head': [len(dims) for dims in all_moderate_dims],
        'total_moderate_dims': sum(len(dims) for dims in all_moderate_dims),
        'moderate_dim_pct': critical_dim_pct,
        'knee_tolerance': knee_tolerance,
        'score_errors': score_errors.tolist(),
        'total_flips': total_flips,
        'flip_rate_pct': (total_flips / (num_heads * head_dim * hidden_dim) * 100)
    }

    return (Wq_quant_reflip, Wq_scales, Wq_zp, Wq_int,
            Wk_quant_reflip, Wk_scales, Wk_zp, Wk_int,
            reflip_stats)


# Import main function from original (use same logic, just replace reflip function)
from quantize_qkv import main as main_original

def main():
    """Main function - same as original but uses fast_quantize_qkv_reflip"""
    import sys
    # Monkey-patch the reflip function to use fast version
    import quantize_qkv
    quantize_qkv.quantize_qkv_reflip = quantize_qkv_reflip_fast

    # Call original main which will now use fast version
    main_original()


if __name__ == '__main__':
    main()
