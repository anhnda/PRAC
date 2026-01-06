"""
INT4 Weight-Only Quantization on GQA Attention with ReFlip Strategy

This script compares three quantization strategies:
1. Nearest Rounding (baseline)
2. Heuristic Flip Correction (global greedy)
3. ReFlip (new): Targeted error correction on critical head dimensions

ReFlip Strategy:
1. Apply initial heuristic quantization
2. Use Kneedle algorithm to identify critical head dimensions (based on |Q_orig|)
3. Select top ~5% critical dimensions per head (configurable)
4. Compute target error correction = -current_error for critical dimensions
5. Redistribute correction proportionally to input magnitudes
6. Apply second heuristic flip to reduce critical dimension errors

Usage:
    python quantize_qkv.py [--critical-dim-pct 0.05] [--knee-tolerance 0.0]
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


def quantize_weight_groupwise_int4_with_flip(W, activation_means, group_size=128,
                                              knee_tolerance=0.0, max_flip_pct=0.01, debug=False):
    """
    Quantize weights to INT4 with heuristic flip correction (from awq_js_xl.py).

    This implements the global greedy rounding correction that flips quantization
    directions to minimize the overall error in X @ W computation.

    Uses DYNAMIC outlier detection via Kneedle algorithm instead of fixed percentile.

    Args:
        W: Weight matrix of shape [..., in_features]
        activation_means: Channel-wise activation means, shape [in_features]
        group_size: Size of each quantization group (default: 128)
        knee_tolerance: Tolerance offset for Kneedle algorithm (default: 0.0)
        max_flip_pct: Max percentage of weights that can be flipped per row (default: 0.01 = 1%)
        debug: Print debug information (default: False)

    Returns:
        W_quant: Dequantized weights (same shape as W)
        scales: Per-group scales
        zp: Per-group zero points
        W_int: Integer weights
        flip_stats: Statistics about flips (includes 'outlier_percent' from dynamic detection)
    """
    original_shape = W.shape

    # Flatten to 2D if needed
    if W.ndim > 2:
        W_flat = W.reshape(-1, W.shape[-1])
    else:
        W_flat = W.copy()

    out_features, in_features = W_flat.shape
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size

    # Pad weights if needed
    if padded_in > in_features:
        W_padded = np.zeros((out_features, padded_in), dtype=W.dtype)
        W_padded[:, :in_features] = W_flat
        act_padded = np.zeros(padded_in, dtype=activation_means.dtype)
        act_padded[:in_features] = activation_means
    else:
        W_padded = W_flat
        act_padded = activation_means

    # Reshape to groups
    W_grouped = W_padded.reshape(out_features, n_groups, group_size)

    # Asymmetric quantization [0, 15]
    w_min = W_grouped.min(axis=2, keepdims=True)
    w_max = W_grouped.max(axis=2, keepdims=True)
    max_int = 15

    scale = (w_max - w_min) / max_int
    scale = np.maximum(scale, 1e-8)
    zp = np.round(-w_min / scale).clip(0, max_int)

    # Expand to full size
    scale_flat = np.repeat(scale, group_size, axis=2).reshape(out_features, padded_in)
    zp_flat = np.repeat(zp, group_size, axis=2).reshape(out_features, padded_in)

    # Initial quantization (nearest rounding)
    W_div = W_padded / scale_flat
    W_int = np.round(W_div + zp_flat).clip(0, max_int)
    W_quant = (W_int - zp_flat) * scale_flat

    # --- HEURISTIC FLIP CORRECTION ---

    # A. Calculate current error (before flipping)
    W_diff = W_padded - W_quant
    current_error = (W_diff * act_padded[np.newaxis, :]).sum(axis=1)  # [out_features]

    # Store error before flipping for statistics
    error_before_flip = np.abs(current_error).sum()  # Total absolute error before flipping

    # B. Identify flip directions
    flip_dir = np.sign(W_div + zp_flat - W_int)
    flip_dir[flip_dir == 0] = 1.0

    # C. Calculate flip impacts
    flip_impacts = act_padded[np.newaxis, :] * flip_dir * scale_flat  # [out, in]

    # D. Validity masks
    target_sign = np.sign(current_error)[:, np.newaxis]
    valid_mask = (np.sign(flip_impacts) == target_sign)

    # Check if flips are in range
    w_int_proposed = W_int + flip_dir
    in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
    valid_mask = valid_mask & in_range

    # DYNAMIC outlier masking using Kneedle algorithm
    outlier_threshold, outlier_percent = compute_dynamic_outlier_threshold(
        act_padded, knee_tolerance=knee_tolerance, debug=debug
    )
    is_outlier = np.abs(act_padded) > outlier_threshold
    valid_mask = valid_mask & (~is_outlier)[np.newaxis, :]

    if debug:
        print(f"    DEBUG: Dynamic outlier detection found {outlier_percent*100:.2f}% outliers")
        print(f"    DEBUG: Outlier threshold: {outlier_threshold:.6f}")

    # E. Sorting & Optimization
    rounding_costs = np.abs(W_div + zp_flat - W_int)
    rounding_costs_masked = rounding_costs.copy()
    rounding_costs_masked[~valid_mask] = -1.0

    # Sort by rounding cost (descending)
    sorted_indices = np.argsort(-rounding_costs_masked, axis=1)  # Descending
    sorted_impacts = np.take_along_axis(flip_impacts, sorted_indices, axis=1)
    sorted_validity = np.take_along_axis(valid_mask.astype(float), sorted_indices, axis=1)
    sorted_impacts = sorted_impacts * sorted_validity

    # Cumulative sum of impacts
    cumsum_impacts = np.cumsum(sorted_impacts, axis=1)
    residuals = np.abs(current_error[:, np.newaxis] - cumsum_impacts)
    error_unsqueezed = np.abs(current_error)[:, np.newaxis]
    all_residuals = np.concatenate([error_unsqueezed, residuals], axis=1)
    best_k = np.argmin(all_residuals, axis=1)

    # F. Apply flips with max flip constraint
    idx_range = np.arange(padded_in)[np.newaxis, :]
    flip_mask_sorted = idx_range < best_k[:, np.newaxis]
    final_flips_sorted = flip_mask_sorted & (sorted_validity > 0)

    # Constraint: limit flips per row
    max_flips_per_row = int(max_flip_pct * in_features)
    cumsum_flips = np.cumsum(final_flips_sorted.astype(int), axis=1)
    within_limit = cumsum_flips <= max_flips_per_row

    # Get flip directions
    sorted_flip_dir = np.take_along_axis(flip_dir, sorted_indices, axis=1)
    sorted_flip_dir[~(final_flips_sorted & within_limit)] = 0.0

    # Apply flips
    W_int_flipped = W_int.copy()
    np.put_along_axis(W_int_flipped, sorted_indices,
                      np.take_along_axis(W_int, sorted_indices, axis=1) + sorted_flip_dir, axis=1)
    W_int_flipped = W_int_flipped.clip(0, max_int)

    # Dequantize
    W_quant_flipped = (W_int_flipped - zp_flat) * scale_flat

    # Calculate error after flipping (for statistics)
    W_diff_after = W_padded - W_quant_flipped
    current_error_after = (W_diff_after * act_padded[np.newaxis, :]).sum(axis=1)  # [out_features]
    error_after_flip = np.abs(current_error_after).sum()  # Total absolute error after flipping

    # Compute error reduction
    error_reduction = error_before_flip - error_after_flip
    error_reduction_pct = (error_reduction / (error_before_flip + 1e-10)) * 100

    # Remove padding
    if padded_in > in_features:
        W_quant_flipped = W_quant_flipped[:, :in_features]
        W_int_flipped = W_int_flipped[:, :in_features]

    # Reshape back
    W_quant_final = W_quant_flipped.reshape(original_shape)
    W_int_final = W_int_flipped.reshape(original_shape)

    # Reshape scales and zp
    scales_flat = scale.reshape(out_features, n_groups)
    zp_flat_out = zp.reshape(out_features, n_groups)

    if len(original_shape) == 3:
        scales_out = scales_flat.reshape(original_shape[0], original_shape[1], n_groups)
        zp_out = zp_flat_out.reshape(original_shape[0], original_shape[1], n_groups)
    else:
        scales_out = scales_flat
        zp_out = zp_flat_out

    # Flip statistics
    total_flips = (final_flips_sorted & within_limit).sum()
    flips_per_row = (final_flips_sorted & within_limit).sum(axis=1)

    flip_stats = {
        'total_flips': int(total_flips),
        'flips_per_row_mean': float(flips_per_row.mean()),
        'flips_per_row_max': int(flips_per_row.max()),
        'flips_per_row_min': int(flips_per_row.min()),
        'flip_rate_pct': float(total_flips / (out_features * in_features) * 100),
        'outlier_percent': float(outlier_percent),  # From dynamic Kneedle detection
        # Error correction statistics
        'error_before_flip': float(error_before_flip),
        'error_after_flip': float(error_after_flip),
        'error_reduction': float(error_reduction),
        'error_reduction_pct': float(error_reduction_pct)
    }

    return W_quant_final, scales_out, zp_out, W_int_final, flip_stats


def quantize_qkv_reflip(Wq, Wk, X, Q_orig_all, Q_heuristic_all,
                         Wq_heuristic, Wk_heuristic, K_heuristic,
                         Wq_int_heuristic, Wq_scales_heuristic, Wq_zp_heuristic,
                         critical_dim_pct=0.15, knee_tolerance=0.0,
                         group_size=128, max_flip_pct=0.05,
                         correction_scale=1.0, debug=False):
    """
    ReFlip: Targeted error correction on attention scores using discrete integer flips.

    Strategy:
    1. Compute attention score error per head: score_error = (Q·K)_orig - (Q·K)_quant (4 scalars)
    2. Identify moderate dimensions using Kneedle on sorted |Q_orig| (start from knee, take next 5%)
    3. Redistribute scalar error to moderate dimensions proportionally to Q values
    4. Apply DISCRETE INTEGER FLIPS (±1) to quantized INT4 weights based on correction direction

    Args:
        Wq: Original query weights [num_heads, head_dim, hidden_dim] = [4, 128, 4096]
        Wk: Original key weights [head_dim, hidden_dim] = [128, 4096]
        X: Input activation vector [hidden_dim] = [4096]
        Q_orig_all: Original Q vectors for all heads [num_heads, head_dim]
        Q_heuristic_all: Heuristic quantized Q vectors [num_heads, head_dim]
        Wq_heuristic: Heuristic quantized Wq weights (dequantized, starting point)
        Wk_heuristic: Heuristic quantized Wk weights (starting point)
        K_heuristic: Heuristic quantized K vector [head_dim]
        Wq_int_heuristic: Integer quantized Wq [num_heads, head_dim, hidden_dim] in [0,15]
        Wq_scales_heuristic: Quantization scales [num_heads, head_dim, n_groups]
        Wq_zp_heuristic: Zero points [num_heads, head_dim, n_groups]
        critical_dim_pct: Percentage of moderate dims to select (default: 0.05 = 5%)
        knee_tolerance: Tolerance for Kneedle algorithm (default: 0.0)
        group_size: Quantization group size (default: 128)
        max_flip_pct: Max percentage of weights to flip per dimension (default: 0.05 = 5%)
        correction_scale: Controls number of flips (default: 1.0)
        debug: Print debug information

    Returns:
        Wq_quant_reflip: ReFlip quantized Wq weights (dequantized)
        Wk_quant_reflip: ReFlip quantized Wk weights (same as heuristic)
        flip_stats_reflip: Statistics about the ReFlip correction
    """
    num_heads = Wq.shape[0]
    head_dim = Wq.shape[1]
    hidden_dim = Wq.shape[2]

    # Start with heuristic-quantized weights
    Wq_reflip = Wq_heuristic.copy()

    # Compute K_orig for reference
    K_orig = X @ Wk.T  # [head_dim]

    all_moderate_dims = []
    all_dim_corrections = []
    all_score_errors = []

    # Step 1: Compute attention score errors (scalars per head)
    for head_idx in range(num_heads):
        Q_orig = Q_orig_all[head_idx]
        Q_heuristic = Q_heuristic_all[head_idx]

        # Compute attention scores (scalars)
        score_orig = Q_orig @ K_orig
        score_heuristic = Q_heuristic @ K_heuristic

        # Scalar error to correct
        score_error = score_orig - score_heuristic

        all_score_errors.append(score_error)

        if debug:
            print(f"\nHead {head_idx}:")
            print(f"  Score original:    {score_orig:.6f}")
            print(f"  Score heuristic:   {score_heuristic:.6f}")
            print(f"  Score error:       {score_error:.6f}")

    # Step 2: Identify moderate dimensions and redistribute error
    for head_idx in range(num_heads):
        Q_orig = Q_orig_all[head_idx]
        score_error = all_score_errors[head_idx]

        # Sort by |Q_orig| magnitude (descending)
        sorted_indices_desc = np.argsort(np.abs(Q_orig))[::-1]
        sorted_magnitudes = np.abs(Q_orig[sorted_indices_desc])

        # Apply Kneedle to find knee point (transition from high to moderate)
        # Use first half to find the knee
        first_half = sorted_magnitudes[:head_dim // 2]
        knee_idx = find_knee_point(first_half[::-1], tolerance_offset=knee_tolerance)
        knee_idx = len(first_half) - knee_idx - 1  # Convert back to descending index

        # Select moderate dimensions: starting from knee, take next critical_dim_pct %
        # These are dimensions after the knee (moderate importance, not too high, not too low)
        num_moderate = max(int(critical_dim_pct * head_dim), 1)
        moderate_start = knee_idx
        moderate_end = min(moderate_start + num_moderate, head_dim)
        moderate_indices = sorted_indices_desc[moderate_start:moderate_end]

        all_moderate_dims.append(moderate_indices)

        # Redistribute scalar error to moderate dimensions proportionally to Q values
        if len(moderate_indices) > 0:
            Q_moderate = Q_orig[moderate_indices]  # Q values for moderate dimensions
            Q_moderate_abs = np.abs(Q_moderate)
            Q_moderate_sum = Q_moderate_abs.sum()

            if Q_moderate_sum > 1e-10:
                # Proportional distribution: correction[i] = score_error * (|Q[i]| / sum(|Q[moderate]|))
                dim_corrections = score_error * (Q_moderate_abs / Q_moderate_sum)
            else:
                # Uniform distribution if all values are near zero
                dim_corrections = np.full(len(moderate_indices), score_error / len(moderate_indices))
        else:
            dim_corrections = np.array([])

        all_dim_corrections.append(dim_corrections)

        if debug:
            print(f"  Knee index: {knee_idx}/{head_dim}, magnitude: {sorted_magnitudes[knee_idx]:.4f}")
            print(f"  Moderate dims: {len(moderate_indices)} (from {moderate_start} to {moderate_end})")
            print(f"  Moderate indices: {moderate_indices[:5]}...")
            print(f"  Dim corrections (first 5): {dim_corrections[:5]}")
            print(f"  Sum of corrections: {dim_corrections.sum():.6f} (should ≈ {score_error:.6f})")

    # Step 3: Apply DISCRETE INTEGER FLIPS (±1) to correct Q dimensions
    # Goal: Reduce attention score error = (Q·K)_orig - (Q·K)_heuristic
    # Strategy:
    #   - Flip direction: sign(-score_error) * sign(K[dim_idx])
    #     Because: score = Σ(Q[i] * K[i]), to change score we need:
    #     - If need to increase score and K[i]>0: increase Q[i] (+1 flip)
    #     - If need to increase score and K[i]<0: decrease Q[i] (-1 flip)
    #   - Flip selection: Choose weights with highest |X| impact
    #   - Flip count: Proportional to |correction| * correction_scale

    # Copy integer representation to modify
    Wq_int_reflip = Wq_int_heuristic.copy()

    # Prepare scales/zp expanded to full dimension for dequantization
    n_groups = hidden_dim // group_size

    X_abs = np.abs(X)
    total_flips = 0

    for head_idx in range(num_heads):
        moderate_indices = all_moderate_dims[head_idx]
        dim_corrections = all_dim_corrections[head_idx]
        score_error = all_score_errors[head_idx]

        if len(moderate_indices) == 0:
            continue

        # Compute K_orig for this analysis (needed for flip direction)
        # K_orig was already computed earlier, but we need it here
        # For each moderate dimension, apply integer flips
        for i, (dim_idx, correction) in enumerate(zip(moderate_indices, dim_corrections)):
            if abs(correction) < 1e-10:  # Skip negligible corrections
                continue

            # CRITICAL: Flip direction depends on BOTH error sign AND K sign
            # error = score_orig - score_heuristic
            # To correct: we need delta_score = score_orig - score_heuristic = error
            # Since score = Q @ K, we need: delta_Q[i] * K[i] to contribute to delta_score
            delta_score_needed = score_error  # NOT -score_error!
            K_value = K_orig[dim_idx]  # K value at this dimension

            # Determine flip direction based on desired score change and K sign
            if delta_score_needed > 0:  # Need to increase score
                flip_direction = 1 if K_value > 0 else -1
            else:  # Need to decrease score (delta_score_needed < 0)
                flip_direction = -1 if K_value > 0 else 1

            # GREEDY FLIP SELECTION (like AWQ heuristic in awq_js_xl.py)
            # Goal: Change Q[dim_idx] by the amount needed to correct this dimension's portion of total error

            # Current Q value at this dimension (using current quantized weights)
            scales_row = Wq_scales_heuristic[head_idx, dim_idx, :]
            zp_row = Wq_zp_heuristic[head_idx, dim_idx, :]
            scales_expanded = np.repeat(scales_row, group_size)[:hidden_dim]
            zp_expanded = np.repeat(zp_row, group_size)[:hidden_dim]

            W_current = (Wq_int_reflip[head_idx, dim_idx, :] - zp_expanded) * scales_expanded
            Q_current = X @ W_current  # Current Q[dim_idx]

            # Target Q value: correct this dimension's portion of total score error
            # dim_corrections[i] is the change in (Q[i] * K[i]) we want
            # So: delta_Q[i] * K[i] = dim_corrections[i]
            # Therefore: delta_Q[i] = dim_corrections[i] / K[i]
            if abs(K_value) < 1e-10:
                continue  # Skip if K is too small (division by zero)

            delta_Q_target = correction / K_value
            Q_target = Q_current + delta_Q_target

            # Current error from target
            error_current = Q_current - Q_target

            # Calculate impact of each potential flip
            flip_impacts = np.zeros(hidden_dim)
            valid_flips = np.zeros(hidden_dim, dtype=bool)

            for j in range(hidden_dim):
                current_qval = Wq_int_reflip[head_idx, dim_idx, j]
                new_qval = current_qval + flip_direction

                # Check if flip is valid
                if 0 <= new_qval <= 15:
                    valid_flips[j] = True
                    # Impact of this flip on Q[dim_idx]
                    delta_Q = flip_direction * scales_expanded[j] * X[j]
                    # Impact on error from target (error = Q_current - Q_target)
                    # We want to reduce |error|, so impact should move Q_current toward Q_target
                    flip_impacts[j] = delta_Q  # Direct impact on Q

            # Sort by impact that reduces error (greedy selection)
            # We want flips that move error_current toward 0
            target_sign = -np.sign(error_current)
            beneficial_flips = (np.sign(flip_impacts) == target_sign) & valid_flips

            if not beneficial_flips.any():
                continue  # No beneficial flips available

            # Sort beneficial flips by magnitude of impact (descending)
            flip_scores = np.abs(flip_impacts) * beneficial_flips
            sorted_indices = np.argsort(-flip_scores)  # Descending

            # Greedy selection: find optimal K flips that minimize residual error
            cumsum_impacts = np.zeros(hidden_dim + 1)
            for k in range(1, hidden_dim + 1):
                if beneficial_flips[sorted_indices[k-1]]:
                    cumsum_impacts[k] = cumsum_impacts[k-1] + flip_impacts[sorted_indices[k-1]]

            # Find K that minimizes |error_current + cumsum_impacts[K]|
            residuals = np.abs(error_current + cumsum_impacts)
            best_k = np.argmin(residuals)

            # Cap at max_flip_pct
            max_flips = int(hidden_dim * max_flip_pct)
            best_k = min(best_k, max_flips)

            # Apply the optimal flips
            flips_applied = 0
            for k in range(best_k):
                j = sorted_indices[k]
                if beneficial_flips[j]:
                    Wq_int_reflip[head_idx, dim_idx, j] += flip_direction
                    flips_applied += 1

            total_flips += flips_applied

            if debug:
                Q_after = Q_current + cumsum_impacts[best_k]
                error_after = Q_after - Q_target
                print(f"    Dim {dim_idx}: Q_current={Q_current:.6f}, "
                      f"Q_target={Q_target:.6f}, "
                      f"Q_after={Q_after:.6f}, "
                      f"error_reduction={abs(error_current) - abs(error_after):.6f}, "
                      f"flips={flips_applied} (optimal={best_k})")

    # Dequantize the modified integer weights
    # Expand scales and zero points to match weight dimensions
    Wq_reflip = np.zeros_like(Wq_int_reflip, dtype=np.float32)

    for head_idx in range(num_heads):
        for dim_idx in range(head_dim):
            # Get scales and zero points for this dimension
            scales_row = Wq_scales_heuristic[head_idx, dim_idx, :]  # [n_groups]
            zp_row = Wq_zp_heuristic[head_idx, dim_idx, :]  # [n_groups]

            # Expand to full dimension (repeat each group's scale/zp)
            scales_expanded = np.repeat(scales_row, group_size)[:hidden_dim]
            zp_expanded = np.repeat(zp_row, group_size)[:hidden_dim]

            # Dequantize: W = (W_int - zp) * scale
            Wq_reflip[head_idx, dim_idx, :] = (Wq_int_reflip[head_idx, dim_idx, :] - zp_expanded) * scales_expanded

    # Return refined weights
    Wq_quant_reflip = Wq_reflip

    # Return actual scales/zp and integer representation
    Wq_scales = Wq_scales_heuristic
    Wq_zp = Wq_zp_heuristic
    Wq_int = Wq_int_reflip

    # Wk remains the same
    Wk_quant_reflip = Wk_heuristic
    Wk_scales = np.ones((head_dim, hidden_dim // group_size))
    Wk_zp = np.zeros((head_dim, hidden_dim // group_size))
    Wk_int = Wk_heuristic

    reflip_stats = {
        'moderate_dims_per_head': [len(dims) for dims in all_moderate_dims],
        'total_moderate_dims': sum(len(dims) for dims in all_moderate_dims),
        'moderate_dim_pct': critical_dim_pct,
        'knee_tolerance': knee_tolerance,
        'score_errors': all_score_errors,  # Store for analysis
        'total_flips': total_flips,  # Number of integer flips applied
        'flip_rate_pct': (total_flips / (num_heads * head_dim * hidden_dim) * 100)
    }

    return (Wq_quant_reflip, Wq_scales, Wq_zp, Wq_int,
            Wk_quant_reflip, Wk_scales, Wk_zp, Wk_int,
            reflip_stats)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='INT4 Quantization with ReFlip Strategy')
    parser.add_argument('--critical-dim-pct', type=float, default=0.15,
                        help='Percentage of head dimensions to protect in ReFlip (default: 0.15 = 15%%)')
    parser.add_argument('--knee-tolerance', type=float, default=0.0,
                        help='Tolerance offset for Kneedle algorithm (default: 0.0)')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Quantization group size (default: 128)')
    parser.add_argument('--max-flip-pct', type=float, default=0.05,
                        help='Max flip percentage for ReFlip (default: 0.05 = 5%%)')
    parser.add_argument('--correction-scale', type=float, default=1.0,
                        help='Error correction scaling factor for ReFlip (default: 1.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    args = parser.parse_args()

    print("="*70)
    print("INT4 Weight Quantization for GQA Attention (Group 0)")
    print("Comparing: Nearest | Heuristic | ReFlip")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Critical dim %%: {args.critical_dim_pct*100:.1f}%% (increased from 5%% for more aggressive correction)")
    print(f"  Knee tolerance: {args.knee_tolerance}")
    print(f"  Group size: {args.group_size}")
    print(f"  Max flip %% (ReFlip): {args.max_flip_pct*100:.1f}%% (5x more than heuristic)")
    print(f"  Correction scale: {args.correction_scale}x (amplifies error correction weighting)")

    # Load data
    print("\n[1] Loading data...")
    try:
        js_means = np.load('./xspot_layer0_group0/js_means.npy')  # [4096]
        Wq = np.load('./xspot_layer0_group0/Wq_group0.npy')  # [4, 128, 4096]
        Wk = np.load('./xspot_layer0_group0/Wk_group0.npy')  # [128, 4096]
        Wv = np.load('./xspot_layer0_group0/Wv_group0.npy')  # [128, 4096]
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        print("Please ensure you've run xspot.py and have the following files:")
        print("  - js_means.npy")
        print("  - Wq_group0.npy")
        print("  - Wk_group0.npy")
        print("  - Wv_group0.npy")
        return

    print(f"  JS means: {js_means.shape}")
    print(f"  Wq (group 0): {Wq.shape} [num_heads, head_dim, hidden_size]")
    print(f"  Wk (group 0): {Wk.shape} [head_dim, hidden_size]")
    print(f"  Wv (group 0): {Wv.shape} [head_dim, hidden_size]")

    # Use JS means as input X (single activation vector)
    X = js_means  # [4096]
    print(f"\n  Using JS means as input X: {X.shape}")
    print(f"  X statistics: min={X.min():.6f}, max={X.max():.6f}, "
          f"mean={X.mean():.6f}, std={X.std():.6f}")

    # Quantize weights with BOTH strategies
    print("\n[2] Quantizing weights to INT4 (group_size=128)...")
    print("  Strategy 1: Nearest rounding")
    print("  Strategy 2: Heuristic flip correction (from awq_js_xl.py)")

    # Strategy 1: Nearest rounding
    print("\n  [2a] Quantizing with NEAREST rounding...")
    print("    Quantizing Wq (4 heads)...")
    Wq_quant_nearest, Wq_scales_nearest, Wq_zp_nearest, Wq_int_nearest = \
        quantize_weight_groupwise_int4(Wq, group_size=128)

    print("    Quantizing Wk (1 head)...")
    Wk_quant_nearest, Wk_scales_nearest, Wk_zp_nearest, Wk_int_nearest = \
        quantize_weight_groupwise_int4(Wk, group_size=128)

    # Strategy 2: Heuristic flip correction
    print("\n  [2b] Quantizing with HEURISTIC FLIP correction...")
    print("    Quantizing Wq (4 heads)...")
    Wq_quant_flip, Wq_scales_flip, Wq_zp_flip, Wq_int_flip, Wq_flip_stats = \
        quantize_weight_groupwise_int4_with_flip(Wq, X, group_size=128)

    print("    Quantizing Wk (1 head)...")
    Wk_quant_flip, Wk_scales_flip, Wk_zp_flip, Wk_int_flip, Wk_flip_stats = \
        quantize_weight_groupwise_int4_with_flip(Wk, X, group_size=128)

    print(f"\n  Wq scales: {Wq_scales_nearest.shape} [num_heads, head_dim, n_groups]")
    print(f"  Wk scales: {Wk_scales_nearest.shape} [head_dim, n_groups]")

    print(f"\n  Flip statistics:")
    print(f"    Wq: {Wq_flip_stats['total_flips']} flips "
          f"({Wq_flip_stats['flip_rate_pct']:.4f}% of weights)")
    print(f"        Dynamic outlier detection: {Wq_flip_stats['outlier_percent']*100:.2f}% outliers")
    print(f"        Error reduction: {Wq_flip_stats['error_reduction']:.6f} "
          f"({Wq_flip_stats['error_reduction_pct']:.2f}%)")
    print(f"        Error before flip: {Wq_flip_stats['error_before_flip']:.6f}")
    print(f"        Error after flip:  {Wq_flip_stats['error_after_flip']:.6f}")
    print(f"    Wk: {Wk_flip_stats['total_flips']} flips "
          f"({Wk_flip_stats['flip_rate_pct']:.4f}% of weights)")
    print(f"        Dynamic outlier detection: {Wk_flip_stats['outlier_percent']*100:.2f}% outliers")
    print(f"        Error reduction: {Wk_flip_stats['error_reduction']:.6f} "
          f"({Wk_flip_stats['error_reduction_pct']:.2f}%)")
    print(f"        Error before flip: {Wk_flip_stats['error_before_flip']:.6f}")
    print(f"        Error after flip:  {Wk_flip_stats['error_after_flip']:.6f}")

    # Strategy 3: ReFlip (targeted error correction on attention scores)
    print("\n  [2c] Applying REFLIP correction (attention score error redistribution)...")

    # Compute Q_orig and Q_heuristic for all heads (needed for ReFlip)
    num_heads = Wq.shape[0]
    Q_orig_all = np.zeros((num_heads, Wq.shape[1]))  # [4, 128]
    Q_heuristic_all = np.zeros((num_heads, Wq.shape[1]))  # [4, 128]

    for head_idx in range(num_heads):
        Q_orig_all[head_idx] = X @ Wq[head_idx].T  # [4096] @ [128, 4096]^T = [128]
        Q_heuristic_all[head_idx] = X @ Wq_quant_flip[head_idx].T

    # Compute K_heuristic (needed for attention score calculation)
    K_heuristic = X @ Wk_quant_flip.T  # [128]

    # Apply ReFlip (build on top of heuristic quantization)
    (Wq_quant_reflip, Wq_scales_reflip, Wq_zp_reflip, Wq_int_reflip,
     Wk_quant_reflip, Wk_scales_reflip, Wk_zp_reflip, Wk_int_reflip,
     reflip_stats) = quantize_qkv_reflip(
        Wq, Wk, X, Q_orig_all, Q_heuristic_all,
        Wq_quant_flip, Wk_quant_flip, K_heuristic,  # Pass heuristic-quantized weights and K
        Wq_int_flip, Wq_scales_flip, Wq_zp_flip,  # Pass integer representation and scales
        critical_dim_pct=args.critical_dim_pct,
        knee_tolerance=args.knee_tolerance,
        group_size=args.group_size,
        max_flip_pct=args.max_flip_pct,
        correction_scale=args.correction_scale,
        debug=args.debug
    )

    print(f"\n  ReFlip statistics:")
    print(f"    Total moderate dims: {reflip_stats['total_moderate_dims']} "
          f"across {num_heads} heads")
    print(f"    Moderate dims per head: {reflip_stats['moderate_dims_per_head']}")
    print(f"    Target percentage: {reflip_stats['moderate_dim_pct']*100:.1f}%")
    print(f"    Attention score errors to correct: {reflip_stats['score_errors']}")
    print(f"    Total integer flips applied: {reflip_stats['total_flips']} "
          f"({reflip_stats['flip_rate_pct']:.4f}% of weights)")

    # Compute weight quantization errors
    print("\n[3] Weight quantization errors:")
    print("\n  Strategy 1: NEAREST rounding")
    for head_idx in range(4):
        err = compute_quantization_error(Wq[head_idx], Wq_quant_nearest[head_idx])
        print(f"    Wq head {head_idx}: MAE={err['mae']:.6f}, "
              f"Max={err['max_error']:.6f}, Rel={err['rel_error_pct']:.4f}%")

    err_k_nearest = compute_quantization_error(Wk, Wk_quant_nearest)
    print(f"    Wk: MAE={err_k_nearest['mae']:.6f}, "
          f"Max={err_k_nearest['max_error']:.6f}, Rel={err_k_nearest['rel_error_pct']:.4f}%")

    print("\n  Strategy 2: HEURISTIC FLIP correction")
    for head_idx in range(4):
        err = compute_quantization_error(Wq[head_idx], Wq_quant_flip[head_idx])
        print(f"    Wq head {head_idx}: MAE={err['mae']:.6f}, "
              f"Max={err['max_error']:.6f}, Rel={err['rel_error_pct']:.4f}%")

    err_k_flip = compute_quantization_error(Wk, Wk_quant_flip)
    print(f"    Wk: MAE={err_k_flip['mae']:.6f}, "
          f"Max={err_k_flip['max_error']:.6f}, Rel={err_k_flip['rel_error_pct']:.4f}%")

    print("\n  Strategy 3: REFLIP correction")
    for head_idx in range(4):
        err = compute_quantization_error(Wq[head_idx], Wq_quant_reflip[head_idx])
        print(f"    Wq head {head_idx}: MAE={err['mae']:.6f}, "
              f"Max={err['max_error']:.6f}, Rel={err['rel_error_pct']:.4f}%")

    err_k_reflip = compute_quantization_error(Wk, Wk_quant_reflip)
    print(f"    Wk: MAE={err_k_reflip['mae']:.6f}, "
          f"Max={err_k_reflip['max_error']:.6f}, Rel={err_k_reflip['rel_error_pct']:.4f}%")

    # Compute attention scores
    print("\n[4] Computing attention scores: (X @ Wq^T) · (X @ Wk^T)")
    print("="*70)

    num_heads = 4
    results = []

    for head_idx in range(num_heads):
        print(f"\n--- Query Head {head_idx} ---")

        # Original computation
        # Q = X @ Wq^T: [4096] @ [4096, 128]^T = [128]
        Q_orig = X @ Wq[head_idx].T  # [128]

        # K = X @ Wk^T: [4096] @ [4096, 128]^T = [128]
        K_orig = X @ Wk.T  # [128] (shared across all heads)

        # Attention score = Q · K (dot product)
        score_orig = Q_orig @ K_orig  # scalar

        # Strategy 1: Nearest rounding
        Q_quant_nearest = X @ Wq_quant_nearest[head_idx].T
        K_quant_nearest = X @ Wk_quant_nearest.T
        score_quant_nearest = Q_quant_nearest @ K_quant_nearest

        # Strategy 2: Heuristic flip
        Q_quant_flip = X @ Wq_quant_flip[head_idx].T
        K_quant_flip = X @ Wk_quant_flip.T
        score_quant_flip = Q_quant_flip @ K_quant_flip

        # Strategy 3: ReFlip
        Q_quant_reflip = X @ Wq_quant_reflip[head_idx].T
        K_quant_reflip = X @ Wk_quant_reflip.T
        score_quant_reflip = Q_quant_reflip @ K_quant_reflip

        # Errors
        error_nearest = score_quant_nearest - score_orig
        rel_error_nearest = error_nearest / (np.abs(score_orig) + 1e-10) * 100

        error_flip = score_quant_flip - score_orig
        rel_error_flip = error_flip / (np.abs(score_orig) + 1e-10) * 100

        error_reflip = score_quant_reflip - score_orig
        rel_error_reflip = error_reflip / (np.abs(score_orig) + 1e-10) * 100

        improvement_h = error_nearest - error_flip
        improvement_h_pct = (abs(error_nearest) - abs(error_flip)) / (abs(error_nearest) + 1e-10) * 100

        improvement_r = error_nearest - error_reflip
        improvement_r_pct = (abs(error_nearest) - abs(error_reflip)) / (abs(error_nearest) + 1e-10) * 100

        improvement_hr = error_flip - error_reflip
        improvement_hr_pct = (abs(error_flip) - abs(error_reflip)) / (abs(error_flip) + 1e-10) * 100

        print(f"Original score:           {score_orig:15.6f}")
        print(f"\nStrategy 1 (Nearest):")
        print(f"  Score:                  {score_quant_nearest:15.6f}")
        print(f"  Absolute error:         {error_nearest:15.6f}")
        print(f"  Relative error:         {rel_error_nearest:15.4f}%")
        print(f"\nStrategy 2 (Heuristic):")
        print(f"  Score:                  {score_quant_flip:15.6f}")
        print(f"  Absolute error:         {error_flip:15.6f}")
        print(f"  Relative error:         {rel_error_flip:15.4f}%")
        print(f"\nStrategy 3 (ReFlip):")
        print(f"  Score:                  {score_quant_reflip:15.6f}")
        print(f"  Absolute error:         {error_reflip:15.6f}")
        print(f"  Relative error:         {rel_error_reflip:15.4f}%")
        print(f"\nImprovements:")
        print(f"  Nearest → Heuristic:    {improvement_h_pct:15.2f}%")
        print(f"  Nearest → ReFlip:       {improvement_r_pct:15.2f}%")
        print(f"  Heuristic → ReFlip:     {improvement_hr_pct:15.2f}%")

        # Scale statistics for this head
        head_scales_nearest = Wq_scales_nearest[head_idx].flatten()
        head_scales_flip = Wq_scales_flip[head_idx].flatten()
        head_scales_reflip = Wq_scales_reflip[head_idx].flatten()

        results.append({
            'head': head_idx,
            'score_orig': score_orig,
            'score_quant_nearest': score_quant_nearest,
            'score_quant_flip': score_quant_flip,
            'score_quant_reflip': score_quant_reflip,
            'error_nearest': error_nearest,
            'error_flip': error_flip,
            'error_reflip': error_reflip,
            'rel_error_nearest': rel_error_nearest,
            'rel_error_flip': rel_error_flip,
            'rel_error_reflip': rel_error_reflip,
            'improvement_h_pct': improvement_h_pct,
            'improvement_r_pct': improvement_r_pct,
            'improvement_hr_pct': improvement_hr_pct,
            'wq_scales_nearest': head_scales_nearest,
            'wq_scales_flip': head_scales_flip,
            'wq_scales_reflip': head_scales_reflip,
            'Q_orig': Q_orig,
            'Q_quant_nearest': Q_quant_nearest,
            'Q_quant_flip': Q_quant_flip,
            'Q_quant_reflip': Q_quant_reflip,
            'K_orig': K_orig,
            'K_quant_nearest': K_quant_nearest,
            'K_quant_flip': K_quant_flip,
            'K_quant_reflip': K_quant_reflip
        })

    # Summary table
    print("\n" + "="*100)
    print("SUMMARY: Attention Score Quantization Comparison (3 Strategies)")
    print("="*100)
    print(f"{'Head':<6} {'Original':<13} {'Nearest':<13} {'Heuristic':<13} {'ReFlip':<13} "
          f"{'Err(N)':<10} {'Err(H)':<10} {'Err(R)':<10}")
    print("-"*100)
    for r in results:
        print(f"{r['head']:<6} {r['score_orig']:<13.6f} "
              f"{r['score_quant_nearest']:<13.6f} {r['score_quant_flip']:<13.6f} {r['score_quant_reflip']:<13.6f} "
              f"{r['error_nearest']:<10.4f} {r['error_flip']:<10.4f} {r['error_reflip']:<10.4f}")

    errors_nearest = [r['error_nearest'] for r in results]
    errors_flip = [r['error_flip'] for r in results]
    errors_reflip = [r['error_reflip'] for r in results]
    rel_errors_nearest = [r['rel_error_nearest'] for r in results]
    rel_errors_flip = [r['rel_error_flip'] for r in results]
    rel_errors_reflip = [r['rel_error_reflip'] for r in results]
    improvements_h = [r['improvement_h_pct'] for r in results]
    improvements_r = [r['improvement_r_pct'] for r in results]
    improvements_hr = [r['improvement_hr_pct'] for r in results]

    print("-"*100)
    print("\nStrategy 1 (Nearest):")
    print(f"  Mean absolute error:    {np.mean(np.abs(errors_nearest)):.6f}")
    print(f"  Mean relative error:    {np.mean(np.abs(rel_errors_nearest)):.4f}%")
    print(f"  Max absolute error:     {np.max(np.abs(errors_nearest)):.6f}")

    print("\nStrategy 2 (Heuristic):")
    print(f"  Mean absolute error:    {np.mean(np.abs(errors_flip)):.6f}")
    print(f"  Mean relative error:    {np.mean(np.abs(rel_errors_flip)):.4f}%")
    print(f"  Max absolute error:     {np.max(np.abs(errors_flip)):.6f}")

    print("\nStrategy 3 (ReFlip):")
    print(f"  Mean absolute error:    {np.mean(np.abs(errors_reflip)):.6f}")
    print(f"  Mean relative error:    {np.mean(np.abs(rel_errors_reflip)):.4f}%")
    print(f"  Max absolute error:     {np.max(np.abs(errors_reflip)):.6f}")

    print("\nImprovements:")
    print(f"  Nearest → Heuristic:    {np.mean(improvements_h):.2f}% (mean)")
    print(f"  Nearest → ReFlip:       {np.mean(improvements_r):.2f}% (mean)")
    print(f"  Heuristic → ReFlip:     {np.mean(improvements_hr):.2f}% (mean)")
    print(f"  Best ReFlip reduction:  {np.max(improvements_r):.2f}% (head {np.argmax(improvements_r)})")
    print(f"  Worst ReFlip reduction: {np.min(improvements_r):.2f}% (head {np.argmin(improvements_r)})")

    print("\nFlip Statistics:")
    print(f"  Wq total flips:         {Wq_flip_stats['total_flips']:,} "
          f"({Wq_flip_stats['flip_rate_pct']:.4f}% of weights)")
    print(f"  Wq error reduction:     {Wq_flip_stats['error_reduction']:.6f} "
          f"({Wq_flip_stats['error_reduction_pct']:.2f}%)")
    print(f"  Wk total flips:         {Wk_flip_stats['total_flips']:,} "
          f"({Wk_flip_stats['flip_rate_pct']:.4f}% of weights)")
    print(f"  Wk error reduction:     {Wk_flip_stats['error_reduction']:.6f} "
          f"({Wk_flip_stats['error_reduction_pct']:.2f}%)")
    print("="*70)

    # Visualization
    print("\n[5] Generating visualizations...")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    heads = [r['head'] for r in results]
    scores_orig = [r['score_orig'] for r in results]
    scores_quant_nearest = [r['score_quant_nearest'] for r in results]
    scores_quant_flip = [r['score_quant_flip'] for r in results]
    scores_quant_reflip = [r['score_quant_reflip'] for r in results]

    # 1. Attention scores comparison (4-way)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(heads))
    width = 0.2
    ax1.bar(x - 1.5*width, scores_orig, width, label='Original', alpha=0.8, color='blue')
    ax1.bar(x - 0.5*width, scores_quant_nearest, width, label='Nearest', alpha=0.8, color='orange')
    ax1.bar(x + 0.5*width, scores_quant_flip, width, label='Heuristic', alpha=0.8, color='green')
    ax1.bar(x + 1.5*width, scores_quant_reflip, width, label='ReFlip', alpha=0.8, color='purple')
    ax1.set_xlabel('Query Head')
    ax1.set_ylabel('Attention Score (Q·K)')
    ax1.set_title('Original vs Quantized Attention Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'H{i}' for i in heads])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Error comparison (3-way)
    ax2 = fig.add_subplot(gs[0, 1])
    width2 = 0.25
    ax2.bar(x - width2, errors_nearest, width2, label='Nearest', alpha=0.7, color='orange')
    ax2.bar(x, errors_flip, width2, label='Heuristic', alpha=0.7, color='green')
    ax2.bar(x + width2, errors_reflip, width2, label='ReFlip', alpha=0.7, color='purple')
    ax2.set_xlabel('Query Head')
    ax2.set_ylabel('Error (Quantized - Original)')
    ax2.set_title('Absolute Error Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'H{i}' for i in heads])
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Error reduction % (comparing all strategies)
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(heads))
    width = 0.25
    ax3.bar(x - width, improvements_h, width, alpha=0.7, color='orange', label='N→H')
    ax3.bar(x, improvements_r, width, alpha=0.7, color='green', label='N→R')
    ax3.bar(x + width, improvements_hr, width, alpha=0.7, color='purple', label='H→R')
    ax3.set_xlabel('Query Head')
    ax3.set_ylabel('Error Reduction (%)')
    ax3.set_title('Error Reduction Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'H{i}' for i in heads])
    ax3.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Relative errors comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x - width/2, rel_errors_nearest, width, label='Nearest', alpha=0.7, color='orange')
    ax4.bar(x + width/2, rel_errors_flip, width, label='Heuristic', alpha=0.7, color='green')
    ax4.set_xlabel('Query Head')
    ax4.set_ylabel('Relative Error (%)')
    ax4.set_title('Relative Error Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'H{i}' for i in heads])
    ax4.axhline(0, color='black', linestyle='--', linewidth=1)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Scale distributions comparison (nearest vs flip)
    ax5 = fig.add_subplot(gs[1, 1])
    scale_data_nearest = [results[i]['wq_scales_nearest'] for i in range(num_heads)]
    scale_data_flip = [results[i]['wq_scales_flip'] for i in range(num_heads)]
    # Show first head as example
    ax5.hist(scale_data_nearest[0], bins=30, alpha=0.5, label='Nearest (H0)', color='orange')
    ax5.hist(scale_data_flip[0], bins=30, alpha=0.5, label='Heuristic (H0)', color='green')
    ax5.set_xlabel('Scale Value')
    ax5.set_ylabel('Count')
    ax5.set_title('Wq Scale Distribution (Head 0)')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Wk scale comparison
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(Wk_scales_nearest.flatten(), bins=30, alpha=0.5, label='Nearest',
             color='orange', edgecolor='black')
    ax6.hist(Wk_scales_flip.flatten(), bins=30, alpha=0.5, label='Heuristic',
             color='green', edgecolor='black')
    ax6.set_xlabel('Scale Value')
    ax6.set_ylabel('Count')
    ax6.set_title(f'Wk Scale Distribution Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Q vector comparison (head 0) - 3-way
    ax7 = fig.add_subplot(gs[2, 0])
    Q0_orig = results[0]['Q_orig']
    Q0_nearest = results[0]['Q_quant_nearest']
    Q0_flip = results[0]['Q_quant_flip']
    ax7.plot(Q0_orig, label='Original', alpha=0.7, linewidth=1.5, color='blue')
    ax7.plot(Q0_nearest, label='Nearest', alpha=0.7, linewidth=1.5, color='orange')
    ax7.plot(Q0_flip, label='Heuristic', alpha=0.7, linewidth=1.5, color='green')
    ax7.set_xlabel('Dimension')
    ax7.set_ylabel('Value')
    ax7.set_title('Query Vector Q (Head 0)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. K vector comparison (shared) - 3-way
    ax8 = fig.add_subplot(gs[2, 1])
    K_orig = results[0]['K_orig']
    K_nearest = results[0]['K_quant_nearest']
    K_flip = results[0]['K_quant_flip']
    ax8.plot(K_orig, label='Original', alpha=0.7, linewidth=1.5, color='blue')
    ax8.plot(K_nearest, label='Nearest', alpha=0.7, linewidth=1.5, color='orange')
    ax8.plot(K_flip, label='Heuristic', alpha=0.7, linewidth=1.5, color='green')
    ax8.set_xlabel('Dimension')
    ax8.set_ylabel('Value')
    ax8.set_title('Key Vector K (Shared)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Q-K error comparison (head 0)
    ax9 = fig.add_subplot(gs[2, 2])
    Q_error_nearest = Q0_nearest - Q0_orig
    Q_error_flip = Q0_flip - Q0_orig
    ax9.scatter(Q_error_nearest, Q_error_flip, alpha=0.5, s=10)
    ax9.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax9.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax9.plot([Q_error_nearest.min(), Q_error_nearest.max()],
             [Q_error_nearest.min(), Q_error_nearest.max()],
             'k--', alpha=0.5, label='y=x (no improvement)')
    ax9.set_xlabel('Q Error (Nearest)')
    ax9.set_ylabel('Q Error (Heuristic)')
    ax9.set_title('Q Error: Nearest vs Heuristic (Head 0)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 10. Flip statistics per head
    ax10 = fig.add_subplot(gs[3, 0])
    # This would require per-head flip stats, skip for now
    ax10.text(0.5, 0.5, f"Wq Flips:\n"
                         f"  Total: {Wq_flip_stats['total_flips']:,} ({Wq_flip_stats['flip_rate_pct']:.4f}%)\n"
                         f"  Error reduction: {Wq_flip_stats['error_reduction_pct']:.2f}%\n\n"
                         f"Wk Flips:\n"
                         f"  Total: {Wk_flip_stats['total_flips']:,} ({Wk_flip_stats['flip_rate_pct']:.4f}%)\n"
                         f"  Error reduction: {Wk_flip_stats['error_reduction_pct']:.2f}%",
              ha='center', va='center', fontsize=11,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax10.set_xlim(0, 1)
    ax10.set_ylim(0, 1)
    ax10.axis('off')
    ax10.set_title('Flip Statistics')

    # 11. Error reduction summary
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.text(0.5, 0.5, f"Error Reduction (mean):\n"
                         f"N→H: {np.mean(improvements_h):.2f}%\n"
                         f"N→R: {np.mean(improvements_r):.2f}%\n"
                         f"H→R: {np.mean(improvements_hr):.2f}%\n\n"
                         f"Strategy MAE:\n"
                         f"Nearest: {np.mean(np.abs(errors_nearest)):.6f}\n"
                         f"Heuristic: {np.mean(np.abs(errors_flip)):.6f}\n"
                         f"ReFlip: {np.mean(np.abs(errors_reflip)):.6f}",
              ha='center', va='center', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax11.set_xlim(0, 1)
    ax11.set_ylim(0, 1)
    ax11.axis('off')
    ax11.set_title('Summary Statistics')

    # 12. Reserved for future use
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis('off')

    plt.savefig('attention_quantization_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: attention_quantization_analysis.png")

    # New figure: Sorted errors over 128 head dimensions for 4 heads
    # Split into two plots per head: errors (top) and magnitude (bottom)
    print("\n[5b] Generating sorted error visualization...")
    fig2 = plt.figure(figsize=(16, 16))
    gs2 = fig2.add_gridspec(4, 4, hspace=0.4, wspace=0.3, height_ratios=[1, 0.6, 1, 0.6])

    for head_idx in range(num_heads):
        # Calculate grid position
        col = head_idx % 2  # 0 or 1
        row_offset = (head_idx // 2) * 2  # 0 or 2

        # Top subplot: Errors
        ax_error = fig2.add_subplot(gs2[row_offset, col*2:(col+1)*2])

        # Bottom subplot: Magnitude
        ax_mag = fig2.add_subplot(gs2[row_offset + 1, col*2:(col+1)*2], sharex=ax_error)

        # Get Q vectors for this head
        Q_orig = results[head_idx]['Q_orig']
        Q_nearest = results[head_idx]['Q_quant_nearest']
        Q_flip = results[head_idx]['Q_quant_flip']

        # Compute errors (keeping sign)
        Q_error_nearest = Q_nearest - Q_orig
        Q_error_flip = Q_flip - Q_orig

        # Get sort indices from nearest error
        sort_indices = np.argsort(Q_error_nearest)

        # Sort errors and Q_orig magnitude using the same indices
        sorted_error_nearest = Q_error_nearest[sort_indices]
        sorted_error_flip = Q_error_flip[sort_indices]
        sorted_Q_magnitude = np.abs(Q_orig[sort_indices])

        x_dims = np.arange(len(sorted_error_nearest))

        # Top plot: Errors
        ax_error.plot(x_dims, sorted_error_nearest, label='Error (Nearest)', alpha=0.8,
                      linewidth=1.5, color='orange', marker='o', markersize=2)
        ax_error.plot(x_dims, sorted_error_flip, label='Error (Heuristic)', alpha=0.8,
                      linewidth=1.5, color='green', marker='s', markersize=2)
        ax_error.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Statistics
        mean_nearest = Q_error_nearest.mean()
        mean_flip = Q_error_flip.mean()
        std_nearest = Q_error_nearest.std()
        std_flip = Q_error_flip.std()

        ax_error.set_ylabel('Error', fontsize=10)
        ax_error.set_title(f'Head {head_idx}: Sorted Q Errors (128 dims)\n'
                          f'Nearest: μ={mean_nearest:.4f}, σ={std_nearest:.4f} | '
                          f'Heuristic: μ={mean_flip:.4f}, σ={std_flip:.4f}',
                          fontsize=9)
        ax_error.legend(loc='best', fontsize=9)
        ax_error.grid(True, alpha=0.3)
        ax_error.tick_params(labelbottom=False)

        # Bottom plot: Magnitude
        ax_mag.plot(x_dims, sorted_Q_magnitude, label='|Q_orig|', alpha=0.7,
                    linewidth=1.5, color='blue', marker='s', markersize=3)
        ax_mag.set_xlabel('Sorted Dimension Index', fontsize=10)
        ax_mag.set_ylabel('|Q_orig| Magnitude', fontsize=10)
        ax_mag.legend(loc='best', fontsize=9)
        ax_mag.grid(True, alpha=0.3)

    plt.savefig('sorted_error_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: sorted_error_comparison.png")

    # Save detailed results
    print("\n[6] Saving detailed results...")
    np.savez('quantization_results.npz',
             # Input
             X=X,
             js_means=js_means,
             # Original weights
             Wq_orig=Wq,
             Wk_orig=Wk,
             # Quantized weights - Nearest
             Wq_quant_nearest=Wq_quant_nearest,
             Wk_quant_nearest=Wk_quant_nearest,
             Wq_int_nearest=Wq_int_nearest,
             Wk_int_nearest=Wk_int_nearest,
             # Quantized weights - Heuristic
             Wq_quant_flip=Wq_quant_flip,
             Wk_quant_flip=Wk_quant_flip,
             Wq_int_flip=Wq_int_flip,
             Wk_int_flip=Wk_int_flip,
             # Scales and zero points - Nearest
             Wq_scales_nearest=Wq_scales_nearest,
             Wk_scales_nearest=Wk_scales_nearest,
             Wq_zp_nearest=Wq_zp_nearest,
             Wk_zp_nearest=Wk_zp_nearest,
             # Scales and zero points - Heuristic
             Wq_scales_flip=Wq_scales_flip,
             Wk_scales_flip=Wk_scales_flip,
             Wq_zp_flip=Wq_zp_flip,
             Wk_zp_flip=Wk_zp_flip,
             # Attention scores
             attention_scores_orig=np.array(scores_orig),
             attention_scores_nearest=np.array(scores_quant_nearest),
             attention_scores_flip=np.array(scores_quant_flip),
             attention_scores_reflip=np.array(scores_quant_reflip),
             # Errors
             errors_nearest=np.array(errors_nearest),
             errors_flip=np.array(errors_flip),
             errors_reflip=np.array(errors_reflip),
             rel_errors_nearest=np.array(rel_errors_nearest),
             rel_errors_flip=np.array(rel_errors_flip),
             rel_errors_reflip=np.array(rel_errors_reflip),
             improvements_h=np.array(improvements_h),
             improvements_r=np.array(improvements_r),
             improvements_hr=np.array(improvements_hr),
             # Query and Key vectors
             Q_orig=[r['Q_orig'] for r in results],
             Q_quant_nearest=[r['Q_quant_nearest'] for r in results],
             Q_quant_flip=[r['Q_quant_flip'] for r in results],
             Q_quant_reflip=[r['Q_quant_reflip'] for r in results],
             K_orig=K_orig,
             K_quant_nearest=K_nearest,
             K_quant_flip=K_flip,
             K_quant_reflip=K_quant_reflip)
    print(f"  Saved: quantization_results.npz")

    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70)
    print("\nKey findings:")
    print(f"  1. Nearest quantization:")
    print(f"     - Mean attention score error: {np.mean(np.abs(errors_nearest)):.6f} "
          f"({np.mean(np.abs(rel_errors_nearest)):.4f}%)")
    print(f"  2. Heuristic quantization:")
    print(f"     - Mean attention score error: {np.mean(np.abs(errors_flip)):.6f} "
          f"({np.mean(np.abs(rel_errors_flip)):.4f}%)")
    print(f"     - Flipped {Wq_flip_stats['total_flips'] + Wk_flip_stats['total_flips']:,} weights "
          f"({((Wq_flip_stats['total_flips'] + Wk_flip_stats['total_flips']) / (Wq.size + Wk.size) * 100):.4f}%)")
    print(f"     - Error reduction from flipping: Wq={Wq_flip_stats['error_reduction_pct']:.2f}%, "
          f"Wk={Wk_flip_stats['error_reduction_pct']:.2f}%")
    print(f"  3. ReFlip quantization:")
    print(f"     - Mean attention score error: {np.mean(np.abs(errors_reflip)):.6f} "
          f"({np.mean(np.abs(rel_errors_reflip)):.4f}%)")
    print(f"     - Corrected {reflip_stats['total_moderate_dims']} moderate dimensions "
          f"({reflip_stats['moderate_dim_pct']*100:.1f}% target)")
    print(f"     - Attention score errors: {[f'{e:.6f}' for e in reflip_stats['score_errors']]}")
    print(f"\n  Improvements:")
    print(f"     - Nearest → Heuristic: {np.mean(improvements_h):.2f}% average error reduction")
    print(f"     - Nearest → ReFlip:    {np.mean(improvements_r):.2f}% average error reduction")
    print(f"     - Heuristic → ReFlip:  {np.mean(improvements_hr):.2f}% average error reduction")
    print(f"     - Best ReFlip improvement: {np.max(improvements_r):.2f}% (head {np.argmax(improvements_r)})")


if __name__ == '__main__':
    main()
