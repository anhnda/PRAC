"""
Utility functions for INT4 quantization on GQA attention.

This module contains stable utility functions that are less likely to change:
- Kneedle algorithm for finding knee points
- Dynamic outlier threshold computation
- Basic group-wise INT4 quantization
- Quantization error metrics
"""

import numpy as np


def find_knee_point(values, tolerance_offset=0.0):
    """
    Find knee point in sorted values using Kneedle algorithm.

    Args:
        values: 1D array of sorted values (ascending order)
        tolerance_offset: Additional offset to add to knee point (default: 0.0)

    Returns:
        index of knee point

    Algorithm:
    1. Normalize values to [0, 1]
    2. Create reference line from start to end
    3. Find point with maximum distance from reference line
    4. Apply tolerance offset
    """
    n = len(values)
    if n < 3:
        return n // 2

    # Normalize to [0, 1]
    y_min, y_max = values.min(), values.max()
    if y_max - y_min < 1e-10:
        # All values are the same, no knee
        return n // 2

    y_norm = (values - y_min) / (y_max - y_min)
    x_norm = np.linspace(0, 1, n)

    # Compute distances from the line connecting first and last point
    # Line equation: y = m*x + b
    # For normalized: line goes from (0, y_norm[0]) to (1, y_norm[-1])
    y_line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm

    # Perpendicular distance from each point to the line
    distances = np.abs(y_norm - y_line)

    # Find point with maximum distance (the knee)
    knee_idx = np.argmax(distances)

    # Apply tolerance offset
    if knee_idx < n - 1:
        # Calculate how many indices to shift based on offset
        offset_indices = int(tolerance_offset * n)
        knee_idx = min(knee_idx + offset_indices, n - 1)
        knee_idx = max(knee_idx, 0)

    return knee_idx


def compute_dynamic_outlier_threshold(activation_means, knee_tolerance=0.0, debug=False):
    """
    Compute dynamic outlier threshold using Kneedle algorithm.

    Strategy:
    1. Sort activation means in DESCENDING order [high ... medium ... low]
    2. Apply Kneedle to FIRST HALF [high ... medium] to find outlier→normal transition
    3. Tolerance offset allows tuning: positive = more conservative (keep more outliers)

    Args:
        activation_means: Array of per-channel activation means (E[X])
        knee_tolerance: Tolerance offset for knee point (default: 0.0)
        debug: Print debug information

    Returns:
        tuple: (threshold value, outlier percentage)
    """
    # Sort activation means in DESCENDING order [high → low]
    sorted_means = np.sort(np.abs(activation_means))[::-1]  # Descending
    n = len(sorted_means)

    # Apply Kneedle to FIRST HALF [high ... medium] to find outlier transition
    first_half = sorted_means[:n // 2]

    if len(first_half) < 3:
        # Not enough data, use a conservative default (top 5%)
        threshold_idx = int(0.05 * n)
        threshold = sorted_means[threshold_idx]
        outlier_percent = 0.05
        if debug:
            print(f"    DEBUG: Not enough data for Kneedle, using top 5% as default")
        return threshold, outlier_percent

    # Find knee point in first half (where outliers end, normal begins)
    knee_idx_in_half = find_knee_point(first_half, tolerance_offset=knee_tolerance)

    # This is already the index in full array (descending sorted)
    knee_idx = knee_idx_in_half

    # The threshold is the value at the knee point
    threshold = sorted_means[knee_idx]

    # Count how many channels are outliers (above or equal to threshold)
    num_outliers = (np.abs(activation_means) >= threshold).sum()
    outlier_percent = num_outliers / n

    if debug:
        print(f"    DEBUG: Sorted means (descending): [{sorted_means[0]:.6f} ... {sorted_means[-1]:.6f}]")
        print(f"    DEBUG: First half range: [{first_half[0]:.6f} ... {first_half[-1]:.6f}]")
        print(f"    DEBUG: Knee point index in first half: {knee_idx_in_half}/{len(first_half)}")
        print(f"    DEBUG: Knee point index in full array: {knee_idx}/{n} ({knee_idx/n*100:.1f}%)")
        print(f"    DEBUG: Knee threshold value: {threshold:.6f}")
        print(f"    DEBUG: Outliers (>= threshold): {num_outliers}/{n} ({outlier_percent*100:.2f}%)")
        print(f"    DEBUG: vs Default 5.00%: {outlier_percent*100 - 5.0:+.2f}% difference")

    return threshold, outlier_percent


def quantize_weight_groupwise_int4(W, group_size=128, method='nearest'):
    """
    Quantize weights to INT4 using group-wise asymmetric quantization [0, 15].

    Args:
        W: Weight matrix of shape [..., in_features]
        group_size: Size of each quantization group (default: 128)
        method: 'nearest' for nearest rounding (default)

    Returns:
        W_quant: Dequantized weights (same shape as W)
        scales: Per-group scales, shape [..., n_groups]
        zp: Per-group zero points, shape [..., n_groups]
        W_int: Integer weights, shape [..., in_features]
    """
    original_shape = W.shape

    # Flatten to 2D if needed
    if W.ndim > 2:
        # Reshape keeping last dimension
        W_flat = W.reshape(-1, W.shape[-1])
    else:
        W_flat = W.copy()

    out_features, in_features = W_flat.shape
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size

    # Pad if needed
    if padded_in > in_features:
        W_padded = np.zeros((out_features, padded_in), dtype=W.dtype)
        W_padded[:, :in_features] = W_flat
    else:
        W_padded = W_flat

    # Reshape to groups: [out_features, n_groups, group_size]
    W_grouped = W_padded.reshape(out_features, n_groups, group_size)

    # Asymmetric quantization [0, 15] (INT4)
    w_min = W_grouped.min(axis=2, keepdims=True)
    w_max = W_grouped.max(axis=2, keepdims=True)
    max_int = 15

    # Compute scale and zero point
    scale = (w_max - w_min) / max_int
    scale = np.maximum(scale, 1e-8)  # Avoid division by zero
    zp = np.round(-w_min / scale).clip(0, max_int)

    # Quantize with NEAREST rounding
    W_div = W_grouped / scale
    W_int = np.round(W_div + zp).clip(0, max_int)

    # Dequantize
    W_quant_grouped = (W_int - zp) * scale

    # Reshape back to flat
    W_quant_flat = W_quant_grouped.reshape(out_features, padded_in)
    W_int_flat = W_int.reshape(out_features, padded_in)

    # Remove padding
    if padded_in > in_features:
        W_quant_flat = W_quant_flat[:, :in_features]
        W_int_flat = W_int_flat[:, :in_features]

    # Reshape back to original shape
    W_quant = W_quant_flat.reshape(original_shape)
    W_int_final = W_int_flat.reshape(original_shape)

    # Flatten scales and zp for output
    scales_flat = scale.reshape(out_features, n_groups)
    zp_flat = zp.reshape(out_features, n_groups)

    # If original was 3D, reshape scales/zp to match original structure
    if len(original_shape) == 3:
        # Original: [num_heads, head_dim, hidden_size] e.g. [4, 128, 4096]
        # Flattened to: [num_heads * head_dim, hidden_size] e.g. [512, 4096]
        # Scales: [num_heads * head_dim, n_groups] e.g. [512, 32]
        # Reshape to: [num_heads, head_dim, n_groups] e.g. [4, 128, 32]
        scales_out = scales_flat.reshape(original_shape[0], original_shape[1], n_groups)
        zp_out = zp_flat.reshape(original_shape[0], original_shape[1], n_groups)
    else:
        scales_out = scales_flat
        zp_out = zp_flat

    return W_quant, scales_out, zp_out, W_int_final


def compute_quantization_error(W_orig, W_quant):
    """Compute quantization error metrics."""
    diff = W_quant - W_orig
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_error = np.max(np.abs(diff))
    rel_error = mae / (np.mean(np.abs(W_orig)) + 1e-10) * 100

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'rel_error_pct': rel_error
    }
