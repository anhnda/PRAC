"""
Moderate Weight Correction for Wanda Pruning

This script implements a novel weight correction strategy:
1. Apply Wanda pruning (50% sparsity by default)
2. Select "moderate" weight positions based on |JS_mean(X_j)|:
   - Sort |JS_mean| in descending order
   - Apply Kneedle algorithm on first half to find knee point
   - Select positions starting from knee + offset (default 0%)
   - Select percent_change (default 5%) of in_features
3. Correct weights at moderate positions to minimize reconstruction error:
   - Error = (W_pruned - W_orig) · JS_mean
   - For each moderate position j:
     delta_W[j] = -error * sign(JS_mean[j]) / sum(|JS_mean[selected]|)
4. Report reconstruction quality for both JS_mean and original mean

Usage:
    python prac_simple.py --input-dir ./exported_wanda_data --sparsity 0.5
"""

import numpy as np
import pandas as pd
import argparse
import os
import glob
import matplotlib.pyplot as plt


def find_knee_point(values, tolerance_offset=0.0):
    """
    Find knee point in sorted values using a simple Kneedle-like algorithm.
    (Borrowed from awq_js_xl.py)

    Args:
        values: 1D array of sorted values (ascending or descending order)
        tolerance_offset: Additional offset to add to knee point

    Returns:
        index of knee point
    """
    n = len(values)
    if n < 3:
        return n // 2

    # Convert to numpy
    y = np.array(values)

    # Normalize to [0, 1]
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return n // 2

    y_norm = (y - y_min) / (y_max - y_min)
    x_norm = np.linspace(0, 1, n)

    # Compute distances from the line connecting first and last point
    y_line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    distances = np.abs(y_norm - y_line)

    # Find point with maximum distance (the knee)
    knee_idx = np.argmax(distances)

    # Apply tolerance offset
    if knee_idx < n - 1:
        offset_indices = int(tolerance_offset * n)
        knee_idx = min(knee_idx + offset_indices, n - 1)
        knee_idx = max(knee_idx, 0)

    return knee_idx


def load_exported_data(input_dir):
    """Load data exported by export_wd.py."""
    npz_files = glob.glob(os.path.join(input_dir, "*_complete.npz"))

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No *_complete.npz file found in {input_dir}")

    npz_path = npz_files[0]
    print(f"Loading data from: {npz_path}")

    data = np.load(npz_path)

    result = {
        'W': data['W'],
        'E_X': data['E_X'],
        'L2_norm_X': data['L2_norm_X'],
        'JS_mean_X': data['JS_mean_X'],
        'layer_name': str(data['layer_name']),
        'out_features': int(data['out_features']),
        'in_features': int(data['in_features']),
    }

    return result


def compute_wanda_scores(W_row, L2_norm_X):
    """Compute Wanda scores: Score_j = |W_j| * ||X_j||_2"""
    scores = np.abs(W_row) * L2_norm_X
    return scores


def prune_weights_wanda(W_row, scores, sparsity):
    """
    Prune weights based on Wanda scores.

    Returns:
        W_pruned: Pruned weight row
        mask: Boolean mask (True = kept, False = pruned)
    """
    in_features = len(W_row)
    num_to_keep = int(in_features * (1 - sparsity))

    # Get indices of top scores to keep
    top_indices = np.argsort(scores)[-num_to_keep:]

    # Create mask
    mask = np.zeros(in_features, dtype=bool)
    mask[top_indices] = True

    # Apply pruning
    W_pruned = W_row * mask

    return W_pruned, mask


def select_moderate_positions(mean_js, knee_tolerance=0.0, offset_percent=0.0,
                              percent_change=0.05, debug=False):
    """
    Select moderate positions based on |mean_js|.

    Algorithm:
    1. Sort |mean_js| in descending order
    2. Apply Kneedle on first half to find transition point
    3. Start from knee_point + offset_percent * in_features
    4. Select percent_change * in_features positions from this point

    Args:
        mean_js: James-Stein mean estimates [in_features]
        knee_tolerance: Tolerance for Kneedle algorithm
        offset_percent: Offset from knee point (as fraction of in_features)
        percent_change: Fraction of positions to select
        debug: Print debug information

    Returns:
        selected_indices: Indices of selected moderate positions (in original order)
        knee_info: Dictionary with knee point information
    """
    abs_mean_js = np.abs(mean_js)
    in_features = len(abs_mean_js)

    # Sort descending
    sorted_indices = np.argsort(abs_mean_js)[::-1]
    sorted_values = abs_mean_js[sorted_indices]

    if debug:
        print(f"\n🔍 Selecting moderate positions:")
        print(f"   |mean_js| range: [{abs_mean_js.min():.6f}, {abs_mean_js.max():.6f}]")
        print(f"   Sorted (desc): [{sorted_values[0]:.6f} ... {sorted_values[-1]:.6f}]")

    # Apply Kneedle on first half
    first_half = sorted_values[:in_features // 2]
    knee_idx = find_knee_point(first_half, tolerance_offset=knee_tolerance)

    if debug:
        print(f"   First half size: {len(first_half)}")
        print(f"   Knee index in first half: {knee_idx}/{len(first_half)}")
        print(f"   Knee value: {sorted_values[knee_idx]:.6f}")

    # Start position: knee_point + offset
    offset_indices = int(offset_percent * in_features)
    start_idx = knee_idx + offset_indices
    start_idx = max(0, min(start_idx, in_features - 1))

    # Number to select
    num_to_select = int(percent_change * in_features)
    num_to_select = max(1, num_to_select)

    # End position
    end_idx = start_idx + num_to_select
    end_idx = min(end_idx, in_features)

    # Adjust if we went past the end
    if end_idx == in_features and start_idx > 0:
        start_idx = max(0, in_features - num_to_select)

    actual_num_selected = end_idx - start_idx

    if debug:
        print(f"   Start index: {start_idx} (knee {knee_idx} + offset {offset_indices})")
        print(f"   End index: {end_idx}")
        print(f"   Num selected: {actual_num_selected} ({actual_num_selected/in_features*100:.2f}%)")
        print(f"   Value range: [{sorted_values[start_idx]:.6f}, {sorted_values[end_idx-1]:.6f}]")

    # Get selected indices in original order
    selected_sorted_indices = sorted_indices[start_idx:end_idx]

    knee_info = {
        'knee_idx': knee_idx,
        'knee_value': sorted_values[knee_idx],
        'start_idx': start_idx,
        'end_idx': end_idx,
        'num_selected': actual_num_selected,
        'value_range': (sorted_values[start_idx], sorted_values[end_idx-1]),
    }

    return selected_sorted_indices, knee_info


def correct_weights_moderate(W_orig, W_pruned, mean_js, selected_positions, debug=False):
    """
    Correct pruned weights at moderate positions to minimize reconstruction error.

    Error = (W_pruned - W_orig) · mean_js

    For each moderate position j:
        delta_W[j] = -error * sign(mean_js[j]) / sum(|mean_js[selected]|)

    This distributes the error correction across selected positions,
    with direction determined by sign of mean_js[j] to reduce total error.

    Args:
        W_orig: Original weights [in_features]
        W_pruned: Pruned weights [in_features]
        mean_js: James-Stein mean estimates [in_features]
        selected_positions: Indices of moderate positions to correct
        debug: Print debug information

    Returns:
        W_corrected: Corrected weights
        correction_stats: Dictionary with correction statistics
    """
    if debug:
        print(f"\n🔧 Correcting weights at {len(selected_positions)} moderate positions:")

    # Current reconstruction error
    recon_orig = np.dot(W_orig, mean_js)
    recon_pruned = np.dot(W_pruned, mean_js)
    error = recon_pruned - recon_orig

    if debug:
        print(f"   Original reconstruction: {recon_orig:.6f}")
        print(f"   Pruned reconstruction: {recon_pruned:.6f}")
        print(f"   Error: {error:.6f}")

    # Sum of |mean_js| at selected positions
    sum_abs_mean_js = np.sum(np.abs(mean_js[selected_positions]))

    if debug:
        print(f"   Sum |mean_js[selected]|: {sum_abs_mean_js:.6f}")

    if sum_abs_mean_js < 1e-10:
        if debug:
            print(f"   ⚠️ Sum too small, no correction possible")
        return W_pruned.copy(), None

    # Correct weights
    W_corrected = W_pruned.copy()

    corrections = np.zeros(len(selected_positions))

    for i, j in enumerate(selected_positions):
        # To reduce error, adjust in opposite direction
        # If error > 0 (output too high) and mean_js[j] > 0, decrease W[j]
        # If error > 0 (output too high) and mean_js[j] < 0, increase W[j]
        delta_W = -error * np.sign(mean_js[j]) / sum_abs_mean_js
        W_corrected[j] += delta_W
        corrections[i] = delta_W

    # Verify correction
    recon_corrected = np.dot(W_corrected, mean_js)
    new_error = recon_corrected - recon_orig
    error_reduction = error - new_error

    correction_stats = {
        'error_before': error,
        'error_after': new_error,
        'error_reduction': error_reduction,
        'corrections': corrections,
        'corrections_mean': corrections.mean(),
        'corrections_std': corrections.std(),
        'corrections_min': corrections.min(),
        'corrections_max': corrections.max(),
    }

    if debug:
        print(f"   After correction:")
        print(f"     New reconstruction: {recon_corrected:.6f}")
        print(f"     New error: {new_error:.6f}")
        print(f"     Error reduction: {error_reduction:.6f} ({error_reduction/abs(error)*100:.2f}%)")
        print(f"     Correction magnitude: [{corrections.min():.6f}, {corrections.max():.6f}]")

    return W_corrected, correction_stats


def analyze_reconstruction(W_orig, W_pruned, W_corrected, E_X, JS_mean_X):
    """
    Analyze reconstruction quality with both original mean and JS mean.

    Returns:
        Dictionary with comprehensive analysis
    """
    results = {}

    # === With JS mean ===
    recon_orig_js = np.dot(W_orig, JS_mean_X)
    recon_pruned_js = np.dot(W_pruned, JS_mean_X)
    recon_corrected_js = np.dot(W_corrected, JS_mean_X)

    error_pruned_js = recon_pruned_js - recon_orig_js
    error_corrected_js = recon_corrected_js - recon_orig_js
    improvement_js = error_pruned_js - error_corrected_js

    results['js_mean'] = {
        'original': recon_orig_js,
        'pruned': recon_pruned_js,
        'corrected': recon_corrected_js,
        'error_pruned': error_pruned_js,
        'error_corrected': error_corrected_js,
        'improvement': improvement_js,
        'improvement_pct': (improvement_js / abs(error_pruned_js) * 100) if abs(error_pruned_js) > 1e-10 else 0,
    }

    # === With original mean ===
    recon_orig_mean = np.dot(W_orig, E_X)
    recon_pruned_mean = np.dot(W_pruned, E_X)
    recon_corrected_mean = np.dot(W_corrected, E_X)

    error_pruned_mean = recon_pruned_mean - recon_orig_mean
    error_corrected_mean = recon_corrected_mean - recon_orig_mean
    improvement_mean = error_pruned_mean - error_corrected_mean

    results['original_mean'] = {
        'original': recon_orig_mean,
        'pruned': recon_pruned_mean,
        'corrected': recon_corrected_mean,
        'error_pruned': error_pruned_mean,
        'error_corrected': error_corrected_mean,
        'improvement': improvement_mean,
        'improvement_pct': (improvement_mean / abs(error_pruned_mean) * 100) if abs(error_pruned_mean) > 1e-10 else 0,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Moderate weight correction for Wanda pruning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", type=str, default="./exported_wanda_data",
                       help="Directory containing exported data from export_wd.py")
    parser.add_argument("--sparsity", type=float, default=0.5,
                       help="Fraction of weights to prune (0.5 = 50%%)")
    parser.add_argument("--out-channel-id", type=int, default=0,
                       help="Which output channel to analyze")
    parser.add_argument("--knee-tolerance", type=float, default=0.0,
                       help="Tolerance for Kneedle algorithm")
    parser.add_argument("--offset-percent", type=float, default=0.0,
                       help="Offset from knee point (as %% of in_features)")
    parser.add_argument("--percent-change", type=float, default=0.05,
                       help="Percentage of in_features to select for correction (default 5%%)")
    parser.add_argument("--output-dir", type=str, default="./moderate_correction",
                       help="Output directory for results")
    args = parser.parse_args()

    print("=" * 80)
    print("MODERATE WEIGHT CORRECTION FOR WANDA PRUNING")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Sparsity: {args.sparsity * 100:.1f}%")
    print(f"Output channel: {args.out_channel_id}")
    print(f"Knee tolerance: {args.knee_tolerance}")
    print(f"Offset percent: {args.offset_percent * 100:.1f}%")
    print(f"Percent to correct: {args.percent_change * 100:.1f}%")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Load data
    data = load_exported_data(args.input_dir)

    print(f"\n📊 Loaded data:")
    print(f"   Layer: {data['layer_name']}")
    print(f"   Weight matrix W: [{data['out_features']}, {data['in_features']}]")

    W = data['W']
    E_X = data['E_X']
    L2_norm_X = data['L2_norm_X']
    JS_mean_X = data['JS_mean_X']

    out_channel_id = args.out_channel_id

    if out_channel_id >= data['out_features']:
        print(f"\n❌ Error: out_channel_id {out_channel_id} >= out_features {data['out_features']}")
        return

    # === STEP 1: Wanda Pruning ===
    print(f"\n{'='*60}")
    print("STEP 1: Wanda Pruning")
    print('='*60)

    W_orig = W[out_channel_id, :]
    scores = compute_wanda_scores(W_orig, L2_norm_X)
    W_pruned, prune_mask = prune_weights_wanda(W_orig, scores, args.sparsity)

    num_pruned = (~prune_mask).sum()
    print(f"Pruned {num_pruned}/{len(W_orig)} weights ({num_pruned/len(W_orig)*100:.1f}%)")

    # === STEP 2: Select Moderate Positions ===
    print(f"\n{'='*60}")
    print("STEP 2: Select Moderate Positions")
    print('='*60)

    moderate_positions, knee_info = select_moderate_positions(
        JS_mean_X,
        knee_tolerance=args.knee_tolerance,
        offset_percent=args.offset_percent,
        percent_change=args.percent_change,
        debug=True
    )

    print(f"\nSelected {len(moderate_positions)} moderate positions")

    # === STEP 3: Correct Weights ===
    print(f"\n{'='*60}")
    print("STEP 3: Correct Weights at Moderate Positions")
    print('='*60)

    W_corrected, correction_stats = correct_weights_moderate(
        W_orig, W_pruned, JS_mean_X, moderate_positions, debug=True
    )

    # === STEP 4: Analyze Reconstruction ===
    print(f"\n{'='*60}")
    print("STEP 4: Analyze Reconstruction Quality")
    print('='*60)

    analysis = analyze_reconstruction(W_orig, W_pruned, W_corrected, E_X, JS_mean_X)

    print(f"\n📊 Reconstruction with JS Mean:")
    print(f"   Original: {analysis['js_mean']['original']:.6f}")
    print(f"   After pruning: {analysis['js_mean']['pruned']:.6f}")
    print(f"   After correction: {analysis['js_mean']['corrected']:.6f}")
    print(f"   Error (pruned): {analysis['js_mean']['error_pruned']:.6f}")
    print(f"   Error (corrected): {analysis['js_mean']['error_corrected']:.6f}")
    print(f"   Improvement: {analysis['js_mean']['improvement']:.6f} ({analysis['js_mean']['improvement_pct']:.2f}%)")

    print(f"\n📊 Reconstruction with Original Mean:")
    print(f"   Original: {analysis['original_mean']['original']:.6f}")
    print(f"   After pruning: {analysis['original_mean']['pruned']:.6f}")
    print(f"   After correction: {analysis['original_mean']['corrected']:.6f}")
    print(f"   Error (pruned): {analysis['original_mean']['error_pruned']:.6f}")
    print(f"   Error (corrected): {analysis['original_mean']['error_corrected']:.6f}")
    print(f"   Improvement: {analysis['original_mean']['improvement']:.6f} ({analysis['original_mean']['improvement_pct']:.2f}%)")

    # === Save Results ===
    os.makedirs(args.output_dir, exist_ok=True)

    # Save summary
    summary_path = os.path.join(args.output_dir, f"summary_ch{out_channel_id}.txt")
    with open(summary_path, 'w') as f:
        f.write("Moderate Weight Correction for Wanda Pruning\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Layer: {data['layer_name']}\n")
        f.write(f"Output channel: {out_channel_id}\n")
        f.write(f"Sparsity: {args.sparsity * 100:.1f}%\n")
        f.write(f"Moderate selection: {args.percent_change * 100:.1f}% of weights\n\n")

        f.write("Moderate Position Selection:\n")
        f.write(f"  Knee index: {knee_info['knee_idx']}\n")
        f.write(f"  Knee value: {knee_info['knee_value']:.6f}\n")
        f.write(f"  Selected positions: {knee_info['num_selected']}\n")
        f.write(f"  Value range: [{knee_info['value_range'][0]:.6f}, {knee_info['value_range'][1]:.6f}]\n\n")

        f.write("Reconstruction with JS Mean:\n")
        f.write(f"  Original: {analysis['js_mean']['original']:.6f}\n")
        f.write(f"  After pruning: {analysis['js_mean']['pruned']:.6f}\n")
        f.write(f"  After correction: {analysis['js_mean']['corrected']:.6f}\n")
        f.write(f"  Error reduction: {analysis['js_mean']['improvement']:.6f} ({analysis['js_mean']['improvement_pct']:.2f}%)\n\n")

        f.write("Reconstruction with Original Mean:\n")
        f.write(f"  Original: {analysis['original_mean']['original']:.6f}\n")
        f.write(f"  After pruning: {analysis['original_mean']['pruned']:.6f}\n")
        f.write(f"  After correction: {analysis['original_mean']['corrected']:.6f}\n")
        f.write(f"  Error reduction: {analysis['original_mean']['improvement']:.6f} ({analysis['original_mean']['improvement_pct']:.2f}%)\n")

    print(f"\n✅ Saved summary: {summary_path}")

    # Save detailed CSV
    csv_path = os.path.join(args.output_dir, f"detailed_ch{out_channel_id}.csv")
    df = pd.DataFrame({
        'input_channel_id': np.arange(len(W_orig)),
        'W_original': W_orig,
        'W_pruned': W_pruned,
        'W_corrected': W_corrected,
        'pruned': ~prune_mask,
        'moderate': np.isin(np.arange(len(W_orig)), moderate_positions),
        'wanda_score': scores,
        'L2_norm': L2_norm_X,
        'E[X]': E_X,
        'JS_mean[X]': JS_mean_X,
        'abs_JS_mean': np.abs(JS_mean_X),
    })
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved detailed CSV: {csv_path}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Key findings:")
    print(f"  JS Mean - Error reduction: {analysis['js_mean']['improvement_pct']:.2f}%")
    print(f"  Original Mean - Error reduction: {analysis['original_mean']['improvement_pct']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
