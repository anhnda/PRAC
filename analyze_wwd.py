"""
Analyze Wanda Pruning on Weight-Activation Data

This script analyzes the effect of Wanda pruning on a single output channel:
1. Loads exported data from export_wd.py
2. Computes Wanda scores: Score_ij = |W_ij| * ||X_j||_2
3. Prunes weights for output channel 0 by zeroing 50% with lowest scores
4. Compares reconstruction loss: W[0,:] · E[X] vs W_pruned[0,:] · E[X]

Usage:
    python analyze_wwd.py --input-dir ./exported_wanda_data --sparsity 0.5
"""

import numpy as np
import pandas as pd
import argparse
import os
import glob
import matplotlib.pyplot as plt


def load_exported_data(input_dir):
    """
    Load data exported by export_wd.py.

    Returns:
        dict with keys: W, E_X, L2_norm_X, JS_mean_X, metadata
    """
    # Find the complete NPZ file
    npz_files = glob.glob(os.path.join(input_dir, "*_complete.npz"))

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No *_complete.npz file found in {input_dir}")

    npz_path = npz_files[0]
    print(f"Loading data from: {npz_path}")

    data = np.load(npz_path)

    result = {
        'W': data['W'],  # [out_features, in_features]
        'E_X': data['E_X'],  # [in_features]
        'L2_norm_X': data['L2_norm_X'],  # [in_features]
        'JS_mean_X': data['JS_mean_X'],  # [in_features]
        'layer_name': str(data['layer_name']),
        'out_features': int(data['out_features']),
        'in_features': int(data['in_features']),
        'grand_mean': float(data['grand_mean']),
        'shrinkage_amount': float(data['shrinkage_amount']),
    }

    return result


def compute_wanda_scores(W_row, L2_norm_X):
    """
    Compute Wanda scores for a single output channel (row of W).

    Score_j = |W_j| * ||X_j||_2

    Args:
        W_row: Weight row [in_features]
        L2_norm_X: L2 norms [in_features]

    Returns:
        scores: Wanda scores [in_features]
    """
    scores = np.abs(W_row) * L2_norm_X
    return scores


def prune_weights_wanda(W_row, scores, sparsity):
    """
    Prune weights based on Wanda scores.

    Args:
        W_row: Weight row [in_features]
        scores: Wanda scores [in_features]
        sparsity: Fraction of weights to prune (e.g., 0.5 = prune 50%)

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


def analyze_reconstruction(W_original, W_pruned, E_X, out_channel_id=0):
    """
    Compare reconstruction loss before and after pruning.

    Reconstruction: output = W[out_channel_id, :] · E[X]

    Args:
        W_original: Original weights [out_features, in_features]
        W_pruned: Pruned weights [out_features, in_features]
        E_X: Mean activations [in_features]
        out_channel_id: Which output channel to analyze

    Returns:
        dict with analysis results
    """
    # Extract the row for the target output channel
    w_orig = W_original[out_channel_id, :]
    w_prun = W_pruned[out_channel_id, :]

    # Compute reconstructions
    recon_original = np.dot(w_orig, E_X)
    recon_pruned = np.dot(w_prun, E_X)

    # Compute losses
    absolute_error = np.abs(recon_original - recon_pruned)
    squared_error = (recon_original - recon_pruned) ** 2
    relative_error = absolute_error / (np.abs(recon_original) + 1e-10)

    results = {
        'recon_original': recon_original,
        'recon_pruned': recon_pruned,
        'absolute_error': absolute_error,
        'squared_error': squared_error,
        'relative_error': relative_error,
        'w_orig': w_orig,
        'w_prun': w_prun,
    }

    return results


def visualize_pruning(W_original, W_pruned, scores, mask, E_X, L2_norm_X,
                      out_channel_id, output_dir):
    """
    Create visualizations of pruning analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Wanda Pruning Analysis - Output Channel {out_channel_id}',
                 fontsize=16, fontweight='bold')

    w_orig = W_original[out_channel_id, :]
    w_prun = W_pruned[out_channel_id, :]
    in_features = len(w_orig)
    channel_ids = np.arange(in_features)

    # 1. Original vs Pruned Weights
    ax = axes[0, 0]
    ax.scatter(channel_ids[mask], w_orig[mask], alpha=0.6, s=10,
               label='Kept', color='blue')
    ax.scatter(channel_ids[~mask], w_orig[~mask], alpha=0.6, s=10,
               label='Pruned', color='red')
    ax.set_xlabel('Input Channel ID')
    ax.set_ylabel('Weight Value')
    ax.set_title('Original Weights (Color = Pruned/Kept)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Wanda Scores Distribution
    ax = axes[0, 1]
    ax.hist(scores, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.percentile(scores, 50), color='red', linestyle='--',
               linewidth=2, label='50th percentile (threshold)')
    ax.set_xlabel('Wanda Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Wanda Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. L2 Norms
    ax = axes[0, 2]
    ax.scatter(channel_ids[mask], L2_norm_X[mask], alpha=0.6, s=10,
               label='Kept', color='blue')
    ax.scatter(channel_ids[~mask], L2_norm_X[~mask], alpha=0.6, s=10,
               label='Pruned', color='red')
    ax.set_xlabel('Input Channel ID')
    ax.set_ylabel('L2 Norm ||X_j||_2')
    ax.set_title('Activation L2 Norms (Color = Pruned/Kept)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Weight Magnitude vs L2 Norm (Wanda components)
    ax = axes[1, 0]
    scatter = ax.scatter(L2_norm_X, np.abs(w_orig), c=scores,
                        cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('L2 Norm ||X_j||_2')
    ax.set_ylabel('Weight Magnitude |W_ij|')
    ax.set_title('Wanda Score Components')
    plt.colorbar(scatter, ax=ax, label='Wanda Score')
    ax.grid(True, alpha=0.3)

    # 5. Contribution to output: W_j * E[X_j]
    contrib_orig = w_orig * E_X
    contrib_prun = w_prun * E_X

    ax = axes[1, 1]
    ax.bar(channel_ids[mask], contrib_orig[mask], alpha=0.7,
           label='Kept', color='blue', width=1.0)
    ax.bar(channel_ids[~mask], contrib_orig[~mask], alpha=0.7,
           label='Pruned (Lost)', color='red', width=1.0)
    ax.set_xlabel('Input Channel ID')
    ax.set_ylabel('Contribution: W_j × E[X_j]')
    ax.set_title('Per-Channel Contribution to Output')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Cumulative contribution
    ax = axes[1, 2]
    sorted_indices = np.argsort(scores)[::-1]  # High to low
    cumsum_orig = np.cumsum(np.abs(contrib_orig[sorted_indices]))
    total_contrib = cumsum_orig[-1]
    cumsum_pct = (cumsum_orig / total_contrib) * 100

    ax.plot(np.arange(in_features), cumsum_pct, linewidth=2, color='purple')
    ax.axvline(in_features * 0.5, color='red', linestyle='--',
               linewidth=2, label=f'50% of channels\n({cumsum_pct[int(in_features*0.5)]:.1f}% contrib)')
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Top Channels (by Wanda score)')
    ax.set_ylabel('Cumulative Contribution (%)')
    ax.set_title('Cumulative Contribution by Top Channels')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(output_dir, f'wanda_pruning_analysis_ch{out_channel_id}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization: {plot_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Wanda pruning on exported weight-activation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", type=str, default="./exported_wanda_data",
                       help="Directory containing exported data from export_wd.py")
    parser.add_argument("--sparsity", type=float, default=0.5,
                       help="Fraction of weights to prune (0.5 = 50%%)")
    parser.add_argument("--out-channel-id", type=int, default=0,
                       help="Which output channel to analyze")
    parser.add_argument("--output-dir", type=str, default="./wanda_analysis",
                       help="Output directory for analysis results")
    args = parser.parse_args()

    print("=" * 80)
    print("WANDA PRUNING ANALYSIS")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Sparsity target: {args.sparsity * 100:.1f}% (prune {args.sparsity*100:.1f}%, keep {(1-args.sparsity)*100:.1f}%)")
    print(f"Output channel: {args.out_channel_id}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Load data
    data = load_exported_data(args.input_dir)

    print(f"\n📊 Loaded data:")
    print(f"   Layer: {data['layer_name']}")
    print(f"   Weight matrix W: [{data['out_features']}, {data['in_features']}]")
    print(f"   Analyzing output channel: {args.out_channel_id}")

    # Extract data
    W = data['W']
    E_X = data['E_X']
    L2_norm_X = data['L2_norm_X']
    JS_mean_X = data['JS_mean_X']

    out_channel_id = args.out_channel_id

    # Validate output channel
    if out_channel_id >= data['out_features']:
        print(f"\n❌ Error: out_channel_id {out_channel_id} >= out_features {data['out_features']}")
        return

    # Get weight row for the target output channel
    W_row = W[out_channel_id, :]

    print(f"\n🔍 Computing Wanda scores...")
    # Compute Wanda scores: Score_j = |W_j| * ||X_j||_2
    scores = compute_wanda_scores(W_row, L2_norm_X)

    print(f"   Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"   Score mean: {scores.mean():.6f}")
    print(f"   Score median: {np.median(scores):.6f}")

    # Prune weights
    print(f"\n✂️  Pruning weights (sparsity = {args.sparsity * 100:.1f}%)...")
    W_row_pruned, mask = prune_weights_wanda(W_row, scores, args.sparsity)

    num_kept = mask.sum()
    num_pruned = len(mask) - num_kept
    actual_sparsity = num_pruned / len(mask)

    print(f"   Kept: {num_kept} channels")
    print(f"   Pruned: {num_pruned} channels")
    print(f"   Actual sparsity: {actual_sparsity * 100:.2f}%")

    # Create pruned weight matrix (only this channel is pruned)
    W_pruned = W.copy()
    W_pruned[out_channel_id, :] = W_row_pruned

    # Analyze reconstruction
    print(f"\n📐 Analyzing reconstruction loss...")
    analysis = analyze_reconstruction(W, W_pruned, E_X, out_channel_id)

    print(f"\n📊 Reconstruction Results:")
    print(f"   Original output: {analysis['recon_original']:.6f}")
    print(f"   Pruned output: {analysis['recon_pruned']:.6f}")
    print(f"   Absolute error: {analysis['absolute_error']:.6f}")
    print(f"   Squared error: {analysis['squared_error']:.6f}")
    print(f"   Relative error: {analysis['relative_error'] * 100:.4f}%")

    # Additional analysis with James-Stein mean
    print(f"\n🔬 Reconstruction with James-Stein mean:")
    recon_orig_js = np.dot(W_row, JS_mean_X)
    recon_prun_js = np.dot(W_row_pruned, JS_mean_X)
    error_js = np.abs(recon_orig_js - recon_prun_js)
    rel_error_js = error_js / (np.abs(recon_orig_js) + 1e-10)

    print(f"   Original (JS): {recon_orig_js:.6f}")
    print(f"   Pruned (JS): {recon_prun_js:.6f}")
    print(f"   Absolute error (JS): {error_js:.6f}")
    print(f"   Relative error (JS): {rel_error_js * 100:.4f}%")

    # Compare standard mean vs JS mean
    print(f"\n📈 E[X] vs James-Stein E[X] comparison:")
    print(f"   Difference in original recon: {np.abs(analysis['recon_original'] - recon_orig_js):.6f}")
    print(f"   Difference in pruned recon: {np.abs(analysis['recon_pruned'] - recon_prun_js):.6f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save numerical results
    results_path = os.path.join(args.output_dir, f"results_ch{out_channel_id}.txt")
    with open(results_path, 'w') as f:
        f.write(f"Wanda Pruning Analysis Results\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Layer: {data['layer_name']}\n")
        f.write(f"Output channel: {out_channel_id}\n")
        f.write(f"Sparsity: {args.sparsity * 100:.1f}%\n")
        f.write(f"\nPruning Statistics:\n")
        f.write(f"  Channels kept: {num_kept}/{len(mask)}\n")
        f.write(f"  Channels pruned: {num_pruned}/{len(mask)}\n")
        f.write(f"  Actual sparsity: {actual_sparsity * 100:.2f}%\n")
        f.write(f"\nReconstruction with E[X]:\n")
        f.write(f"  Original output: {analysis['recon_original']:.6f}\n")
        f.write(f"  Pruned output: {analysis['recon_pruned']:.6f}\n")
        f.write(f"  Absolute error: {analysis['absolute_error']:.6f}\n")
        f.write(f"  Squared error: {analysis['squared_error']:.6f}\n")
        f.write(f"  Relative error: {analysis['relative_error'] * 100:.4f}%\n")
        f.write(f"\nReconstruction with James-Stein E[X]:\n")
        f.write(f"  Original output: {recon_orig_js:.6f}\n")
        f.write(f"  Pruned output: {recon_prun_js:.6f}\n")
        f.write(f"  Absolute error: {error_js:.6f}\n")
        f.write(f"  Relative error: {rel_error_js * 100:.4f}%\n")
        f.write(f"\nWanda Score Statistics:\n")
        f.write(f"  Min: {scores.min():.6f}\n")
        f.write(f"  Max: {scores.max():.6f}\n")
        f.write(f"  Mean: {scores.mean():.6f}\n")
        f.write(f"  Median: {np.median(scores):.6f}\n")
        f.write(f"  Std: {scores.std():.6f}\n")

    print(f"\n✅ Saved results: {results_path}")

    # Save detailed CSV
    csv_path = os.path.join(args.output_dir, f"analysis_ch{out_channel_id}.csv")
    df = pd.DataFrame({
        'input_channel_id': np.arange(len(W_row)),
        'W_original': W_row,
        'W_pruned': W_row_pruned,
        'kept': mask,
        'wanda_score': scores,
        'L2_norm': L2_norm_X,
        'E[X]': E_X,
        'JS_mean[X]': JS_mean_X,
        'contribution_orig': W_row * E_X,
        'contribution_pruned': W_row_pruned * E_X,
    })
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved detailed CSV: {csv_path}")

    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    visualize_pruning(W, W_pruned, scores, mask, E_X, L2_norm_X,
                      out_channel_id, args.output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print(f"\nKey finding:")
    print(f"  Pruning {args.sparsity*100:.1f}% of weights causes:")
    print(f"    {analysis['relative_error']*100:.4f}% relative error in reconstruction")
    print("=" * 80)


if __name__ == "__main__":
    main()
