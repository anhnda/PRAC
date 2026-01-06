"""
Visualize James-Stein means and GQA weight distributions from xspot.py output.

This script creates comprehensive visualizations of:
1. James-Stein channel importance distribution
2. Q/K/V weight distributions (overall and per-head)
3. Weight magnitude comparisons across Q, K, V
4. Per-head analysis for query weights

Usage:
    python visualize_xspot.py --data-dir ./xspot_layer0_group0
    python visualize_xspot.py --data-dir ./xspot_layer5_group2 --output-dir ./viz_layer5_group2
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)


def load_data(data_dir):
    """Load exported data from xspot.py."""
    data_dir = Path(data_dir)

    print(f"Loading data from: {data_dir}")

    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Extract group_id from metadata
    group_id = metadata['group_id']

    # Load James-Stein means
    js_means = np.load(data_dir / 'js_means.npy')

    # Load naive means for comparison
    naive_means = np.load(data_dir / 'naive_means.npy')

    # Load Q/K/V weights
    Wq = np.load(data_dir / f'Wq_group{group_id}.npy')
    Wk = np.load(data_dir / f'Wk_group{group_id}.npy')
    Wv = np.load(data_dir / f'Wv_group{group_id}.npy')

    print(f"\nData shapes:")
    print(f"  JS means: {js_means.shape}")
    print(f"  Naive means: {naive_means.shape}")
    print(f"  Wq (group {group_id}): {Wq.shape}")
    print(f"  Wk (group {group_id}): {Wk.shape}")
    print(f"  Wv (group {group_id}): {Wv.shape}")

    return {
        'js_means': js_means,
        'naive_means': naive_means,
        'Wq': Wq,
        'Wk': Wk,
        'Wv': Wv,
        'metadata': metadata
    }


def plot_js_means_distribution(data, output_dir):
    """Plot James-Stein means distribution."""
    js_means = data['js_means']
    naive_means = data['naive_means']
    metadata = data['metadata']

    # Compute statistics for comparison
    stats_naive = {
        'min': naive_means.min(),
        'max': naive_means.max(),
        'mean': naive_means.mean(),
        'median': np.median(naive_means),
        'std': naive_means.std()
    }

    stats_js = {
        'min': js_means.min(),
        'max': js_means.max(),
        'mean': js_means.mean(),
        'median': np.median(js_means),
        'std': js_means.std()
    }

    # Range reduction
    range_naive = stats_naive['max'] - stats_naive['min']
    range_js = stats_js['max'] - stats_js['min']
    range_reduction = (range_naive - range_js) / range_naive * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram comparison
    ax = axes[0, 0]
    ax.hist(naive_means, bins=50, alpha=0.6, label='Naive means', color='blue', density=True)
    ax.hist(js_means, bins=50, alpha=0.6, label='James-Stein means', color='red', density=True)
    ax.axvline(metadata['james_stein']['grand_mean'], color='green', linestyle='--',
               linewidth=2, label=f'Grand mean: {metadata["james_stein"]["grand_mean"]:.4f}')
    ax.set_xlabel('Channel Mean Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Channel Means (Naive vs James-Stein)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Sorted comparison
    ax = axes[0, 1]
    sorted_naive = np.sort(naive_means)
    sorted_js = np.sort(js_means)
    channels = np.arange(len(js_means))
    ax.plot(channels, sorted_naive, label='Naive means (sorted)', alpha=0.7, linewidth=1.5)
    ax.plot(channels, sorted_js, label='James-Stein means (sorted)', alpha=0.7, linewidth=1.5)
    ax.axhline(metadata['james_stein']['grand_mean'], color='green', linestyle='--',
               linewidth=2, label=f'Grand mean: {metadata["james_stein"]["grand_mean"]:.4f}')
    ax.set_xlabel('Channel Index (sorted)')
    ax.set_ylabel('Mean Value')
    ax.set_title('Sorted Channel Means')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Difference distribution
    ax = axes[1, 0]
    diff = js_means - naive_means
    ax.hist(diff, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No shrinkage')
    ax.axvline(diff.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean diff: {diff.mean():.6f}')
    ax.set_xlabel('Shrinkage (JS - Naive)')
    ax.set_ylabel('Count')
    ax.set_title(f'Shrinkage Distribution (λ={metadata["james_stein"]["shrinkage_factor"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Scatter plot with statistics table
    ax = axes[1, 1]
    ax.scatter(naive_means, js_means, alpha=0.3, s=10)
    ax.plot([naive_means.min(), naive_means.max()],
            [naive_means.min(), naive_means.max()],
            'r--', linewidth=2, label='No shrinkage (y=x)')
    # Plot shrinkage line towards grand mean
    grand_mean = metadata['james_stein']['grand_mean']
    ax.axhline(grand_mean, color='green', linestyle='--', alpha=0.5, label='Grand mean')
    ax.axvline(grand_mean, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Naive Mean')
    ax.set_ylabel('James-Stein Mean')
    ax.set_title('Shrinkage Scatter Plot')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add statistics table
    stats_text = f"""Statistics Comparison:

                Naive        JS          Diff
Min:      {stats_naive['min']:8.5f}  {stats_js['min']:8.5f}  {stats_js['min']-stats_naive['min']:+8.5f}
Max:      {stats_naive['max']:8.5f}  {stats_js['max']:8.5f}  {stats_js['max']-stats_naive['max']:+8.5f}
Mean:     {stats_naive['mean']:8.5f}  {stats_js['mean']:8.5f}  {stats_js['mean']-stats_naive['mean']:+8.5f}
Median:   {stats_naive['median']:8.5f}  {stats_js['median']:8.5f}  {stats_js['median']-stats_naive['median']:+8.5f}
Std:      {stats_naive['std']:8.5f}  {stats_js['std']:8.5f}  {stats_js['std']-stats_naive['std']:+8.5f}

Range:    {range_naive:8.5f}  {range_js:8.5f}  ({range_reduction:+.2f}%)
"""
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    output_path = output_dir / 'js_means_distribution.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Return statistics for use in summary report
    return {
        'naive': stats_naive,
        'js': stats_js,
        'range_reduction': range_reduction
    }


def plot_weight_distributions(data, output_dir):
    """Plot Q/K/V weight distributions."""
    Wq = data['Wq']  # [queries_per_group, head_dim, hidden_size]
    Wk = data['Wk']  # [head_dim, hidden_size]
    Wv = data['Wv']  # [head_dim, hidden_size]
    metadata = data['metadata']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Flatten weights for overall distribution
    wq_flat = Wq.flatten()
    wk_flat = Wk.flatten()
    wv_flat = Wv.flatten()

    # 1. Overall weight distributions (histograms)
    ax = axes[0, 0]
    ax.hist(wq_flat, bins=100, alpha=0.6, label=f'Wq (μ={wq_flat.mean():.4f}, σ={wq_flat.std():.4f})',
            color='blue', density=True)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Query Weight Distribution\n(Group {metadata["group_id"]}, {metadata["group_info"]["num_query_heads_in_group"]} heads)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(wk_flat, bins=100, alpha=0.6, label=f'Wk (μ={wk_flat.mean():.4f}, σ={wk_flat.std():.4f})',
            color='green', density=True)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Key Weight Distribution\n(Group {metadata["group_id"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.hist(wv_flat, bins=100, alpha=0.6, label=f'Wv (μ={wv_flat.mean():.4f}, σ={wv_flat.std():.4f})',
            color='red', density=True)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Value Weight Distribution\n(Group {metadata["group_id"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Combined comparison
    ax = axes[1, 0]
    ax.hist(wq_flat, bins=100, alpha=0.5, label='Wq', color='blue', density=True)
    ax.hist(wk_flat, bins=100, alpha=0.5, label='Wk', color='green', density=True)
    ax.hist(wv_flat, bins=100, alpha=0.5, label='Wv', color='red', density=True)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title('Q/K/V Weight Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Box plot comparison
    ax = axes[1, 1]
    box_data = [wq_flat, wk_flat, wv_flat]
    bp = ax.boxplot(box_data, labels=['Wq', 'Wk', 'Wv'], patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Weight Value')
    ax.set_title('Weight Distribution Statistics')
    ax.grid(True, alpha=0.3)

    # 4. Magnitude statistics
    ax = axes[1, 2]
    metrics = ['Mean', 'Std', 'L2 Norm']
    q_stats = [np.abs(wq_flat).mean(), wq_flat.std(), np.linalg.norm(wq_flat)]
    k_stats = [np.abs(wk_flat).mean(), wk_flat.std(), np.linalg.norm(wk_flat)]
    v_stats = [np.abs(wv_flat).mean(), wv_flat.std(), np.linalg.norm(wv_flat)]

    x = np.arange(len(metrics))
    width = 0.25

    ax.bar(x - width, q_stats, width, label='Wq', color='blue', alpha=0.7)
    ax.bar(x, k_stats, width, label='Wk', color='green', alpha=0.7)
    ax.bar(x + width, v_stats, width, label='Wv', color='red', alpha=0.7)

    ax.set_ylabel('Value')
    ax.set_title('Weight Statistics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'weight_distributions.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_per_head_analysis(data, output_dir):
    """Plot per-head analysis for query weights."""
    Wq = data['Wq']  # [queries_per_group, head_dim, hidden_size]
    metadata = data['metadata']

    num_heads = Wq.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Per-head weight distributions
    ax = axes[0, 0]
    for head_idx in range(num_heads):
        wq_head = Wq[head_idx].flatten()
        ax.hist(wq_head, bins=50, alpha=0.5,
                label=f'Head {head_idx} (μ={wq_head.mean():.4f})', density=True)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title('Per-Head Query Weight Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Per-head statistics
    ax = axes[0, 1]
    head_means = [Wq[i].mean() for i in range(num_heads)]
    head_stds = [Wq[i].std() for i in range(num_heads)]
    head_l2 = [np.linalg.norm(Wq[i]) for i in range(num_heads)]

    x = np.arange(num_heads)
    width = 0.25

    ax.bar(x - width, head_means, width, label='Mean', alpha=0.7)
    ax.bar(x, head_stds, width, label='Std', alpha=0.7)
    ax.bar(x + width, [l2/1000 for l2 in head_l2], width, label='L2/1000', alpha=0.7)

    ax.set_xlabel('Query Head Index')
    ax.set_ylabel('Value')
    ax.set_title('Per-Head Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels([f'H{i}' for i in range(num_heads)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Weight magnitude heatmap (per head, averaged over head_dim)
    ax = axes[1, 0]
    # Average over head_dim to get [num_heads, hidden_size]
    wq_avg_headcfim = np.abs(Wq).mean(axis=1)
    im = ax.imshow(wq_avg_headcfim, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Input Channel (hidden_size)')
    ax.set_ylabel('Query Head')
    ax.set_title('Query Weight Magnitude (averaged over head_dim)')
    ax.set_yticks(range(num_heads))
    ax.set_yticklabels([f'Head {i}' for i in range(num_heads)])
    plt.colorbar(im, ax=ax, label='|Weight|')

    # 4. Column-wise L2 norms (importance per input channel)
    ax = axes[1, 1]
    # For each input channel, compute L2 norm across all query heads
    # Wq shape: [num_heads, head_dim, hidden_size]
    # Reshape to [num_heads * head_dim, hidden_size] and compute column L2
    wq_reshaped = Wq.reshape(-1, Wq.shape[2])  # [num_heads*head_dim, hidden_size]
    col_l2_norms = np.linalg.norm(wq_reshaped, axis=0)  # [hidden_size]

    # Plot sorted L2 norms
    sorted_norms = np.sort(col_l2_norms)[::-1]  # Descending
    ax.plot(sorted_norms, linewidth=1.5)
    ax.set_xlabel('Input Channel (sorted by importance)')
    ax.set_ylabel('L2 Norm')
    ax.set_title('Query Weight Column L2 Norms (Input Channel Importance)')
    ax.grid(True, alpha=0.3)

    # Add percentile markers
    for pct in [90, 95, 99]:
        idx = int(len(sorted_norms) * pct / 100)
        ax.axvline(idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(idx, sorted_norms[idx], f'{pct}%', fontsize=8, color='red')

    plt.tight_layout()
    output_path = output_dir / 'per_head_analysis.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_kv_analysis(data, output_dir):
    """Plot K/V weight analysis."""
    Wk = data['Wk']  # [head_dim, hidden_size]
    Wv = data['Wv']  # [head_dim, hidden_size]
    metadata = data['metadata']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. K weight heatmap
    ax = axes[0, 0]
    im = ax.imshow(np.abs(Wk), aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Input Channel (hidden_size)')
    ax.set_ylabel('Head Dimension')
    ax.set_title(f'Key Weight Magnitude (Group {metadata["group_id"]})')
    plt.colorbar(im, ax=ax, label='|Weight|')

    # 2. V weight heatmap
    ax = axes[0, 1]
    im = ax.imshow(np.abs(Wv), aspect='auto', cmap='plasma', interpolation='nearest')
    ax.set_xlabel('Input Channel (hidden_size)')
    ax.set_ylabel('Head Dimension')
    ax.set_title(f'Value Weight Magnitude (Group {metadata["group_id"]})')
    plt.colorbar(im, ax=ax, label='|Weight|')

    # 3. K/V column L2 norms
    ax = axes[1, 0]
    k_col_norms = np.linalg.norm(Wk, axis=0)  # [hidden_size]
    v_col_norms = np.linalg.norm(Wv, axis=0)  # [hidden_size]

    sorted_k = np.sort(k_col_norms)[::-1]
    sorted_v = np.sort(v_col_norms)[::-1]

    channels = np.arange(len(k_col_norms))
    ax.plot(sorted_k, label='Key (sorted)', alpha=0.7, linewidth=1.5)
    ax.plot(sorted_v, label='Value (sorted)', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Input Channel (sorted by importance)')
    ax.set_ylabel('L2 Norm')
    ax.set_title('K/V Column L2 Norms (Input Channel Importance)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. K vs V correlation
    ax = axes[1, 1]
    ax.scatter(k_col_norms, v_col_norms, alpha=0.3, s=10)

    # Compute correlation
    correlation = np.corrcoef(k_col_norms, v_col_norms)[0, 1]

    # Plot diagonal
    max_val = max(k_col_norms.max(), v_col_norms.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='y=x')

    ax.set_xlabel('Key L2 Norm')
    ax.set_ylabel('Value L2 Norm')
    ax.set_title(f'K vs V Column Importance (Correlation: {correlation:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'kv_analysis.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_importance(data, output_dir):
    """Plot combined importance analysis (JS means + weight norms)."""
    js_means = data['js_means']
    Wq = data['Wq']
    Wk = data['Wk']
    Wv = data['Wv']
    metadata = data['metadata']

    # Compute column L2 norms for each weight matrix
    wq_reshaped = Wq.reshape(-1, Wq.shape[2])
    q_importance = np.linalg.norm(wq_reshaped, axis=0)
    k_importance = np.linalg.norm(Wk, axis=0)
    v_importance = np.linalg.norm(Wv, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. JS means vs Q importance
    ax = axes[0, 0]
    ax.scatter(js_means, q_importance, alpha=0.3, s=10)
    correlation_q = np.corrcoef(js_means, q_importance)[0, 1]
    ax.set_xlabel('James-Stein Mean (Activation)')
    ax.set_ylabel('Query L2 Norm (Weight)')
    ax.set_title(f'Activation vs Query Weight Importance (ρ={correlation_q:.4f})')
    ax.grid(True, alpha=0.3)

    # 2. JS means vs K importance
    ax = axes[0, 1]
    ax.scatter(js_means, k_importance, alpha=0.3, s=10)
    correlation_k = np.corrcoef(js_means, k_importance)[0, 1]
    ax.set_xlabel('James-Stein Mean (Activation)')
    ax.set_ylabel('Key L2 Norm (Weight)')
    ax.set_title(f'Activation vs Key Weight Importance (ρ={correlation_k:.4f})')
    ax.grid(True, alpha=0.3)

    # 3. JS means vs V importance
    ax = axes[1, 0]
    ax.scatter(js_means, v_importance, alpha=0.3, s=10)
    correlation_v = np.corrcoef(js_means, v_importance)[0, 1]
    ax.set_xlabel('James-Stein Mean (Activation)')
    ax.set_ylabel('Value L2 Norm (Weight)')
    ax.set_title(f'Activation vs Value Weight Importance (ρ={correlation_v:.4f})')
    ax.grid(True, alpha=0.3)

    # 4. Combined importance (sorted)
    ax = axes[1, 1]

    # Normalize importances to [0, 1] for comparison
    js_norm = (js_means - js_means.min()) / (js_means.max() - js_means.min() + 1e-10)
    q_norm = (q_importance - q_importance.min()) / (q_importance.max() - q_importance.min() + 1e-10)
    k_norm = (k_importance - k_importance.min()) / (k_importance.max() - k_importance.min() + 1e-10)
    v_norm = (v_importance - v_importance.min()) / (v_importance.max() - v_importance.min() + 1e-10)

    channels = np.arange(len(js_means))
    ax.plot(np.sort(js_norm)[::-1], label='JS Activation', alpha=0.7, linewidth=1.5)
    ax.plot(np.sort(q_norm)[::-1], label='Query Weight', alpha=0.7, linewidth=1.5)
    ax.plot(np.sort(k_norm)[::-1], label='Key Weight', alpha=0.7, linewidth=1.5)
    ax.plot(np.sort(v_norm)[::-1], label='Value Weight', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Channel (sorted by importance)')
    ax.set_ylabel('Normalized Importance')
    ax.set_title('Combined Importance Distributions (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'combined_importance.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_report(data, output_dir, js_stats=None):
    """Generate text summary report."""
    metadata = data['metadata']
    js_means = data['js_means']
    naive_means = data['naive_means']
    Wq = data['Wq']
    Wk = data['Wk']
    Wv = data['Wv']

    report = []
    report.append("="*60)
    report.append("XSPOT VISUALIZATION SUMMARY REPORT")
    report.append("="*60)
    report.append(f"\nLayer: {metadata['layer_id']}")
    report.append(f"GQA Group: {metadata['group_id']}")
    report.append(f"Query heads in group: {metadata['group_info']['query_head_start']}-{metadata['group_info']['query_head_end']}")

    report.append("\n--- James-Stein Statistics ---")
    report.append(f"Grand mean: {metadata['james_stein']['grand_mean']:.6f}")
    report.append(f"Shrinkage factor (λ): {metadata['james_stein']['shrinkage_factor']:.6f}")
    report.append(f"Mean absolute shrinkage: {metadata['james_stein']['mean_abs_diff']:.6f}")
    report.append(f"Max absolute shrinkage: {metadata['james_stein']['max_abs_diff']:.6f}")

    # Add detailed statistics comparison if available
    if js_stats is not None:
        report.append("\n--- Detailed Statistics Comparison (Naive vs JS) ---")
        report.append(f"{'Metric':<12} {'Naive':<12} {'JS':<12} {'Difference':<12}")
        report.append("-" * 50)
        stats_naive = js_stats['naive']
        stats_js = js_stats['js']

        report.append(f"{'Min':<12} {stats_naive['min']:<12.6f} {stats_js['min']:<12.6f} {stats_js['min']-stats_naive['min']:<+12.6f}")
        report.append(f"{'Max':<12} {stats_naive['max']:<12.6f} {stats_js['max']:<12.6f} {stats_js['max']-stats_naive['max']:<+12.6f}")
        report.append(f"{'Mean':<12} {stats_naive['mean']:<12.6f} {stats_js['mean']:<12.6f} {stats_js['mean']-stats_naive['mean']:<+12.6f}")
        report.append(f"{'Median':<12} {stats_naive['median']:<12.6f} {stats_js['median']:<12.6f} {stats_js['median']-stats_naive['median']:<+12.6f}")
        report.append(f"{'Std Dev':<12} {stats_naive['std']:<12.6f} {stats_js['std']:<12.6f} {stats_js['std']-stats_naive['std']:<+12.6f}")

        range_naive = stats_naive['max'] - stats_naive['min']
        range_js = stats_js['max'] - stats_js['min']
        report.append(f"{'Range':<12} {range_naive:<12.6f} {range_js:<12.6f} {range_js-range_naive:<+12.6f}")
        report.append(f"\nRange reduction: {js_stats['range_reduction']:.2f}%")
        report.append(f"Std reduction: {(stats_naive['std']-stats_js['std'])/stats_naive['std']*100:.2f}%")

    report.append("\n--- Weight Statistics ---")
    report.append(f"Query weights (Wq):")
    report.append(f"  Shape: {Wq.shape}")
    report.append(f"  Mean: {Wq.mean():.6f}")
    report.append(f"  Std: {Wq.std():.6f}")
    report.append(f"  L2 norm: {np.linalg.norm(Wq):.2f}")

    report.append(f"\nKey weights (Wk):")
    report.append(f"  Shape: {Wk.shape}")
    report.append(f"  Mean: {Wk.mean():.6f}")
    report.append(f"  Std: {Wk.std():.6f}")
    report.append(f"  L2 norm: {np.linalg.norm(Wk):.2f}")

    report.append(f"\nValue weights (Wv):")
    report.append(f"  Shape: {Wv.shape}")
    report.append(f"  Mean: {Wv.mean():.6f}")
    report.append(f"  Std: {Wv.std():.6f}")
    report.append(f"  L2 norm: {np.linalg.norm(Wv):.2f}")

    report.append("\n--- Importance Correlations ---")
    wq_reshaped = Wq.reshape(-1, Wq.shape[2])
    q_importance = np.linalg.norm(wq_reshaped, axis=0)
    k_importance = np.linalg.norm(Wk, axis=0)
    v_importance = np.linalg.norm(Wv, axis=0)

    corr_js_q = np.corrcoef(js_means, q_importance)[0, 1]
    corr_js_k = np.corrcoef(js_means, k_importance)[0, 1]
    corr_js_v = np.corrcoef(js_means, v_importance)[0, 1]
    corr_k_v = np.corrcoef(k_importance, v_importance)[0, 1]

    report.append(f"JS activation vs Query weight: {corr_js_q:.4f}")
    report.append(f"JS activation vs Key weight: {corr_js_k:.4f}")
    report.append(f"JS activation vs Value weight: {corr_js_v:.4f}")
    report.append(f"Key weight vs Value weight: {corr_k_v:.4f}")

    report.append("\n--- Visualizations Generated ---")
    report.append("1. js_means_distribution.png - James-Stein shrinkage analysis")
    report.append("2. weight_distributions.png - Q/K/V weight distributions")
    report.append("3. per_head_analysis.png - Per-head query analysis")
    report.append("4. kv_analysis.png - Key/Value weight analysis")
    report.append("5. combined_importance.png - Combined importance analysis")

    report.append("\n" + "="*60)

    # Save report
    report_text = "\n".join(report)
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nSaved: {output_dir / 'summary_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description='Visualize xspot.py output data')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing xspot.py output (e.g., ./xspot_layer0_group0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualizations (default: {data_dir}/visualizations)')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.data_dir) / 'visualizations')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"XSPOT Visualization Tool")
    print(f"{'='*60}")

    # Load data
    data = load_data(args.data_dir)

    # Generate visualizations
    print(f"\nGenerating visualizations...")
    js_stats = plot_js_means_distribution(data, output_dir)
    plot_weight_distributions(data, output_dir)
    plot_per_head_analysis(data, output_dir)
    plot_kv_analysis(data, output_dir)
    plot_combined_importance(data, output_dir)

    # Generate summary report
    generate_summary_report(data, output_dir, js_stats=js_stats)

    print(f"\n{'='*60}")
    print(f"✓ Visualization complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  1. js_means_distribution.png")
    print(f"  2. weight_distributions.png")
    print(f"  3. per_head_analysis.png")
    print(f"  4. kv_analysis.png")
    print(f"  5. combined_importance.png")
    print(f"  6. summary_report.txt")


if __name__ == '__main__':
    main()
