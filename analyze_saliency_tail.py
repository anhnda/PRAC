"""
Analyze saliency with dual knee-point detection and tail distribution analysis.

This script:
1. Computes saliency scores for all channels in a target layer
2. Identifies top 10 channels with largest saliency
3. For these channels, visualizes:
   - Sorted E[X²] with BOTH 1st-half and 2nd-half knee points
   - Sorted grad² with BOTH 1st-half and 2nd-half knee points
4. Analyzes tail distribution:
   - Distribution of weights/importance in tail (after 2nd-half knee)
   - Comparison of tail vs whole distribution
   - Statistics showing tail characteristics

NEW Features:
- Dual knee detection (1st half vs 2nd half)
- Tail analysis (weights after 2nd-half knee)
- Distribution comparison (tail vs whole)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import pandas as pd
from kneed import KneeLocator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class SaliencyTailAnalyzer:
    """Analyzer for saliency with dual knee detection and tail analysis."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

    def get_calibration_data(self, n_samples=128, max_length=512):
        """Load calibration data from WikiText-2."""
        print(f"Loading {n_samples} calibration samples from WikiText-2...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        texts = []
        for example in dataset:
            if example['text'].strip():
                texts.append(example['text'])
                if len(texts) >= n_samples * 2:
                    break

        calibration_data = []
        for text in tqdm(texts[:n_samples], desc="Tokenizing"):
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            calibration_data.append(inputs['input_ids'].to(self.device))

        return calibration_data

    def find_target_layer(self, layer_idx=3):
        """Find the target linear layer (gate_proj with SiLU)."""
        target_module = self.model.model.layers[layer_idx].mlp.gate_proj
        print(f"Target layer: model.layers[{layer_idx}].mlp.gate_proj")
        print(f"Weight shape: {target_module.weight.shape}")
        return target_module, layer_idx

    def compute_saliency_and_gradients(self, target_module, calibration_data, layer_idx):
        """Compute saliency scores and collect gradients."""
        pre_activations = []
        post_activations = []
        input_activations = []

        def forward_hook(module, input, output):
            input_activations.append(input[0].detach().cpu())
            pre_activations.append(output.detach().cpu())

        def silu_hook(module, input, output):
            post_activations.append(output.detach().cpu())

        hook1 = target_module.register_forward_hook(forward_hook)
        silu_module = self.model.model.layers[layer_idx].mlp.act_fn
        hook2 = silu_module.register_forward_hook(silu_hook)

        print("Collecting activations from calibration data...")
        self.model.eval()
        with torch.no_grad():
            for input_ids in tqdm(calibration_data, desc="Forward passes"):
                self.model(input_ids, use_cache=False)

        hook1.remove()
        hook2.remove()

        X = torch.cat(input_activations, dim=0).view(-1, input_activations[0].shape[-1]).float()
        pre_act = torch.cat(pre_activations, dim=0).view(-1, pre_activations[0].shape[-1]).float()
        post_act = torch.cat(post_activations, dim=0).view(-1, post_activations[0].shape[-1]).float()

        print(f"Collected {X.shape[0]} tokens")
        print(f"Input shape: {X.shape}, Pre-activation shape: {pre_act.shape}")

        # Compute saliency: E[|XW^T|]
        saliency = torch.mean(torch.abs(pre_act), dim=0)

        # Compute weight gradients
        print("Computing weight gradients...")
        weight_gradients = self.compute_weight_gradients(X, pre_act, post_act, target_module.weight)

        return {
            'saliency': saliency,
            'weight_gradients': weight_gradients,
            'input_activations': X,
            'pre_activations': pre_act,
            'post_activations': post_act
        }

    def compute_weight_gradients(self, X, pre_act, post_act, weight):
        """Compute gradients of weights w.r.t. SiLU output."""
        X_gpu = X.to(self.device)
        pre_act_gpu = pre_act.to(self.device)

        sigmoid_z = torch.sigmoid(pre_act_gpu)
        silu_grad = sigmoid_z * (1 + pre_act_gpu * (1 - sigmoid_z))

        num_tokens = X_gpu.shape[0]
        out_features, in_features = weight.shape
        weight_grad_squared = torch.zeros(out_features, in_features, device=self.device)

        batch_size = 1000
        for start_idx in tqdm(range(0, num_tokens, batch_size), desc="Computing gradients"):
            end_idx = min(start_idx + batch_size, num_tokens)
            X_batch = X_gpu[start_idx:end_idx]
            silu_grad_batch = silu_grad[start_idx:end_idx]

            for j in range(out_features):
                grad_batch = silu_grad_batch[:, j:j+1] * X_batch
                weight_grad_squared[j] += torch.sum(grad_batch ** 2, dim=0)

        weight_grad_squared = weight_grad_squared / num_tokens
        return weight_grad_squared.cpu()

    def detect_dual_knees(self, sorted_values):
        """
        Detect knee points using TWO methods:
        1. First-half knee: Perpendicular distance from line connecting start to half point
        2. Tail knee: Perpendicular distance from line connecting half point to end point

        Returns:
            knee_1st: Knee index from first-half detection (absolute index)
            knee_2nd: Knee index from tail detection (absolute index)
        """
        n = len(sorted_values)
        half_n = n // 2

        values = sorted_values.cpu().numpy()

        # First-half knee: Find furthest point from line (0, values[0]) to (half_n, values[half_n])
        knee_1st = self._find_knee_perpendicular(values, 0, half_n)

        # Tail knee: Find furthest point from line (half_n, values[half_n]) to (n-1, values[n-1])
        knee_2nd = self._find_knee_perpendicular(values, half_n, n - 1)

        return knee_1st, knee_2nd

    def _find_knee_perpendicular(self, values, start_idx, end_idx):
        """
        Find knee point using perpendicular distance method.

        Algorithm:
        1. Draw a line from (start_idx, values[start_idx]) to (end_idx, values[end_idx])
        2. Find the point between start and end with maximum perpendicular distance to this line
        3. Return the index of that point
        """
        if end_idx - start_idx < 2:
            return start_idx

        # Line endpoints
        x1, y1 = start_idx, values[start_idx]
        x2, y2 = end_idx, values[end_idx]

        # Compute perpendicular distances for all points between start and end
        max_distance = 0
        max_idx = start_idx

        for i in range(start_idx + 1, end_idx):
            x0, y0 = i, values[i]

            # Perpendicular distance from point (x0, y0) to line through (x1, y1) and (x2, y2)
            # Formula: |((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)| / sqrt((y2-y1)^2 + (x2-x1)^2)
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

            if denominator > 0:
                distance = numerator / denominator

                if distance > max_distance:
                    max_distance = distance
                    max_idx = i

        return max_idx

    def analyze_top_channels(self, results, top_k=10, output_dir='./visualizations/saliency_tail_analysis', target_module=None):
        """Analyze top-k channels by saliency with dual knee and tail analysis."""
        os.makedirs(output_dir, exist_ok=True)

        saliency = results['saliency']
        weight_gradients = results['weight_gradients']
        X = results['input_activations']
        pre_act = results['pre_activations']

        top_k_indices = torch.argsort(saliency, descending=True)[:top_k]

        print(f"\nTop {top_k} channels by saliency:")
        for rank, ch_idx in enumerate(top_k_indices):
            print(f"  Rank {rank+1}: Channel {ch_idx.item()}, Saliency = {saliency[ch_idx]:.4f}")

        tail_stats_all = []

        for rank, ch_idx in enumerate(top_k_indices):
            ch_idx = ch_idx.item()
            tail_stats = self.visualize_channel_with_tail(
                ch_idx, rank + 1, X, pre_act, weight_gradients,
                saliency[ch_idx].item(), output_dir, target_module
            )
            tail_stats_all.append(tail_stats)

        self.create_tail_summary(tail_stats_all, output_dir)
        self.plot_saliency_distribution(saliency, top_k_indices, output_dir)

    def visualize_channel_with_tail(self, ch_idx, rank, X, pre_act, weight_gradients, saliency_score, output_dir, target_module=None):
        """
        Visualize channel with dual knee points and tail distribution analysis.
        """
        # Compute per-feature statistics
        X_squared = X ** 2
        X_squared_mean = torch.mean(X_squared, dim=0)
        X_mean = torch.mean(X, dim=0)
        grad_squared = weight_gradients[ch_idx]

        if target_module is not None:
            weights = target_module.weight[ch_idx].detach().cpu().float()
        else:
            weights = None

        # Sort by importance
        X_sorted_values, X_sorted_indices = torch.sort(X_squared_mean, descending=True)
        grad_sorted_values, grad_sorted_indices = torch.sort(grad_squared, descending=True)

        # Detect dual knees (first-half and second-half)
        knee_1st_x2, knee_2nd_x2 = self.detect_dual_knees(X_sorted_values)
        knee_1st_grad2, knee_2nd_grad2 = self.detect_dual_knees(grad_sorted_values)

        # Create figure with 4 rows, 3 columns
        fig, axes = plt.subplots(4, 3, figsize=(18, 18))

        n_features = len(X_sorted_values)

        # Row 1: Sorted X² with dual knees
        ax = axes[0, 0]
        ax.plot(range(n_features), X_sorted_values.numpy(), linewidth=1.5, alpha=0.8)
        ax.axvline(knee_1st_x2, color='blue', linestyle='--', linewidth=2, label=f'1st-half knee: {knee_1st_x2}')
        ax.axvline(knee_2nd_x2, color='red', linestyle='--', linewidth=2, label=f'2nd-half knee: {knee_2nd_x2}')
        ax.plot(knee_1st_x2, X_sorted_values[knee_1st_x2].numpy(), 'bo', markersize=10)
        ax.plot(knee_2nd_x2, X_sorted_values[knee_2nd_x2].numpy(), 'ro', markersize=10)
        ax.set_xlabel('Feature Rank (sorted by X²)', fontsize=11)
        ax.set_ylabel('Mean Squared Input (X²)', fontsize=11)
        ax.set_title(f'Rank {rank}: Ch {ch_idx} - Sorted X² (Dual Knees)', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Row 1: Sorted grad² with dual knees
        ax = axes[0, 1]
        ax.plot(range(n_features), grad_sorted_values.numpy(), linewidth=1.5, alpha=0.8, color='orangered')
        ax.axvline(knee_1st_grad2, color='blue', linestyle='--', linewidth=2, label=f'1st-half knee: {knee_1st_grad2}')
        ax.axvline(knee_2nd_grad2, color='red', linestyle='--', linewidth=2, label=f'2nd-half knee: {knee_2nd_grad2}')
        ax.plot(knee_1st_grad2, grad_sorted_values[knee_1st_grad2].numpy(), 'bo', markersize=10)
        ax.plot(knee_2nd_grad2, grad_sorted_values[knee_2nd_grad2].numpy(), 'ro', markersize=10)
        ax.set_xlabel('Weight Rank (sorted by grad²)', fontsize=11)
        ax.set_ylabel('Squared Weight Gradient', fontsize=11)
        ax.set_title(f'Rank {rank}: Ch {ch_idx} - Sorted grad² (Dual Knees)', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Row 1: Knee comparison
        ax = axes[0, 2]
        knee_data = {
            'X² (1st)': knee_1st_x2,
            'X² (2nd)': knee_2nd_x2,
            'grad² (1st)': knee_1st_grad2,
            'grad² (2nd)': knee_2nd_grad2
        }
        bars = ax.bar(range(len(knee_data)), list(knee_data.values()), color=['blue', 'red', 'blue', 'red'])
        ax.set_xticks(range(len(knee_data)))
        ax.set_xticklabels(list(knee_data.keys()), rotation=45, ha='right')
        ax.set_ylabel('Knee Index', fontsize=11)
        ax.set_title(f'Knee Point Comparison', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (k, v) in enumerate(knee_data.items()):
            ax.text(i, v, f'{v}', ha='center', va='bottom', fontsize=9)

        # Row 2: Tail distributions for X²
        if weights is not None:
            ax = axes[1, 0]
            tail_indices_x2 = X_sorted_indices[knee_2nd_x2:]
            tail_weights_x2 = weights[tail_indices_x2]
            all_weights = weights

            ax.hist(all_weights.numpy(), bins=50, alpha=0.3, label='All weights', color='gray', density=True)
            ax.hist(tail_weights_x2.numpy(), bins=30, alpha=0.7, label=f'Tail weights (n={len(tail_weights_x2)})', color='red', density=True)
            ax.set_xlabel('Weight Value', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'X² Tail Distribution (after 2nd knee)', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

            # Stats text
            tail_ratio = len(tail_weights_x2) / len(all_weights)
            stats_text = f'Tail: {len(tail_weights_x2)}/{len(all_weights)} ({tail_ratio*100:.1f}%)\n'
            stats_text += f'Tail μ={tail_weights_x2.mean():.3f}, σ={tail_weights_x2.std():.3f}\n'
            stats_text += f'All μ={all_weights.mean():.3f}, σ={all_weights.std():.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax = axes[1, 0]
            ax.text(0.5, 0.5, 'Weights not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')

        # Row 2: Tail distributions for grad²
        if weights is not None:
            ax = axes[1, 1]
            tail_indices_grad2 = grad_sorted_indices[knee_2nd_grad2:]
            tail_weights_grad2 = weights[tail_indices_grad2]

            ax.hist(all_weights.numpy(), bins=50, alpha=0.3, label='All weights', color='gray', density=True)
            ax.hist(tail_weights_grad2.numpy(), bins=30, alpha=0.7, label=f'Tail weights (n={len(tail_weights_grad2)})', color='orange', density=True)
            ax.set_xlabel('Weight Value', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'grad² Tail Distribution (after 2nd knee)', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

            # Stats text
            tail_ratio = len(tail_weights_grad2) / len(all_weights)
            stats_text = f'Tail: {len(tail_weights_grad2)}/{len(all_weights)} ({tail_ratio*100:.1f}%)\n'
            stats_text += f'Tail μ={tail_weights_grad2.mean():.3f}, σ={tail_weights_grad2.std():.3f}\n'
            stats_text += f'All μ={all_weights.mean():.3f}, σ={all_weights.std():.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax = axes[1, 1]
            ax.text(0.5, 0.5, 'Weights not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')

        # Row 2: Box plot comparison
        if weights is not None:
            ax = axes[1, 2]
            data_to_plot = [
                all_weights.numpy(),
                tail_weights_x2.numpy(),
                tail_weights_grad2.numpy()
            ]
            labels = ['All', f'X² tail\n({len(tail_weights_x2)})', f'grad² tail\n({len(tail_weights_grad2)})']
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgray')
            bp['boxes'][1].set_facecolor('lightcoral')
            bp['boxes'][2].set_facecolor('orange')
            ax.set_ylabel('Weight Value', fontsize=11)
            ax.set_title('Weight Range Comparison', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        else:
            ax = axes[1, 2]
            ax.axis('off')

        # Row 3 & 4: Existing plots (importance values in tail)
        ax = axes[2, 0]
        tail_x2_importance = X_squared_mean[tail_indices_x2] if weights is not None else torch.tensor([])
        if len(tail_x2_importance) > 0:
            ax.hist(X_squared_mean.numpy(), bins=50, alpha=0.3, label='All importance', color='gray', density=True)
            ax.hist(tail_x2_importance.numpy(), bins=30, alpha=0.7, label='Tail importance', color='red', density=True)
            ax.set_xlabel('X² Importance', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title('X² Importance: Tail vs All', fontsize=12)
            ax.set_xscale('log')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        ax = axes[2, 1]
        tail_grad2_importance = grad_squared[tail_indices_grad2] if weights is not None else torch.tensor([])
        if len(tail_grad2_importance) > 0:
            ax.hist(grad_squared.numpy(), bins=50, alpha=0.3, label='All importance', color='gray', density=True)
            ax.hist(tail_grad2_importance.numpy(), bins=30, alpha=0.7, label='Tail importance', color='orange', density=True)
            ax.set_xlabel('grad² Importance', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title('grad² Importance: Tail vs All', fontsize=12)
            ax.set_xscale('log')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        # Summary statistics
        ax = axes[2, 2]
        ax.axis('off')
        if weights is not None:
            summary_text = f"TAIL ANALYSIS SUMMARY\n\n"
            summary_text += f"X² 2nd-half knee: {knee_2nd_x2} ({knee_2nd_x2/n_features*100:.1f}%)\n"
            summary_text += f"Tail size: {len(tail_weights_x2)} ({len(tail_weights_x2)/n_features*100:.1f}%)\n\n"
            summary_text += f"grad² 2nd-half knee: {knee_2nd_grad2} ({knee_2nd_grad2/n_features*100:.1f}%)\n"
            summary_text += f"Tail size: {len(tail_weights_grad2)} ({len(tail_weights_grad2)/n_features*100:.1f}%)\n\n"
            summary_text += "INTERPRETATION:\n"
            summary_text += "• 2nd-half knee = where importance → 0\n"
            summary_text += "• Tail = unimportant weights\n"
            summary_text += "• Clamping ignores tail, focuses on\n  important weights (before knee)"
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # Additional rows for detailed analysis
        for i in range(3, 4):
            for j in range(3):
                axes[i, j].axis('off')

        plt.suptitle(f'Channel {ch_idx} - Dual Knee & Tail Analysis (Saliency={saliency_score:.4f})',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/channel_{ch_idx}_rank_{rank}_tail_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved tail analysis for channel {ch_idx} (rank {rank})")

        # Return stats
        tail_stats = {
            'channel_idx': ch_idx,
            'rank': rank,
            'knee_1st_x2': knee_1st_x2,
            'knee_2nd_x2': knee_2nd_x2,
            'knee_1st_grad2': knee_1st_grad2,
            'knee_2nd_grad2': knee_2nd_grad2,
        }

        if weights is not None:
            tail_stats.update({
                'tail_size_x2': len(tail_weights_x2),
                'tail_ratio_x2': len(tail_weights_x2) / len(all_weights),
                'tail_size_grad2': len(tail_weights_grad2),
                'tail_ratio_grad2': len(tail_weights_grad2) / len(all_weights),
            })

        return tail_stats

    def create_tail_summary(self, tail_stats_all, output_dir):
        """Create summary of tail statistics across all channels."""
        df = pd.DataFrame(tail_stats_all)
        df.to_csv(f'{output_dir}/tail_statistics_summary.csv', index=False)
        print(f"\nSaved tail summary to {output_dir}/tail_statistics_summary.csv")
        print("\nTail Statistics Summary:")
        print(df.to_string(index=False))

    def plot_saliency_distribution(self, saliency, top_k_indices, output_dir):
        """Plot overall saliency distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(saliency.numpy(), bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Saliency Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Saliency Scores (All Channels)', fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        threshold = saliency[top_k_indices[-1]].item()
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Top-{len(top_k_indices)} threshold')
        ax.legend()

        ax = axes[1]
        sorted_saliency, _ = torch.sort(saliency, descending=True)
        ax.plot(range(len(sorted_saliency)), sorted_saliency.numpy(), linewidth=1.5)
        ax.set_xlabel('Channel Rank', fontsize=12)
        ax.set_ylabel('Saliency Score', fontsize=12)
        ax.set_title('Sorted Saliency Scores', fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        ax.axvline(len(top_k_indices), color='red', linestyle='--', linewidth=2,
                   label=f'Top-{len(top_k_indices)} cutoff')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/saliency_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved saliency distribution plot")


def main():
    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    layer_idx = 3
    n_calibration_samples = 128
    top_k = 10
    output_dir = './visualizations/saliency_tail_analysis'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map='auto' if device == 'cuda' else None,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    analyzer = SaliencyTailAnalyzer(model, tokenizer, device)
    calibration_data = analyzer.get_calibration_data(n_samples=n_calibration_samples)
    target_module, layer_idx = analyzer.find_target_layer(layer_idx=layer_idx)

    print("\n" + "="*60)
    print(f"Analyzing Layer {layer_idx} with Dual Knee & Tail Analysis")
    print("="*60)
    results = analyzer.compute_saliency_and_gradients(
        target_module, calibration_data, layer_idx
    )

    analyzer.analyze_top_channels(results, top_k=top_k, output_dir=output_dir, target_module=target_module)

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
