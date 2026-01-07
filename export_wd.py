"""
Export Weight and Activation Data for Wanda Analysis

Exports from a specific layer:
1. W - Full weight matrix [out_features, in_features]
2. E[X_j] - Mean activation per input channel j [in_features]
3. ||X_j||_2 - L2 norm of activations per channel (for Wanda scoring) [in_features]
4. James-Stein Estimator for E[X_j] [in_features]

This allows detailed analysis of:
- Weight patterns across all output channels
- Activation statistics per input channel
- Wanda pruning scores (|W_ij| * ||X_j||_2)
- James-Stein shrinkage effect on activation means

Usage:
    python export_wd.py --layer-id 3 --channel-id 0
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import os


def compute_james_stein_mean(raw_means, variance_estimate=None):
    """
    Apply James-Stein shrinkage estimator to channel-wise means.

    The James-Stein estimator shrinks individual means toward the grand mean,
    provably dominating the MLE (simple mean) in terms of MSE when p ≥ 3.

    Args:
        raw_means: Tensor of shape [in_features] containing sample means for each channel
        variance_estimate: Optional variance estimate. If None, uses empirical variance.

    Returns:
        Tensor of James-Stein estimated means

    Formula:
        μ̂_JS[j] = μ̄ + (1 - c) × (X̄[j] - μ̄)
        where c = (p - 2) × σ² / Σ(X̄[j] - μ̄)²
    """
    p = len(raw_means)

    # Need at least 3 dimensions for James-Stein to be beneficial
    if p < 3:
        return raw_means

    # Compute grand mean
    grand_mean = raw_means.mean()

    # Compute deviations from grand mean
    deviations = raw_means - grand_mean

    # Compute sum of squared deviations
    sum_sq_dev = (deviations ** 2).sum()

    # Prevent division by zero
    if sum_sq_dev < 1e-10:
        # All means are the same, no shrinkage needed
        return raw_means

    # Estimate variance
    if variance_estimate is None:
        # Use a conservative estimate: mean absolute deviation squared
        # This is more robust to outliers than variance
        variance_estimate = ((raw_means - grand_mean).abs().mean()) ** 2
        # Add small constant for numerical stability
        variance_estimate = variance_estimate.clamp(min=1e-8)

    # Compute shrinkage factor c
    # c = (p - 2) × σ² / Σ(X̄[j] - μ̄)²
    shrinkage_factor = ((p - 2) * variance_estimate) / sum_sq_dev

    # Clamp shrinkage factor to [0, 1] for stability
    # c > 1 means we overshoot, c < 0 means we expand (both bad)
    shrinkage_factor = shrinkage_factor.clamp(0, 1)

    # Apply James-Stein shrinkage
    # μ̂_JS[j] = μ̄ + (1 - c) × (X̄[j] - μ̄)
    js_means = grand_mean + (1 - shrinkage_factor) * deviations

    return js_means


class WandaDataExporter:
    """Export weight and activation data for Wanda analysis."""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Storage
        self.activation_data = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activation_data:
                    self.activation_data[name] = []
                if isinstance(input, tuple):
                    inp = input[0].detach().cpu()
                else:
                    inp = input.detach().cpu()
                self.activation_data[name].append(inp)
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_hook(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    @torch.no_grad()
    def get_activation_mean(self, name):
        """
        Compute E[X[:,j]] - mean activation per input channel.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        mean_sum = torch.zeros(in_features, dtype=torch.float32)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            mean_sum += x_flat.sum(dim=0)

        mean_activation = mean_sum / total_samples
        return mean_activation

    @torch.no_grad()
    def get_activation_l2_norm(self, name):
        """
        Compute ||X_j||_2 - L2 norm per input channel (for Wanda scoring).

        Where X_j is the j-th input channel across all calibration tokens.
        ||X_j||_2 = sqrt(sum(X_j^2))
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        in_features = X_list[0].shape[-1]

        # Accumulate squared values for L2 norm
        squared_sum = torch.zeros(in_features, dtype=torch.float32)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            # Accumulate sum of squares
            squared_sum += x_flat.pow(2).sum(dim=0)

        # L2 norm: sqrt of sum of squares
        l2_norm = torch.sqrt(squared_sum)
        return l2_norm

    @torch.no_grad()
    def export_layer_data(self, layer_name, module, channel_id=0):
        """
        Export weight and activation data for a specific layer.

        Args:
            layer_name: Name of the layer
            module: The linear module
            channel_id: Input channel ID (for reference/metadata)

        Returns:
            Dictionary with exported data
        """
        print(f"\n📊 Exporting data for layer: {layer_name}")
        print(f"   Channel ID (reference): {channel_id}")

        # Get weight matrix W
        W = module.weight.data.cpu()
        out_features, in_features = W.shape

        print(f"   Weight matrix shape: [{out_features}, {in_features}]")

        # Get activation mean E[X[:,j]]
        activation_mean = self.get_activation_mean(layer_name)
        if activation_mean is None:
            print("   ⚠️  No activation data available")
            return None

        # Get L2 norm ||X_j||_2
        activation_l2_norm = self.get_activation_l2_norm(layer_name)

        # Compute James-Stein estimator
        js_mean = compute_james_stein_mean(activation_mean)

        # Calculate shrinkage statistics
        grand_mean = activation_mean.mean()
        shrinkage_amount = (activation_mean - js_mean).abs().mean()

        export_dict = {
            'layer_name': layer_name,
            'channel_id': channel_id,
            'out_features': out_features,
            'in_features': in_features,

            # Weight matrix (full)
            'W': W.numpy(),  # Shape: [out_features, in_features]

            # Activation statistics (per input channel)
            'E[X]': activation_mean.numpy(),  # Shape: [in_features]
            'L2_norm[X]': activation_l2_norm.numpy(),  # Shape: [in_features]
            'JS_mean[X]': js_mean.numpy(),  # Shape: [in_features]

            # Metadata
            'grand_mean': grand_mean.item(),
            'shrinkage_amount': shrinkage_amount.item(),
        }

        print(f"   ✅ Exported weight matrix: [{out_features}, {in_features}]")
        print(f"   ✅ Exported activation stats: [{in_features}]")
        print(f"   Grand mean: {grand_mean:.6f}")
        print(f"   JS shrinkage amount: {shrinkage_amount:.6f}")

        return export_dict

    def calibrate(self, calibration_data, n_samples=128):
        """Run calibration on the dataset to collect activations."""
        print(f"\nCalibrating with {n_samples} samples...")
        self.model.eval()
        self.register_hooks()

        successful = 0
        for i, text in enumerate(tqdm(calibration_data[:n_samples], desc="Calibration")):
            try:
                inputs = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    _ = self.model(**inputs, use_cache=False, return_dict=True)

                successful += 1

            except Exception:
                continue

        self.remove_hooks()
        print(f"Calibration complete! Successfully processed {successful}/{n_samples} samples")

    def save_export_data(self, export_dict, output_dir):
        """
        Save exported data to CSV and numpy files.
        """
        os.makedirs(output_dir, exist_ok=True)

        layer_name = export_dict['layer_name'].replace('/', '_').replace('.', '_')
        ch_id = export_dict['channel_id']
        prefix = f"{layer_name}_ch{ch_id}"

        # --- 1. Save per-channel statistics as CSV ---
        df_data = {
            'input_channel_id': np.arange(export_dict['in_features']),
            'E[X]': export_dict['E[X]'],
            'L2_norm[X]': export_dict['L2_norm[X]'],
            'JS_mean[X]': export_dict['JS_mean[X]'],
        }
        df = pd.DataFrame(df_data)

        csv_path = os.path.join(output_dir, f"{prefix}_stats.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved per-channel stats CSV: {csv_path}")

        # --- 2. Save full weight matrix W as NPZ ---
        npz_path = os.path.join(output_dir, f"{prefix}_weights.npz")
        np.savez(
            npz_path,
            W=export_dict['W'],  # Full weight matrix [out_features, in_features]
            layer_name=export_dict['layer_name'],
            out_features=export_dict['out_features'],
            in_features=export_dict['in_features'],
        )
        print(f"✅ Saved weight matrix NPZ: {npz_path}")

        # --- 3. Save complete data as comprehensive NPZ ---
        npz_complete_path = os.path.join(output_dir, f"{prefix}_complete.npz")
        np.savez(
            npz_complete_path,
            layer_name=export_dict['layer_name'],
            channel_id=export_dict['channel_id'],
            out_features=export_dict['out_features'],
            in_features=export_dict['in_features'],
            W=export_dict['W'],
            E_X=export_dict['E[X]'],
            L2_norm_X=export_dict['L2_norm[X]'],
            JS_mean_X=export_dict['JS_mean[X]'],
            grand_mean=export_dict['grand_mean'],
            shrinkage_amount=export_dict['shrinkage_amount'],
        )
        print(f"✅ Saved complete data NPZ: {npz_complete_path}")

        # --- 4. Save metadata ---
        metadata_path = os.path.join(output_dir, f"{prefix}_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Layer: {export_dict['layer_name']}\n")
            f.write(f"Channel ID (reference): {export_dict['channel_id']}\n")
            f.write(f"Output Features: {export_dict['out_features']}\n")
            f.write(f"Input Features: {export_dict['in_features']}\n")
            f.write(f"Grand Mean: {export_dict['grand_mean']:.6f}\n")
            f.write(f"JS Shrinkage Amount: {export_dict['shrinkage_amount']:.6f}\n")
            f.write(f"\nData Files:\n")
            f.write(f"  Per-channel stats CSV: {prefix}_stats.csv\n")
            f.write(f"  Weight matrix NPZ: {prefix}_weights.npz\n")
            f.write(f"  Complete data NPZ: {prefix}_complete.npz\n")
            f.write(f"\nColumns in stats CSV:\n")
            f.write(f"  input_channel_id: Index j of input channel\n")
            f.write(f"  E[X]: Mean activation E[X[:,j]]\n")
            f.write(f"  L2_norm[X]: L2 norm ||X_j||_2 (for Wanda scoring)\n")
            f.write(f"  JS_mean[X]: James-Stein estimated mean\n")
            f.write(f"\nArrays in NPZ files:\n")
            f.write(f"  W: Weight matrix [out_features, in_features]\n")
            f.write(f"  E_X: Mean activation [in_features]\n")
            f.write(f"  L2_norm_X: L2 norm [in_features]\n")
            f.write(f"  JS_mean_X: James-Stein mean [in_features]\n")
            f.write(f"\nWanda Scoring:\n")
            f.write(f"  Score_ij = |W_ij| * L2_norm_X[j]\n")
            f.write(f"  where W_ij is element [i,j] of weight matrix W\n")

        print(f"✅ Saved metadata: {metadata_path}")


def load_wikitext2(split="train", n_samples=None):
    """Load WikiText-2 dataset."""
    print(f"Loading WikiText-2 {split} dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    if n_samples:
        texts = texts[:n_samples]
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Export weight and activation data for Wanda analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--layer-id", type=int, default=3,
                       help="Layer ID to export (e.g., 3 for model.layers.3.*)")
    parser.add_argument("--layer-type", type=str, default="mlp.gate_proj",
                       help="Layer type suffix (e.g., mlp.gate_proj, self_attn.q_proj)")
    parser.add_argument("--channel-id", type=int, default=0,
                       help="Channel ID (for reference/metadata)")
    parser.add_argument("--n-calib", type=int, default=128,
                       help="Calibration samples")
    parser.add_argument("--output-dir", type=str, default="./exported_wanda_data",
                       help="Output directory")
    parser.add_argument("--model-path", type=str, default="../FPRAG/models/Mistral-7B-v0.3",
                       help="Model name or local path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("WANDA DATA EXPORTER")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Target layer: model.layers.{args.layer_id}.{args.layer_type}")
    print(f"Channel ID (reference): {args.channel_id}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Fix for Mistral/Llama models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # Load calibration data
    calib_texts = load_wikitext2(split="train", n_samples=args.n_calib)

    # Initialize exporter
    exporter = WandaDataExporter(
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Calibrate
    exporter.calibrate(calib_texts, n_samples=args.n_calib)

    # Find target layer
    target_layer_name = f"model.layers.{args.layer_id}.{args.layer_type}"
    target_module = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name == target_layer_name:
            target_module = module
            break

    if target_module is None:
        print(f"\n❌ Error: Layer '{target_layer_name}' not found!")
        print("\nAvailable layers:")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name.startswith(f"model.layers.{args.layer_id}"):
                print(f"  {name}")
        return

    # Export data
    export_dict = exporter.export_layer_data(
        target_layer_name,
        target_module,
        channel_id=args.channel_id
    )

    if export_dict is not None:
        exporter.save_export_data(export_dict, args.output_dir)

        print("\n" + "=" * 80)
        print("EXPORT COMPLETE!")
        print("=" * 80)
        print(f"Exported data for: {target_layer_name}")
        print(f"Weight matrix W: [{export_dict['out_features']}, {export_dict['in_features']}]")
        print(f"Activation stats: [{export_dict['in_features']}]")
        print(f"Files saved to: {args.output_dir}")
        print("\nYou can now analyze:")
        print(f"  1. Weight patterns: W matrix")
        print(f"  2. Activation means: E[X] and James-Stein E[X]")
        print(f"  3. Wanda scores: |W_ij| * L2_norm[X_j]")
        print(f"  4. Shrinkage effect: E[X] vs JS_mean[X]")
        print("=" * 80)


if __name__ == "__main__":
    main()
