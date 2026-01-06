"""
Group-Wise AWQ ASYMMETRIC L2 - Rounding Statistics Analysis

This script extends gw_awq_asym_l2.py to track and report:
- How many weights are rounded UP at each layer
- How many weights are rounded DOWN at each layer
- How many weights need NO rounding (already integers)
- Distribution of rounding magnitudes
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
import pandas as pd


class GroupWiseAWQAsymmetricL2StatsQuantizer:
    """
    Enhanced version that tracks rounding statistics during quantization.
    """

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        # NEW: Storage for rounding statistics
        self.rounding_stats = {}

        print(f"\n[Group-Wise AWQ ASYMMETRIC L2 Stats Quantizer Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Grid search points: {n_grid}")
        print(f"  Group size: {group_size}")
        print(f"  Tracking: Rounding statistics (UP/DOWN/EXACT)")

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
    def get_activation_salience_l2(self, name):
        """
        Compute per-input-channel activation salience using L2 norm: E[X[:, j]²]
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Accumulate L2 salience on CPU
        salience_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            salience_sum += x_flat.pow(2).sum(dim=0)

        salience = salience_sum / total_samples
        return salience

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric_with_stats(self, W, layer_name):
        """
        Group-wise ASYMMETRIC quantization WITH ROUNDING STATISTICS.

        Tracks:
        - rounded_up: Number of weights rounded up
        - rounded_down: Number of weights rounded down
        - exact: Number of weights that were already integers
        - rounding_magnitudes: Distribution of |rounded - original|

        Args:
            W: Weight tensor [out_features, in_features]
            layer_name: Name of the layer for statistics tracking

        Returns:
            W_quant: Quantized and dequantized weights
        """
        out_features, in_features = W.shape

        # Pad to make in_features divisible by group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=W.device, dtype=W.dtype)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        # Reshape to [out_features, n_groups, group_size]
        W_grouped = W_padded.reshape(out_features, n_groups, self.group_size)

        # Compute min and max per group
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]

        # Asymmetric quantization parameters
        scale = (W_max - W_min) / 15.0
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-W_min / scale).clamp(0, 15)

        # ===== ROUNDING STATISTICS TRACKING =====
        # Compute the continuous value before rounding
        W_continuous = W_grouped / scale + zero_point

        # Apply rounding
        W_int = torch.round(W_continuous).clamp(0, 15)

        # Calculate rounding differences
        rounding_diff = W_int - W_continuous  # Positive = rounded up, Negative = rounded down

        # Count rounding directions
        rounded_up = (rounding_diff > 1e-6).sum().item()
        rounded_down = (rounding_diff < -1e-6).sum().item()
        exact = (torch.abs(rounding_diff) <= 1e-6).sum().item()

        # Collect rounding magnitude statistics
        rounding_magnitudes = torch.abs(rounding_diff).cpu().numpy().flatten()

        # Store statistics for this layer
        self.rounding_stats[layer_name] = {
            'rounded_up': rounded_up,
            'rounded_down': rounded_down,
            'exact': exact,
            'total_weights': W_grouped.numel(),
            'rounding_mag_mean': float(np.mean(rounding_magnitudes)),
            'rounding_mag_std': float(np.std(rounding_magnitudes)),
            'rounding_mag_max': float(np.max(rounding_magnitudes)),
            'rounding_mag_percentile_90': float(np.percentile(rounding_magnitudes, 90)),
            'rounding_mag_percentile_99': float(np.percentile(rounding_magnitudes, 99)),
        }
        # ===== END STATISTICS TRACKING =====

        # Dequantize
        W_dequant_grouped = (W_int - zero_point) * scale

        # Reshape back
        W_dequant = W_dequant_grouped.reshape(out_features, padded_in_features)

        # Remove padding if added
        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """
        Grid search for optimal per-input-channel scaling factor using L2 salience.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        # Prepare calibration data
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        max_samples = min(2048, X_cpu.shape[0])
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)

        del X_cpu

        W = module.weight.data
        b = module.bias.data if module.bias is not None else None

        # Compute original output
        if b is not None:
            Y_orig = torch.matmul(X_search, W.t()) + b
        else:
            Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        activation_salience = activation_salience.to(self.device)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            # Compute per-input-channel scales from L2 salience
            scales = activation_salience.pow(alpha).clamp(min=1e-5)

            # Scale weight COLUMNS
            W_scaled = W * scales.unsqueeze(0)

            # Quantize (without stats during grid search to avoid overhead)
            # We'll compute stats only once during final quantization
            W_quant = self.quantize_weight_groupwise_asymmetric_no_stats(W_scaled)

            # Compensate input
            X_compensated = X_search / scales.unsqueeze(0)

            if b is not None:
                Y_quant = torch.matmul(X_compensated, W_quant.t()) + b
            else:
                Y_quant = torch.matmul(X_compensated, W_quant.t())

            # Compute reconstruction error (MSE)
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

        del X_search, Y_orig
        if 'Y_quant' in locals():
            del Y_quant
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_weight_groupwise_asymmetric_no_stats(self, W):
        """
        Fast version without statistics tracking for grid search.
        """
        out_features, in_features = W.shape

        # Pad to make in_features divisible by group_size
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=W.device, dtype=W.dtype)
            W_padded[:, :in_features] = W
        else:
            W_padded = W

        # Reshape to [out_features, n_groups, group_size]
        W_grouped = W_padded.reshape(out_features, n_groups, self.group_size)

        # Compute min and max per group
        W_min = W_grouped.min(dim=2, keepdim=True)[0]
        W_max = W_grouped.max(dim=2, keepdim=True)[0]

        # Asymmetric quantization parameters
        scale = (W_max - W_min) / 15.0
        scale = scale.clamp(min=1e-8)
        zero_point = torch.round(-W_min / scale).clamp(0, 15)

        # Quantize to [0, 15]
        W_int = torch.round(W_grouped / scale + zero_point).clamp(0, 15)

        # Dequantize
        W_dequant_grouped = (W_int - zero_point) * scale

        # Reshape back
        W_dequant = W_dequant_grouped.reshape(out_features, padded_in_features)

        # Remove padding if added
        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """
        Apply Group-Wise AWQ with Asymmetric Quantization and L2 Salience.
        Now with rounding statistics tracking.
        """
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data

        # Scale weight COLUMNS
        W_scaled = W * best_scales.unsqueeze(0)

        # Quantize with GROUP-WISE ASYMMETRIC quantization + STATS TRACKING
        W_quant = self.quantize_weight_groupwise_asymmetric_with_stats(W_scaled, name)

        # Divide by scales to restore original magnitude
        W_final = W_quant / best_scales.unsqueeze(0)

        # Update module weights
        module.weight.data = W_final

        # Store metadata
        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error
        }

    def calibrate(self, calibration_data, n_samples=500):
        """Run calibration on the dataset to collect activations."""
        print(f"\nCalibrating with {min(n_samples, len(calibration_data))} samples...")
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

            except Exception as e:
                if i % 100 == 0 and i > 0:
                    print(f"\nNote: Some samples skipped due to errors")
                continue

        self.remove_hooks()
        print(f"Calibration complete! Successfully processed {successful}/{n_samples} samples")

    def quantize_model(self):
        """Quantize all linear layers using Group-Wise AWQ with L2 Salience."""
        print("\n" + "=" * 80)
        print("Quantizing with Group-Wise AWQ ASYMMETRIC + L2 Salience + STATS")
        print("=" * 80)
        print("Tracking: Rounding UP/DOWN/EXACT counts per layer")
        print("=" * 80)

        quantized_count = 0
        skipped_count = 0

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        for name, module in tqdm(layer_names, desc="Quantizing layers"):
            try:
                self.quantize_layer(name, module)

                if quantized_count % 10 == 0 and quantized_count > 0:
                    if name in self.rounding_stats:
                        stats = self.rounding_stats[name]
                        print(f"\n  Layer {name}:")
                        print(f"    Rounded UP: {stats['rounded_up']:,} ({100*stats['rounded_up']/stats['total_weights']:.2f}%)")
                        print(f"    Rounded DOWN: {stats['rounded_down']:,} ({100*stats['rounded_down']/stats['total_weights']:.2f}%)")
                        print(f"    Exact: {stats['exact']:,} ({100*stats['exact']/stats['total_weights']:.2f}%)")

                quantized_count += 1

                # Clear activation data
                if name in self.activation_data:
                    del self.activation_data[name]

                if quantized_count % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️  Error quantizing layer {name}: {e}")
                import traceback
                traceback.print_exc()
                skipped_count += 1
                continue

        print(f"\n✅ Quantization complete!")
        print(f"   Total linear layers quantized: {quantized_count}")
        if skipped_count > 0:
            print(f"   ⚠️  Skipped {skipped_count} layers due to errors")

        self.activation_data = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def export_rounding_statistics(self, output_path):
        """
        Export detailed rounding statistics to CSV.
        """
        print(f"\n📊 Exporting rounding statistics to {output_path}...")

        rows = []
        for layer_name, stats in self.rounding_stats.items():
            row = {
                'layer_name': layer_name,
                'total_weights': stats['total_weights'],
                'rounded_up': stats['rounded_up'],
                'rounded_down': stats['rounded_down'],
                'exact': stats['exact'],
                'rounded_up_pct': 100 * stats['rounded_up'] / stats['total_weights'],
                'rounded_down_pct': 100 * stats['rounded_down'] / stats['total_weights'],
                'exact_pct': 100 * stats['exact'] / stats['total_weights'],
                'rounding_mag_mean': stats['rounding_mag_mean'],
                'rounding_mag_std': stats['rounding_mag_std'],
                'rounding_mag_max': stats['rounding_mag_max'],
                'rounding_mag_p90': stats['rounding_mag_percentile_90'],
                'rounding_mag_p99': stats['rounding_mag_percentile_99'],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        print(f"✅ Statistics exported to {output_path}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("ROUNDING STATISTICS SUMMARY")
        print("=" * 80)
        print(f"Total layers analyzed: {len(df)}")
        print(f"\nAggregate statistics across all layers:")
        print(f"  Total weights: {df['total_weights'].sum():,}")
        print(f"  Rounded UP: {df['rounded_up'].sum():,} ({100*df['rounded_up'].sum()/df['total_weights'].sum():.2f}%)")
        print(f"  Rounded DOWN: {df['rounded_down'].sum():,} ({100*df['rounded_down'].sum()/df['total_weights'].sum():.2f}%)")
        print(f"  Exact (no rounding): {df['exact'].sum():,} ({100*df['exact'].sum()/df['total_weights'].sum():.2f}%)")

        print(f"\nRounding magnitude statistics:")
        print(f"  Mean magnitude (avg across layers): {df['rounding_mag_mean'].mean():.6f}")
        print(f"  Max magnitude (across all layers): {df['rounding_mag_max'].max():.6f}")
        print(f"  90th percentile (avg): {df['rounding_mag_p90'].mean():.6f}")
        print(f"  99th percentile (avg): {df['rounding_mag_p99'].mean():.6f}")

        # Find layers with most/least rounding
        print(f"\nLayers with most rounding (rounded_up + rounded_down):")
        df['total_rounded'] = df['rounded_up'] + df['rounded_down']
        df['total_rounded_pct'] = 100 * df['total_rounded'] / df['total_weights']
        top_rounded = df.nlargest(5, 'total_rounded_pct')
        for _, row in top_rounded.iterrows():
            print(f"  {row['layer_name']}: {row['total_rounded_pct']:.2f}%")

        print(f"\nLayers with least rounding (most exact):")
        top_exact = df.nlargest(5, 'exact_pct')
        for _, row in top_exact.iterrows():
            print(f"  {row['layer_name']}: {row['exact_pct']:.2f}%")

        print("=" * 80)


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
        description="Group-Wise AWQ ASYMMETRIC L2 with Rounding Statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20, help="Grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_gw_awq_asym_l2_stats",
                       help="Output directory")
    parser.add_argument("--stats-csv", type=str, default="./rounding_stats.csv",
                       help="Path to save rounding statistics CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Group-Wise AWQ ASYMMETRIC L2 + ROUNDING STATISTICS")
    print("=" * 80)
    print("This script tracks:")
    print("  - How many weights are rounded UP at each layer")
    print("  - How many weights are rounded DOWN at each layer")
    print("  - How many weights require NO rounding (exact)")
    print("  - Distribution of rounding magnitudes")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Grid search points: {args.n_grid + 1}")
    print(f"Group size: {args.group_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Statistics CSV: {args.stats_csv}")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # Get model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb_before = (param_size + buffer_size) / 1024**2
    print(f"Model size before quantization: {size_mb_before:.2f} MB")

    # Load calibration data
    calib_texts = load_wikitext2(split="train", n_samples=args.n_calib)

    # Initialize quantizer with stats tracking
    quantizer = GroupWiseAWQAsymmetricL2StatsQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size
    )

    # Calibrate and quantize
    quantizer.calibrate(calib_texts, n_samples=args.n_calib)
    quantizer.quantize_model()

    # Export rounding statistics
    quantizer.export_rounding_statistics(args.stats_csv)

    # Get model size after
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb_after = (param_size + buffer_size) / 1024**2
    print(f"\nModel size after quantization: {size_mb_after:.2f} MB")
    print(f"Compression ratio: {size_mb_before / size_mb_after:.2f}x")

    # Save model
    print(f"\nSaving quantized model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("QUANTIZATION + STATISTICS COMPLETE!")
    print("=" * 80)
    print(f"Quantized model saved to: {args.output_dir}")
    print(f"Rounding statistics saved to: {args.stats_csv}")
    print("\nKey insights:")
    print("  ✓ Tracked rounding UP/DOWN/EXACT for each layer")
    print("  ✓ Analyzed rounding magnitude distributions")
    print("  ✓ Identified layers with most/least rounding")
    print("=" * 80)


if __name__ == "__main__":
    main()
