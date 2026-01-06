"""
Export Scaled Activation and Weight Data for Analysis

Exports from a specific layer after AWQ scaling:
1. E[X[:,j]] - Mean activation per input channel j (scaled)
2. W[:,out_channel_id] - Weight column for a specific output channel (scaled)

This allows detailed analysis of how AWQ scaling affects:
- Activation distributions per channel
- Weight patterns for specific neurons
- Relationship between activations and weights after scaling

Usage:
    python export_data.py --layer-id 3 --out-channel-id 0
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


class DataExporter:
    """Export scaled activation and weight data from AWQ quantization."""

    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20, group_size=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size

        # Storage
        self.activation_data = {}
        self.hooks = []
        self.export_data = {}

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
        """Compute E[X²] for activation salience."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        salience_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            salience_sum += x_flat.pow(2).sum(dim=0)

        salience = salience_sum / total_samples
        return salience

    @torch.no_grad()
    def get_activation_mean(self, name):
        """Compute E[X[:,j]] - mean activation per input channel."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        mean_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            mean_sum += x_flat.sum(dim=0)

        mean_activation = mean_sum / total_samples
        return mean_activation

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """
        Grid search for optimal scaling (simplified for export).
        Returns best scales found.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0

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
        Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        activation_salience = activation_salience.to(self.device)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid

            scales = activation_salience.pow(alpha).clamp(min=1e-5)
            W_scaled = W * scales.unsqueeze(0)

            # Simple quantization simulation (just scale for speed)
            X_compensated = X_search / scales.unsqueeze(0)
            Y_quant = torch.matmul(X_compensated, W_scaled.t())

            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

        del X_search, Y_orig

        return best_scales, best_alpha

    @torch.no_grad()
    def export_layer_data(self, layer_name, module, out_channel_id=0):
        """
        Export scaled activation and weight data for a specific layer.

        Args:
            layer_name: Name of the layer
            module: The linear module
            out_channel_id: Which output channel to export weights for

        Returns:
            Dictionary with exported data
        """
        print(f"\n📊 Exporting data for layer: {layer_name}")
        print(f"   Output channel ID: {out_channel_id}")

        # Get best scales from AWQ
        best_scales, best_alpha = self.search_best_scale(layer_name, module)

        print(f"   Best α: {best_alpha:.3f}")

        # Get activation mean E[X[:,j]]
        activation_mean = self.get_activation_mean(layer_name)
        if activation_mean is None:
            print("   ⚠️  No activation data available")
            return None

        # Compute scaled activation mean: E[Xs[:,j]] = E[X[:,j]] / scales
        scaled_activation_mean = activation_mean.cpu() / best_scales.cpu()

        # Get weight column W[:,out_channel_id]
        W = module.weight.data.cpu()
        if out_channel_id >= W.shape[0]:
            print(f"   ⚠️  Invalid out_channel_id {out_channel_id} (max: {W.shape[0]-1})")
            return None

        weight_column = W[out_channel_id, :]  # Shape: [in_features]

        # Compute scaled weight column: W_scaled[:,out_channel_id] = W[:,out_channel_id] * scales
        scaled_weight_column = weight_column * best_scales.cpu()

        export_dict = {
            'layer_name': layer_name,
            'out_channel_id': out_channel_id,
            'best_alpha': best_alpha,
            'in_features': W.shape[1],
            'out_features': W.shape[0],
            # Activation data (per input channel)
            'activation_mean_original': activation_mean.cpu().numpy(),  # E[X[:,j]]
            'activation_mean_scaled': scaled_activation_mean.numpy(),    # E[Xs[:,j]]
            # Weight data (for specific output channel)
            'weight_column_original': weight_column.numpy(),             # W[out_id, :]
            'weight_column_scaled': scaled_weight_column.numpy(),        # W_scaled[out_id, :]
            # Scales
            'scales': best_scales.cpu().numpy(),                         # scales[j]
        }

        print(f"   ✅ Exported {W.shape[1]} input channels")

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
        out_id = export_dict['out_channel_id']
        prefix = f"{layer_name}_out{out_id}"

        # Save as CSV (for easy viewing)
        df_data = {
            'input_channel_id': np.arange(export_dict['in_features']),
            'E[X]': export_dict['activation_mean_original'],
            'E[Xs]': export_dict['activation_mean_scaled'],
            'W': export_dict['weight_column_original'],
            'W_scaled': export_dict['weight_column_scaled'],
            'scale': export_dict['scales'],
        }
        df = pd.DataFrame(df_data)

        csv_path = os.path.join(output_dir, f"{prefix}_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved CSV: {csv_path}")

        # Save as numpy (for numerical analysis)
        npz_path = os.path.join(output_dir, f"{prefix}_data.npz")
        np.savez(
            npz_path,
            layer_name=export_dict['layer_name'],
            out_channel_id=export_dict['out_channel_id'],
            best_alpha=export_dict['best_alpha'],
            activation_mean_original=export_dict['activation_mean_original'],
            activation_mean_scaled=export_dict['activation_mean_scaled'],
            weight_column_original=export_dict['weight_column_original'],
            weight_column_scaled=export_dict['weight_column_scaled'],
            scales=export_dict['scales'],
        )
        print(f"✅ Saved NumPy: {npz_path}")

        # Save metadata
        metadata_path = os.path.join(output_dir, f"{prefix}_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Layer: {export_dict['layer_name']}\n")
            f.write(f"Output Channel ID: {export_dict['out_channel_id']}\n")
            f.write(f"Best Alpha: {export_dict['best_alpha']:.6f}\n")
            f.write(f"Input Features: {export_dict['in_features']}\n")
            f.write(f"Output Features: {export_dict['out_features']}\n")
            f.write(f"\nData Files:\n")
            f.write(f"  CSV: {prefix}_data.csv\n")
            f.write(f"  NumPy: {prefix}_data.npz\n")
            f.write(f"\nColumns in CSV:\n")
            f.write(f"  input_channel_id: Index j of input channel\n")
            f.write(f"  E[X]: Original activation mean E[X[:,j]]\n")
            f.write(f"  E[Xs]: Scaled activation mean E[X[:,j]/scales[j]]\n")
            f.write(f"  W: Original weight W[{out_id}, j]\n")
            f.write(f"  W_scaled: Scaled weight W[{out_id}, j] * scales[j]\n")
            f.write(f"  scale: AWQ scale factor scales[j] = E[X[:,j]²]^α\n")

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
        description="Export scaled activation and weight data from AWQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--layer-id", type=int, default=3,
                       help="Layer ID to export (e.g., 3 for model.layers.3.*)")
    parser.add_argument("--layer-type", type=str, default="mlp.gate_proj",
                       help="Layer type suffix (e.g., mlp.gate_proj, self_attn.q_proj)")
    parser.add_argument("--out-channel-id", type=int, default=0,
                       help="Output channel ID to export weight column for")
    parser.add_argument("--n-calib", type=int, default=128,
                       help="Calibration samples")
    parser.add_argument("--n-grid", type=int, default=20,
                       help="Grid search points for AWQ")
    parser.add_argument("--output-dir", type=str, default="./exported_data",
                       help="Output directory")
    args = parser.parse_args()

    model_name = "openbmb/MiniCPM-2B-sft-bf16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("AWQ DATA EXPORTER")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Target layer: model.layers.{args.layer_id}.{args.layer_type}")
    print(f"Output channel ID: {args.out_channel_id}")
    print(f"Calibration samples: {args.n_calib}")
    print(f"Output directory: {args.output_dir}")
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

    # Load calibration data
    calib_texts = load_wikitext2(split="train", n_samples=args.n_calib)

    # Initialize exporter
    exporter = DataExporter(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=128
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
        out_channel_id=args.out_channel_id
    )

    if export_dict is not None:
        exporter.save_export_data(export_dict, args.output_dir)

        print("\n" + "=" * 80)
        print("EXPORT COMPLETE!")
        print("=" * 80)
        print(f"Exported data for: {target_layer_name}")
        print(f"Output channel: {args.out_channel_id}")
        print(f"Files saved to: {args.output_dir}")
        print("\nYou can now analyze:")
        print(f"  1. How AWQ scaling affects activations E[Xs]")
        print(f"  2. How AWQ scaling affects weights W_scaled")
        print(f"  3. Relationship between scaled activations and weights")
        print("=" * 80)


if __name__ == "__main__":
    main()
