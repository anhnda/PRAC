"""
Wanda Pruning with Moderate Weight Correction - XL Version

Combines Wanda pruning with moderate weight correction:
1. Apply Wanda pruning (Score_ij = |W_ij| * ||X_j||_2)
2. Select "moderate" positions based on |JS_mean(X_j)|
3. Correct weights at moderate positions to minimize reconstruction error
4. Fully vectorized across all output channels

Key Features:
- L2 norm salience for Wanda scoring
- James-Stein mean estimator for weight correction
- Moderate position selection using Kneedle algorithm
- Vectorized correction across all channels
- Batched sequential processing for memory efficiency

Usage:
    python prac_xl.py --sparsity 0.5 --percent-change 0.05
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
import gc

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  Warning: psutil not installed. Memory monitoring disabled.")

from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data


def find_knee_point(values, tolerance_offset=0.0):
    """
    Find knee point in sorted values using Kneedle algorithm.

    Args:
        values: 1D tensor of sorted values
        tolerance_offset: Additional offset to add to knee point

    Returns:
        index of knee point
    """
    n = len(values)
    if n < 3:
        return n // 2

    # Convert to numpy (handle bfloat16)
    if torch.is_tensor(values):
        y = values.cpu().float().numpy()
    else:
        y = np.array(values)

    # Normalize to [0, 1]
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return n // 2

    y_norm = (y - y_min) / (y_max - y_min)
    x_norm = np.linspace(0, 1, n)

    # Compute distances from reference line
    y_line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    distances = np.abs(y_norm - y_line)

    # Find knee point
    knee_idx = np.argmax(distances)

    # Apply tolerance offset
    if knee_idx < n - 1:
        offset_indices = int(tolerance_offset * n)
        knee_idx = min(knee_idx + offset_indices, n - 1)
        knee_idx = max(knee_idx, 0)

    return knee_idx


def compute_james_stein_mean(raw_means, variance_estimate=None):
    """
    Apply James-Stein shrinkage estimator to channel-wise means.

    Formula:
        μ̂_JS[j] = μ̄ + (1 - c) × (X̄[j] - μ̄)
        where c = (p - 2) × σ² / Σ(X̄[j] - μ̄)²
    """
    p = len(raw_means)

    if p < 3:
        return raw_means

    # Compute grand mean
    grand_mean = raw_means.mean()

    # Compute deviations
    deviations = raw_means - grand_mean

    # Sum of squared deviations
    sum_sq_dev = (deviations ** 2).sum()

    if sum_sq_dev < 1e-10:
        return raw_means

    # Estimate variance
    if variance_estimate is None:
        variance_estimate = ((raw_means - grand_mean).abs().mean()) ** 2
        variance_estimate = variance_estimate.clamp(min=1e-8)

    # Compute shrinkage factor
    shrinkage_factor = ((p - 2) * variance_estimate) / sum_sq_dev
    shrinkage_factor = shrinkage_factor.clamp(0, 1)

    # Apply James-Stein shrinkage
    js_means = grand_mean + (1 - shrinkage_factor) * deviations

    return js_means


class WandaPrunerWithCorrection:
    """
    Wanda Pruning with Moderate Weight Correction.
    Fully vectorized for all output channels.
    """

    def __init__(self, model, tokenizer, device="cuda", sparsity=0.5,
                 max_tokens_per_sample=512, use_correction=True,
                 knee_tolerance=0.0, offset_percent=0.0, percent_change=0.05,
                 max_correction_magnitude=0.05):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sparsity = sparsity
        self.max_tokens_per_sample = max_tokens_per_sample
        self.use_correction = use_correction
        self.knee_tolerance = knee_tolerance
        self.offset_percent = offset_percent
        self.percent_change = percent_change
        self.max_correction_magnitude = max_correction_magnitude

        # Storage
        self.activation_data = {}
        self.hooks = []
        self.layer_stats = {}

        print(f"\n[Wanda Pruner with Moderate Correction - XL Version]")
        print(f"  Target sparsity: {sparsity*100:.1f}% (keep {(1-sparsity)*100:.1f}%)")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample")
        print(f"  Salience metric: ||X_j||_2 (L2 norm)")
        print(f"  Scoring: Score_ij = |W_ij| * ||X_j||_2")
        if use_correction:
            print(f"  Moderate correction: ENABLED")
            print(f"    - Correction percent: {percent_change*100:.1f}% of weights")
            print(f"    - Max magnitude: {max_correction_magnitude*100:.1f}% of original weight")
            print(f"    - Knee tolerance: {knee_tolerance}")
            print(f"    - Offset percent: {offset_percent*100:.1f}%")
        else:
            print(f"  Moderate correction: DISABLED")
        print(f"  Special: lm_head is NOT pruned")

    @torch.no_grad()
    def get_activation_stats(self, name):
        """
        Compute L2 salience and James-Stein mean using float32 precision.

        Returns:
            tuple: (l2_salience, js_mean, raw_mean)
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None, None, None

        X_list = self.activation_data[name]
        total_samples = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in X_list)
        in_features = X_list[0].shape[-1]

        # Use float32 for accumulation
        l2_sum = torch.zeros(in_features, dtype=torch.float32)
        mean_sum = torch.zeros(in_features, dtype=torch.float32)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            l2_sum += x_flat.pow(2).sum(dim=0)
            mean_sum += x_flat.sum(dim=0)

        l2_salience = torch.sqrt(l2_sum / total_samples)
        raw_mean = mean_sum / total_samples

        # Apply James-Stein estimator
        js_mean = compute_james_stein_mean(raw_mean)

        return l2_salience, js_mean, raw_mean

    @torch.no_grad()
    def select_moderate_positions(self, js_mean, debug=False):
        """
        Select moderate positions based on |js_mean|.

        Algorithm:
        1. Sort |js_mean| in descending order
        2. Apply Kneedle on first half
        3. Start from knee + offset
        4. Select percent_change * in_features positions

        Returns:
            selected_indices: Tensor of selected positions
            knee_info: Dict with selection statistics
        """
        abs_js_mean = js_mean.abs()
        in_features = len(abs_js_mean)

        # Sort descending
        sorted_values, sorted_indices = torch.sort(abs_js_mean, descending=True)

        # Apply Kneedle on first half
        first_half = sorted_values[:in_features // 2]
        knee_idx = find_knee_point(first_half, tolerance_offset=self.knee_tolerance)

        # Start position
        offset_indices = int(self.offset_percent * in_features)
        start_idx = knee_idx + offset_indices
        start_idx = max(0, min(start_idx, in_features - 1))

        # Number to select
        num_to_select = max(1, int(self.percent_change * in_features))
        end_idx = min(start_idx + num_to_select, in_features)

        # Adjust if needed
        if end_idx == in_features and start_idx > 0:
            start_idx = max(0, in_features - num_to_select)

        actual_num = end_idx - start_idx

        # Get selected indices (in original order)
        selected_indices = sorted_indices[start_idx:end_idx]

        knee_info = {
            'knee_idx': knee_idx,
            'knee_value': sorted_values[knee_idx].item(),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'num_selected': actual_num,
            'selection_pct': actual_num / in_features * 100,
        }

        if debug:
            print(f"    Moderate selection: {actual_num} positions ({knee_info['selection_pct']:.2f}%)")
            print(f"    Knee idx: {knee_idx}, value: {knee_info['knee_value']:.6f}")

        return selected_indices, knee_info

    @torch.no_grad()
    def correct_weights_vectorized(self, W_orig, W_pruned, js_mean,
                                   selected_positions, debug=False):
        """
        Vectorized weight correction across all output channels.

        For each output channel i:
            error[i] = (W_pruned[i,:] - W_orig[i,:]) · js_mean

        For each moderate position j:
            delta_W[i,j] = -error[i] * sign(js_mean[j]) / sum(|js_mean[selected]|)

        Args:
            W_orig: Original weights [out_features, in_features]
            W_pruned: Pruned weights [out_features, in_features]
            js_mean: James-Stein mean [in_features]
            selected_positions: Indices of moderate positions
            debug: Print debug info

        Returns:
            W_corrected: Corrected weights
            correction_stats: Dictionary with statistics
        """
        # Store original dtype and convert everything to float32 for computation
        # This avoids bfloat16 compatibility issues with certain operations
        weight_dtype = W_orig.dtype

        # Convert to float32 for computation
        W_orig_f32 = W_orig.float()
        W_pruned_f32 = W_pruned.float()
        js_mean_f32 = js_mean.float()

        # Compute reconstruction errors for all channels
        # error[i] = (W_pruned[i,:] - W_orig[i,:]) · js_mean
        W_diff = W_pruned_f32 - W_orig_f32
        errors = torch.matmul(W_diff, js_mean_f32)  # [out_features]

        # Sum of |js_mean| at selected positions
        sum_abs_js_mean = js_mean_f32[selected_positions].abs().sum()

        if sum_abs_js_mean < 1e-10:
            if debug:
                print(f"    ⚠️ Sum too small, no correction")
            return W_pruned.clone(), None

        # Vectorized correction
        W_corrected = W_pruned_f32.clone()

        # Track clamping statistics
        num_clamped = 0
        total_corrections = 0

        # For each moderate position j:
        # delta_W[:,j] = -errors * sign(js_mean[j]) / sum_abs_js_mean
        # Constrain: |delta_W[i,j]| <= max_correction_magnitude * |W_orig[i,j]|
        for j in selected_positions:
            sign_val = torch.sign(js_mean_f32[j])
            delta_W_j = -errors * sign_val / sum_abs_js_mean

            # Compute magnitude constraint: max 5% of original weight magnitude
            max_change = self.max_correction_magnitude * W_orig_f32[:, j].abs()

            # Clamp delta to constraint
            delta_W_j_clamped = torch.clamp(delta_W_j, -max_change, max_change)

            # Track how many were clamped
            was_clamped = (delta_W_j.abs() > max_change).sum().item()
            num_clamped += was_clamped
            total_corrections += len(delta_W_j)

            W_corrected[:, j] += delta_W_j_clamped

        # Convert back to original dtype
        W_corrected = W_corrected.to(weight_dtype)

        # Compute correction statistics (in float32 for accuracy)
        errors_after = torch.matmul(W_corrected.float() - W_orig_f32, js_mean_f32)
        error_reduction = (errors.abs() - errors_after.abs()).mean().item()

        correction_stats = {
            'error_before_mean': errors.abs().mean().item(),
            'error_after_mean': errors_after.abs().mean().item(),
            'error_reduction_mean': error_reduction,
            'num_corrected_positions': len(selected_positions),
            'num_clamped': num_clamped,
            'total_corrections': total_corrections,
            'clamp_percentage': num_clamped / total_corrections * 100 if total_corrections > 0 else 0,
        }

        if debug:
            print(f"    Error before: {correction_stats['error_before_mean']:.6f}")
            print(f"    Error after: {correction_stats['error_after_mean']:.6f}")
            print(f"    Reduction: {error_reduction:.6f}")
            print(f"    Clamped: {num_clamped}/{total_corrections} ({correction_stats['clamp_percentage']:.1f}%)")

        return W_corrected, correction_stats

    @torch.no_grad()
    def prune_weights_wanda(self, W, salience, sparsity):
        """
        Vectorized Wanda pruning.

        Returns:
            W_pruned: Pruned weights
            actual_sparsity: Achieved sparsity
        """
        in_features = W.shape[1]

        # Compute scores
        scores = W.abs() * salience.unsqueeze(0)

        # Top-k selection
        num_to_keep = int(in_features * (1 - sparsity))

        if num_to_keep > 0:
            _, top_indices = torch.topk(scores, num_to_keep, dim=1, largest=True)
            mask = torch.zeros_like(W, dtype=torch.bool)
            mask.scatter_(1, top_indices, True)
        else:
            mask = torch.zeros_like(W, dtype=torch.bool)

        W_pruned = W * mask.float()
        actual_sparsity = (W_pruned == 0).float().mean().item()

        return W_pruned, actual_sparsity

    @torch.no_grad()
    def prune_layer(self, name, module, debug=False):
        """
        Prune layer with optional moderate weight correction.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            print(f"  ⚠️  No activation data for {name}, skipping")
            return

        # Get activation statistics
        l2_salience, js_mean, _ = self.get_activation_stats(name)
        if l2_salience is None:
            if debug:
                print(f"  DEBUG: No activation stats for {name}, skipping")
            return

        if debug:
            print(f"  DEBUG: Layer {name}")
            print(f"    L2 salience shape: {l2_salience.shape}")
            print(f"    JS mean range: [{js_mean.min():.6f}, {js_mean.max():.6f}]")

        W = module.weight.data
        original_dtype = W.dtype

        # Move to device
        W_device = W.to(self.device)
        l2_salience_device = l2_salience.to(self.device).to(W.dtype)
        js_mean_device = js_mean.to(self.device).to(W.dtype)

        # Step 1: Wanda pruning
        W_pruned, actual_sparsity = self.prune_weights_wanda(
            W_device, l2_salience_device, self.sparsity
        )

        # Step 2: Moderate weight correction (if enabled)
        if self.use_correction:
            # Select moderate positions
            selected_positions, knee_info = self.select_moderate_positions(
                js_mean_device, debug=debug
            )

            # Correct weights
            W_final, correction_stats = self.correct_weights_vectorized(
                W_device, W_pruned, js_mean_device, selected_positions, debug=debug
            )
        else:
            W_final = W_pruned
            knee_info = None
            correction_stats = None

        # Update module weights
        module.weight.data = W_final.to(original_dtype)

        # Store statistics
        self.layer_stats[name] = {
            'target_sparsity': self.sparsity,
            'actual_sparsity': actual_sparsity,
            'l2_salience_mean': l2_salience.mean().item(),
            'js_mean_mean': js_mean.mean().item(),
            'use_correction': self.use_correction,
        }

        if self.use_correction and correction_stats is not None:
            self.layer_stats[name].update({
                'knee_info': knee_info,
                'correction_stats': correction_stats,
            })

        if debug:
            print(f"  → Sparsity: {actual_sparsity*100:.2f}%")

        del W_device, l2_salience_device, js_mean_device, W_pruned, W_final
        torch.cuda.empty_cache()

    def get_hook(self, name):
        """Create hook for activation capture."""
        def hook(module, input, output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Subsample tokens if needed
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                indices = torch.randperm(seq_len, device=inp.device)[:self.max_tokens_per_sample]
                indices = indices.sort()[0]
                inp = inp[:, indices, :]

            # Store on CPU as float32
            self.activation_data[name].append(inp.detach().cpu().float())
        return hook

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """Calibrate batch of layers."""
        self.activation_data = {}

        handles = []
        for name, module in layer_names_batch:
            handle = module.register_forward_hook(self.get_hook(name))
            handles.append((name, handle))

        successful = 0
        with torch.no_grad():
            for i, text in enumerate(calibration_data[:n_samples]):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    _ = self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1
                    del inputs

                    if (i + 1) % 10 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                except Exception:
                    continue

        for _, handle in handles:
            handle.remove()

        if successful == 0:
            print(f"\n❌ FATAL: No successful forward passes!")

        for name, _ in layer_names_batch:
            if name not in self.activation_data:
                self.activation_data[name] = []

    def prune_model_sequential(self, calibration_data, n_samples=500, layer_batch_size=16):
        """Batched sequential pruning with correction."""
        print("\n" + "=" * 80)
        print("BATCHED SEQUENTIAL PRUNING WITH MODERATE CORRECTION (XL Version)")
        print("=" * 80)

        if HAS_PSUTIL:
            initial_ram = psutil.virtual_memory().percent
            print(f"Initial System RAM: {initial_ram:.1f}%")

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        print(f"\nFound {len(layer_names)} linear layers")
        print(f"Batch size: {layer_batch_size} layers per batch")
        num_batches = (len(layer_names) + layer_batch_size - 1) // layer_batch_size
        print(f"Total batches: {num_batches}")

        pruned_count = 0
        skipped_count = 0

        for batch_idx in range(num_batches):
            batch_start = batch_idx * layer_batch_size
            batch_end = min(batch_start + layer_batch_size, len(layer_names))
            batch_layers = layer_names[batch_start:batch_end]

            print(f"\n{'='*60}")
            print(f"Batch {batch_idx + 1}/{num_batches}: Layers {batch_start}-{batch_end-1}")
            print(f"{'='*60}")

            # Calibrate
            self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

            # Prune
            for _, (name, module) in enumerate(tqdm(batch_layers, desc=f"Pruning Batch {batch_idx+1}")):
                try:
                    is_lmhead = 'lm_head' in name.lower() or name.endswith('lm_head')

                    if is_lmhead:
                        print(f"\n  ⏭️  Skipping {name} (keeping original)")
                        skipped_count += 1
                    else:
                        debug = (pruned_count < 2)
                        self.prune_layer(name, module, debug=debug)
                        pruned_count += 1

                except Exception as e:
                    print(f"\n⚠️  Error pruning {name}: {e}")
                    skipped_count += 1
                    continue

            # Cleanup
            self.activation_data = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                ram_pct = psutil.virtual_memory().percent
                print(f"Batch {batch_idx+1} complete. RAM: {ram_pct:.1f}%")

        print(f"\n✅ Sequential Pruning Complete!")
        print(f"   Total layers pruned: {pruned_count}/{len(layer_names)}")
        print(f"   Skipped layers: {skipped_count}")

        # Print statistics
        if self.layer_stats:
            sparsities = [info['actual_sparsity'] for info in self.layer_stats.values()]
            print(f"\nSparsity statistics:")
            print(f"  Mean: {np.mean(sparsities)*100:.2f}%")
            print(f"  Median: {np.median(sparsities)*100:.2f}%")

            if self.use_correction:
                corrected_layers = [k for k, v in self.layer_stats.items()
                                   if v.get('correction_stats') is not None]
                if corrected_layers:
                    errors_before = [self.layer_stats[k]['correction_stats']['error_before_mean']
                                    for k in corrected_layers]
                    errors_after = [self.layer_stats[k]['correction_stats']['error_after_mean']
                                   for k in corrected_layers]
                    reductions = [self.layer_stats[k]['correction_stats']['error_reduction_mean']
                                 for k in corrected_layers]
                    clamp_pcts = [self.layer_stats[k]['correction_stats']['clamp_percentage']
                                 for k in corrected_layers]

                    print(f"\nModerate correction statistics ({len(corrected_layers)} layers):")
                    print(f"  Mean error before: {np.mean(errors_before):.6f}")
                    print(f"  Mean error after: {np.mean(errors_after):.6f}")
                    print(f"  Mean reduction: {np.mean(reductions):.6f}")
                    print(f"  Mean clamp %: {np.mean(clamp_pcts):.1f}% (range: {np.min(clamp_pcts):.1f}% - {np.max(clamp_pcts):.1f}%)")

        # Final cleanup
        self.activation_data = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def load_wikitext2_simple(n_samples=128):
    print(f"Loading WikiText-2...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    return texts[:n_samples]


def main():
    parser = argparse.ArgumentParser(
        description="Wanda Pruning with Moderate Weight Correction (XL Version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--sparsity", type=float, default=0.5,
                       help="Target sparsity (0.5 = prune 50%%)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens per sample")
    parser.add_argument("--use-correction", action="store_true", default=True,
                       help="Enable moderate weight correction")
    parser.add_argument("--no-correction", dest="use_correction", action="store_false",
                       help="Disable moderate weight correction")
    parser.add_argument("--knee-tolerance", type=float, default=0.0,
                       help="Tolerance for Kneedle algorithm")
    parser.add_argument("--offset-percent", type=float, default=0.0,
                       help="Offset from knee point (as fraction)")
    parser.add_argument("--percent-change", type=float, default=0.05,
                       help="Percentage of weights to correct (default 5%%)")
    parser.add_argument("--max-correction-magnitude", type=float, default=0.05,
                       help="Max correction magnitude as fraction of original weight (default 5%%)")
    parser.add_argument("--output-dir", type=str, default="./pruned_models/model_prac_xl",
                       help="Output directory")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3",
                       help="Model path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--layer-batch-size", type=int, default=16,
                       help="Layers per batch")
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Wanda Pruning with Moderate Weight Correction (XL Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Target Sparsity: {args.sparsity*100:.1f}%")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"Moderate correction: {args.use_correction}")
    if args.use_correction:
        print(f"  Correction percent: {args.percent_change*100:.1f}%")
        print(f"  Knee tolerance: {args.knee_tolerance}")
        print(f"  Offset percent: {args.offset_percent}")
    print("=" * 80)

    # Load model
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load calibration data
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib,
                                             seqlen=2048, seed=args.seed,
                                             cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        calib_texts = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib,
                                                     seqlen=2048, seed=args.seed,
                                                     cache_dir=args.cache_dir)

    # Initialize pruner
    pruner = WandaPrunerWithCorrection(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sparsity=args.sparsity,
        max_tokens_per_sample=args.max_tokens_per_sample,
        use_correction=args.use_correction,
        knee_tolerance=args.knee_tolerance,
        offset_percent=args.offset_percent,
        percent_change=args.percent_change,
        max_correction_magnitude=args.max_correction_magnitude
    )

    # Prune model
    pruner.prune_model_sequential(calib_texts, n_samples=args.n_calib,
                                  layer_batch_size=args.layer_batch_size)

    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("PRUNING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
