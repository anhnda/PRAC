"""
Wanda Pruning with Greedy Direct Error Correction (prac_h.py)

Implements greedy iterative correction using DIRECT activation error measurement.

Mathematical Foundation:
    Directly minimize the reconstruction error on calibration data:

    Error(W) = ||X @ W^T - X @ W_orig^T||²

    where:
    - X = calibration activations [num_samples, in_features]
    - W = current weights [out_features, in_features]
    - W_orig = original dense weights

    NO Hessian needed! Just direct error measurement on stored activations.

Key Differences from prac_full.py:
1. **Direct error measurement**: Computes ||XW^T - XW_orig^T||² on actual activations
2. **No Hessian**: Stores activation samples instead of computing XX^T (much faster!)
3. **Greedy search**: Process positions one-by-one, stop when no improvement
4. **Fixed magnitude**: 1% correction (default) avoids clamping issues
5. **Simpler**: No linear system solving, just error evaluation

Algorithm:
    1. Select MODERATE candidate positions (Kneedle algorithm)
    2. For each candidate position j:
       a. For each output channel i:
          - Try ΔW_ij = +1% * W_orig_ij, compute J(ΔW_i)
          - Try ΔW_ij = -1% * W_orig_ij, compute J(ΔW_i)
          - Pick direction with lowest J
       b. Sum improvements across all channels
       c. If total improvement > 0: apply corrections, continue
       d. If no improvement: stop (early termination)

Benefits:
    - **Mathematically correct**: Directly optimizes what we care about (activation error)
    - **Efficient**: Per-channel objectives computed independently
    - **Adaptive**: Natural early stopping when improvements stop
    - **Stable**: Small fixed magnitude prevents disruption

Usage:
    python prac_h.py --sparsity 0.5 --percent-change 0.05 --correction-magnitude 0.01
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


class WandaPrunerWithGreedyHessian:
    """
    Wanda Pruning with Greedy Direct Error Correction.

    Uses greedy iterative approach with fixed magnitude corrections.
    Measures direct activation error ||XW^T - XW_orig^T||² on calibration data.
    """

    def __init__(self, model, tokenizer, device="cuda", sparsity=0.5,
                 max_tokens_per_sample=512, use_correction=True,
                 knee_tolerance=0.0, offset_percent=0.0, percent_change=0.05,
                 correction_magnitude=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sparsity = sparsity
        self.max_tokens_per_sample = max_tokens_per_sample
        self.use_correction = use_correction
        self.knee_tolerance = knee_tolerance
        self.offset_percent = offset_percent
        self.percent_change = percent_change
        self.correction_magnitude = correction_magnitude

        # Storage for activation samples (for direct error measurement)
        # Memory: O(num_samples * hidden_dim) - much cheaper than Hessian!
        self.activation_stats = {}  # Dict of {name: {'activations': tensor, 'mean_sum': tensor, 'count': int}}
        self.hooks = []
        self.layer_stats = {}

        print(f"\n[Wanda Pruner with Greedy Direct Error Correction]")
        print(f"  Target sparsity: {sparsity*100:.1f}% (keep {(1-sparsity)*100:.1f}%)")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample")
        print(f"  Salience metric: ||X_j||_2 (L2 norm from activations)")
        print(f"  Scoring: Score_ij = |W_ij| * ||X_j||_2")
        print(f"  Error measurement: Direct ||X·W^T - X·W_orig^T||² on calibration data")
        if use_correction:
            print(f"  Greedy direct error correction: ENABLED")
            print(f"    - Selection strategy: MODERATE (Kneedle algorithm)")
            print(f"    - Max corrections: {percent_change*100:.1f}% of surviving weights")
            print(f"    - Fixed magnitude: {correction_magnitude*100:.1f}% per weight")
            print(f"    - Method: Greedy iterative with early stopping")
            print(f"    - Error: Measured directly on calibration activations")
        else:
            print(f"  Correction: DISABLED")
        print(f"  Special: lm_head is NOT pruned")

    @torch.no_grad()
    def get_activation_stats(self, name):
        """
        Compute L2 salience and James-Stein mean from stored activations.

        Returns:
            tuple: (l2_salience, js_mean, raw_mean, activations)
        """
        if name not in self.activation_stats:
            return None, None, None, None

        stats = self.activation_stats[name]

        # Check if we have any data
        if stats['count'] == 0 or 'activations' not in stats:
            return None, None, None, None

        # Concatenate activation samples if still a list
        if isinstance(stats['activations'], list):
            if len(stats['activations']) == 0:
                return None, None, None, None
            stats['activations'] = torch.cat(stats['activations'], dim=0)  # [total_tokens, hidden_dim]

        # Get stored activations [num_samples, hidden_dim]
        activations = stats['activations']

        # L2 salience: sqrt(mean(X_j^2))
        l2_salience = torch.sqrt((activations ** 2).mean(dim=0))

        # Mean from accumulated sums
        raw_mean = stats['mean_sum'] / stats['count']

        # Apply James-Stein estimator
        js_mean = compute_james_stein_mean(raw_mean)

        return l2_salience, js_mean, raw_mean, activations

    @torch.no_grad()
    def select_moderate_positions(self, js_mean, prune_mask, debug=False):
        """
        Select MODERATE weights using Kneedle algorithm.

        Returns candidates sorted by salience (descending) for greedy processing.
        """
        # Count surviving weights per input channel
        survival_count = prune_mask.sum(dim=0)  # [in_features]
        is_surviving = survival_count > 0

        # Filter to surviving channels only
        surviving_indices = torch.where(is_surviving)[0]

        if len(surviving_indices) == 0:
            print(f"    ⚠️ No surviving weights to correct!")
            return torch.tensor([], dtype=torch.long), None

        abs_js_mean_surviving = js_mean[surviving_indices].abs()

        # Sort descending by salience (L2 norm)
        sorted_values, sorted_order = torch.sort(abs_js_mean_surviving, descending=True)
        sorted_indices = surviving_indices[sorted_order]

        num_surviving = len(surviving_indices)
        max_to_select = max(1, int(self.percent_change * num_surviving))

        # Apply Kneedle on first half to find moderate region
        first_half = sorted_values[:num_surviving // 2]
        if len(first_half) >= 3:
            knee_idx = find_knee_point(first_half, tolerance_offset=self.knee_tolerance)
            offset_indices = int(self.offset_percent * num_surviving)
            start_idx = knee_idx + offset_indices
            start_idx = max(0, min(start_idx, num_surviving - 1))
        else:
            start_idx = 0

        # Candidates: from start_idx to end of sorted list (or max_to_select)
        # Greedy will select from these, stopping when error stops decreasing
        end_idx = min(start_idx + max_to_select, num_surviving)
        if end_idx == num_surviving and start_idx > 0:
            start_idx = max(0, num_surviving - max_to_select)

        candidate_indices = sorted_indices[start_idx:end_idx]

        knee_info = {
            'knee_idx': knee_idx if len(first_half) >= 3 else 0,
            'knee_value': sorted_values[min(start_idx, len(sorted_values) - 1)].item() if len(sorted_values) > 0 else 0,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'num_candidates': len(candidate_indices),
            'num_surviving': num_surviving,
            'candidate_pct': len(candidate_indices) / num_surviving * 100 if num_surviving > 0 else 0,
        }

        if debug:
            print(f"    Surviving channels: {num_surviving}")
            print(f"    MODERATE candidates: {knee_info['num_candidates']} positions ({knee_info['candidate_pct']:.2f}%)")
            if len(candidate_indices) > 0:
                cand_values = abs_js_mean_surviving[sorted_order[start_idx:end_idx]]
                print(f"    Salience range: [{cand_values[0].item():.6f}, {cand_values[-1].item():.6f}]")

        return candidate_indices, knee_info

    @torch.no_grad()
    def correct_weights_greedy_direct(self, W_orig, W_pruned, js_mean, activations,
                                      candidate_positions, prune_mask, debug=False):
        """
        Greedy iterative weight correction using DIRECT activation error.

        Algorithm:
            1. Start with pruned weights
            2. For each candidate position j:
               a. Try correction: W[:, j] ± 1%
               b. Compute DIRECT error: ||X @ W^T - X @ W_orig^T||²
               c. Pick direction with lowest error
               d. If error decreases: apply and continue
               e. If no improvement: stop
            3. Return best weights found

        Args:
            W_orig: Original weights [out_features, in_features]
            W_pruned: Pruned weights [out_features, in_features]
            js_mean: James-Stein mean [in_features] (not used)
            activations: Calibration activations [num_samples, in_features]
            candidate_positions: Indices of candidate positions (MODERATE weights)
            prune_mask: Boolean mask [out_features, in_features] (True = kept)
            debug: Print debug info

        Returns:
            W_corrected: Corrected weights
            correction_stats: Dictionary with statistics
        """
        # Convert to float32 for numerical stability
        weight_dtype = W_orig.dtype
        W_orig_f32 = W_orig.float()
        W_pruned_f32 = W_pruned.float()

        # Activations already on GPU
        X = activations.float().to(self.device)  # [num_samples, in_features]

        if len(candidate_positions) == 0:
            if debug:
                print(f"    ⚠️ No candidates, no correction")
            return W_pruned.clone(), None

        # Helper function to compute DIRECT activation error
        def compute_error(W):
            """
            Compute ||X @ W^T - X @ W_orig^T||²

            This is the TRUE reconstruction error on calibration data.
            """
            # activations @ weights^T = outputs
            outputs_orig = X @ W_orig_f32.t()  # [num_samples, out_features]
            outputs_curr = X @ W.t()            # [num_samples, out_features]
            error = ((outputs_curr - outputs_orig) ** 2).sum().item()
            return error

        # Initial error
        W_current = W_pruned_f32.clone()
        error_current = compute_error(W_current)
        error_initial = error_current

        # Greedy correction: process each candidate position
        num_applied = 0
        stopped_early = False

        for idx, j in enumerate(candidate_positions):
            j_item = j.item()

            # Only correct surviving weights at this position
            survivors_at_j = prune_mask[:, j]  # [out_features]
            if not survivors_at_j.any():
                continue

            # For each output channel at position j, try ±correction and pick best
            best_corrections = []  # List of (i, direction) tuples

            for i in range(W_current.shape[0]):
                if not survivors_at_j[i]:
                    continue

                # Try +correction
                W_test = W_current.clone()
                W_test[i, j] += self.correction_magnitude * W_orig_f32[i, j]
                error_pos = compute_error(W_test)

                # Try -correction
                W_test = W_current.clone()
                W_test[i, j] -= self.correction_magnitude * W_orig_f32[i, j]
                error_neg = compute_error(W_test)

                # Pick best direction (lowest error)
                if error_pos < error_current and error_pos <= error_neg:
                    best_corrections.append((i, +1))
                elif error_neg < error_current:
                    best_corrections.append((i, -1))

            # Apply all corrections for this position and measure combined effect
            if len(best_corrections) > 0:
                W_test = W_current.clone()
                for i, direction in best_corrections:
                    delta = direction * self.correction_magnitude * W_orig_f32[i, j]
                    W_test[i, j] += delta

                error_new = compute_error(W_test)

                if error_new < error_current:
                    # Accept corrections
                    improvement = error_current - error_new
                    W_current = W_test
                    error_current = error_new
                    num_applied += 1

                    if debug and num_applied <= 3:
                        print(f"      Position {idx+1}/{len(candidate_positions)}: j={j_item}, improvement={improvement:.6e}, error={error_current:.6e}")
                else:
                    # No improvement, stop
                    stopped_early = True
                    if debug:
                        print(f"      Stopped at position {idx+1}/{len(candidate_positions)}: no improvement")
                    break
            else:
                # No corrections found for this position, continue
                continue

        # Convert back to original dtype
        W_corrected = W_current.to(weight_dtype)

        # Final error
        error_final = error_current
        error_reduction = error_initial - error_final

        correction_stats = {
            'error_before_mean': error_initial,
            'error_after_mean': error_final,
            'error_reduction_mean': error_reduction,
            'num_candidates': len(candidate_positions),
            'num_applied': num_applied,
            'stopped_early': stopped_early,
            'apply_percentage': num_applied / len(candidate_positions) * 100 if len(candidate_positions) > 0 else 0,
        }

        if debug:
            print(f"    Error before: {error_initial:.6e}")
            print(f"    Error after: {error_final:.6e}")
            print(f"    Reduction: {error_reduction:.6e} ({error_reduction/error_initial*100 if error_initial > 0 else 0:.2f}%)")
            print(f"    Applied: {num_applied}/{len(candidate_positions)} ({correction_stats['apply_percentage']:.1f}%)")
            if stopped_early:
                print(f"    Status: Early stopping (corrections stopped helping)")
            else:
                print(f"    Status: Applied all candidate corrections")

        # Cleanup
        torch.cuda.empty_cache()

        return W_corrected, correction_stats

    @torch.no_grad()
    def prune_weights_wanda(self, W, salience, sparsity):
        """
        Vectorized Wanda pruning.

        Returns:
            W_pruned: Pruned weights
            prune_mask: Boolean mask (True = kept, False = pruned)
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

        return W_pruned, mask, actual_sparsity

    @torch.no_grad()
    def prune_layer(self, name, module, debug=False):
        """
        Prune layer with optional greedy direct error correction.
        """
        if name not in self.activation_stats:
            print(f"  ⚠️  No activation stats for {name}, skipping")
            return

        # Get activation statistics
        l2_salience, js_mean, _, activations = self.get_activation_stats(name)
        if l2_salience is None:
            if debug:
                print(f"  DEBUG: No activation stats for {name}, skipping")
            return

        if debug:
            print(f"  DEBUG: Layer {name}")
            print(f"    L2 salience shape: {l2_salience.shape}")
            print(f"    JS mean range: [{js_mean.min():.6f}, {js_mean.max():.6f}]")
            print(f"    Activations shape: {activations.shape}")

        W = module.weight.data
        original_dtype = W.dtype

        # Move to device
        W_device = W.to(self.device)
        l2_salience_device = l2_salience.to(self.device).to(W.dtype)
        js_mean_device = js_mean.to(self.device).to(W.dtype)

        # Step 1: Wanda pruning
        W_pruned, prune_mask, actual_sparsity = self.prune_weights_wanda(
            W_device, l2_salience_device, self.sparsity
        )

        # Step 2: Greedy direct error correction (if enabled)
        if self.use_correction:
            # Select MODERATE candidate positions (using Kneedle)
            candidate_positions, knee_info = self.select_moderate_positions(
                js_mean_device, prune_mask, debug=debug
            )

            # Correct weights using greedy iterative approach with direct error measurement
            if knee_info is not None and len(candidate_positions) > 0:
                W_final, correction_stats = self.correct_weights_greedy_direct(
                    W_device, W_pruned, js_mean_device, activations,
                    candidate_positions, prune_mask, debug=debug
                )
            else:
                W_final = W_pruned
                correction_stats = None
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
        """
        Create hook for storing activation samples.

        Instead of Hessian (O(d²)), stores activation samples (O(samples × d)).
        Much faster and more memory efficient!
        """
        def hook(module, input, output):
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

            # Flatten to [num_tokens, hidden_dim]
            inp_flat = inp.reshape(-1, inp.shape[-1])
            num_tokens = inp_flat.shape[0]
            hidden_dim = inp_flat.shape[1]

            # Convert to float32 for numerical stability
            inp_flat = inp_flat.detach().float()

            # Initialize statistics storage if needed
            if name not in self.activation_stats:
                self.activation_stats[name] = {
                    'activations': [],  # Store samples as list (will concatenate later)
                    'mean_sum': torch.zeros(hidden_dim, dtype=torch.float32, device=inp.device),
                    'count': 0
                }

            # Store activation samples
            stats = self.activation_stats[name]

            # Store on GPU during calibration for speed
            stats['activations'].append(inp_flat)

            # Also accumulate mean for JS estimator
            stats['mean_sum'] += inp_flat.sum(dim=0)
            stats['count'] += num_tokens

        return hook

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """
        Calibrate batch of layers by storing activation samples.

        Note: This requires O(n_samples × hidden_dim) memory per layer batch.
        Much faster than Hessian computation!
        """
        # Clear previous statistics for this batch
        for name, _ in layer_names_batch:
            if name in self.activation_stats:
                del self.activation_stats[name]

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

        # Verify all layers got some data
        for name, _ in layer_names_batch:
            if name not in self.activation_stats:
                print(f"  ⚠️ No activations captured for {name}")
                self.activation_stats[name] = {
                    'hessian': None,
                    'mean_sum': None,
                    'count': 0
                }

    def prune_model_sequential(self, calibration_data, n_samples=500, layer_batch_size=4):
        """
        Batched sequential pruning with greedy direct error correction.

        Note: Stores activation samples O(n_samples × hidden_dim) per layer.
        Much faster than Hessian-based methods!
        """
        print("\n" + "=" * 80)
        print("BATCHED SEQUENTIAL PRUNING WITH GREEDY DIRECT ERROR CORRECTION")
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

            # Cleanup - clear statistics for this batch
            for name, _ in batch_layers:
                if name in self.activation_stats:
                    del self.activation_stats[name]

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
                    num_applied = [self.layer_stats[k]['correction_stats']['num_applied']
                                  for k in corrected_layers]
                    apply_pcts = [self.layer_stats[k]['correction_stats']['apply_percentage']
                                 for k in corrected_layers]
                    stopped_early_count = sum(1 for k in corrected_layers
                                            if self.layer_stats[k]['correction_stats'].get('stopped_early', False))

                    print(f"\nGreedy direct error correction statistics ({len(corrected_layers)} layers):")
                    print(f"  Mean error before: {np.mean(errors_before):.6e}")
                    print(f"  Mean error after: {np.mean(errors_after):.6e}")
                    print(f"  Mean reduction: {np.mean(reductions):.6e} ({np.mean(reductions)/np.mean(errors_before)*100:.2f}%)")
                    print(f"  Mean corrections applied: {np.mean(num_applied):.1f} ({np.mean(apply_pcts):.1f}% of candidates)")
                    print(f"  Early stopping: {stopped_early_count}/{len(corrected_layers)} layers")

        # Final cleanup
        self.activation_stats = {}
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
        description="Wanda Pruning with Greedy Direct Error Correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--sparsity", type=float, default=0.5,
                       help="Target sparsity (0.5 = prune 50%%)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens per sample")
    parser.add_argument("--use-correction", action="store_true", default=True,
                       help="Enable greedy direct error correction")
    parser.add_argument("--no-correction", dest="use_correction", action="store_false",
                       help="Disable correction")
    parser.add_argument("--knee-tolerance", type=float, default=0.0,
                       help="Tolerance for Kneedle algorithm")
    parser.add_argument("--offset-percent", type=float, default=0.0,
                       help="Offset from knee point (as fraction)")
    parser.add_argument("--percent-change", type=float, default=0.05,
                       help="Percentage of weights to correct (default 5%%)")
    parser.add_argument("--correction-magnitude", type=float, default=0.01,
                       help="Fixed correction magnitude as fraction of original weight (default 1%%)")
    parser.add_argument("--output-dir", type=str, default="./pruned_models/model_prac_h",
                       help="Output directory")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3",
                       help="Model path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--layer-batch-size", type=int, default=4,
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
    print("Wanda Pruning with Greedy Direct Error Correction")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Target Sparsity: {args.sparsity*100:.1f}%")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"Correction magnitude: {args.correction_magnitude*100:.1f}%")
    print(f"Greedy direct error correction: {args.use_correction}")
    if args.use_correction:
        print(f"  Max candidates: {args.percent_change*100:.1f}% of surviving weights")
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
    pruner = WandaPrunerWithGreedyHessian(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sparsity=args.sparsity,
        max_tokens_per_sample=args.max_tokens_per_sample,
        use_correction=args.use_correction,
        knee_tolerance=args.knee_tolerance,
        offset_percent=args.offset_percent,
        percent_change=args.percent_change,
        correction_magnitude=args.correction_magnitude
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
