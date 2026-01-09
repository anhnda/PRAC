"""
Wanda Pruning with Weakest Weight Correction - XM Version

Combines Wanda pruning with weakest surviving weight correction:
1. Apply Wanda pruning (Score_ij = |W_ij| * ||X_j||_2)
2. Select WEAKEST surviving weights (smallest Wanda scores) based on fixed percentage
3. Correct weights greedily at weakest positions to minimize dot product error
4. Early stopping when error approaches 0 or no improvement

Key Features:
- L2 norm salience for Wanda scoring
- James-Stein mean estimator for weight correction
- Weakest-first selection: targets surviving weights with smallest L2 salience
- Simple fixed-percentage selection (NO Kneedle algorithm)
- Greedy iterative correction (like prac_ho.py)
- Fixed magnitude changes: correction_magnitude * |W_orig[i,j]|
- Objective: Minimize dot product error with JS mean
- Early stopping: stops when no improvement or error close to 0
- Batched sequential processing for memory efficiency

Differences from prac_xl.py:
- NO Kneedle algorithm for position selection
- Selects WEAKEST surviving weights (smallest Wanda scores) like prac_ho.py
- Greedy iterative correction (not batch vectorized)
- Fixed magnitude with early stopping

Differences from prac_ho.py:
- Objective function: dot product with JS mean (not direct activation error)
- Same greedy iterative approach
- Same fixed magnitude correction style
- Same early stopping mechanism

Usage:
    python prac_xm.py --sparsity 0.5 --percent-change 0.05 --correction-magnitude 0.01
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
    Wanda Pruning with Weakest Weight Correction (XM Version).
    Selects and corrects weakest surviving weights (smallest Wanda scores).
    Uses greedy iterative correction with fixed magnitude and early stopping.
    """

    def __init__(self, model, tokenizer, device="cuda", sparsity=0.5,
                 max_tokens_per_sample=512, use_correction=True,
                 percent_change=0.05, correction_magnitude=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sparsity = sparsity
        self.max_tokens_per_sample = max_tokens_per_sample
        self.use_correction = use_correction
        self.percent_change = percent_change
        self.correction_magnitude = correction_magnitude

        # Storage - use incremental statistics instead of storing all activations
        # Memory: O(hidden_dim) instead of O(samples × hidden_dim)
        self.activation_stats = {}  # Dict of {name: {'l2_sum': tensor, 'mean_sum': tensor, 'count': int}}
        self.hooks = []
        self.layer_stats = {}

        print(f"\n[Wanda Pruner with Weakest Weight Correction - XM Version]")
        print(f"  Target sparsity: {sparsity*100:.1f}% (keep {(1-sparsity)*100:.1f}%)")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample")
        print(f"  Salience metric: ||X_j||_2 (L2 norm)")
        print(f"  Scoring: Score_ij = |W_ij| * ||X_j||_2")
        print(f"  Memory optimization: Incremental statistics (no activation storage)")
        if use_correction:
            print(f"  Weakest weight correction: ENABLED")
            print(f"    - Selection: WEAKEST {percent_change*100:.1f}% of surviving weights (smallest Wanda scores)")
            print(f"    - Method: Greedy iterative (like prac_ho.py)")
            print(f"    - Magnitude: {correction_magnitude*100:.1f}% of original weight")
            print(f"    - Objective: Minimize dot product error with JS mean")
            print(f"    - Early stopping: when no improvement or error close to 0")
        else:
            print(f"  Weakest weight correction: DISABLED")
        print(f"  Special: lm_head is NOT pruned")

    @torch.no_grad()
    def get_activation_stats(self, name):
        """
        Compute L2 salience and James-Stein mean from pre-accumulated statistics.

        Memory efficient: Uses O(hidden_dim) instead of O(samples × hidden_dim).

        Returns:
            tuple: (l2_salience, js_mean, raw_mean)
        """
        if name not in self.activation_stats:
            return None, None, None

        stats = self.activation_stats[name]

        # Check if we have any data
        if stats['count'] == 0:
            return None, None, None

        # Compute statistics from accumulated sums
        l2_salience = torch.sqrt(stats['l2_sum'] / stats['count'])
        raw_mean = stats['mean_sum'] / stats['count']

        # Apply James-Stein estimator
        js_mean = compute_james_stein_mean(raw_mean)

        return l2_salience, js_mean, raw_mean

    @torch.no_grad()
    def select_weakest_positions(self, l2_salience, prune_mask, debug=False):
        """
        Select WEAKEST surviving weights (smallest Wanda scores) for correction.

        Simple fixed-percentage selection (NO Kneedle algorithm).

        Algorithm:
        1. Filter to only surviving weights (where prune_mask == True)
        2. Among survivors, sort by L2 salience (Wanda score) in ASCENDING order
        3. Select percent_change * num_surviving positions (the weakest ones)

        Args:
            l2_salience: L2 salience (Wanda score metric) [in_features]
            prune_mask: Boolean mask [out_features, in_features] (True = kept, False = pruned)
            debug: Print debug info

        Returns:
            selected_indices: Tensor of selected positions (column indices)
            selection_info: Dict with selection statistics
        """
        # Count surviving weights per input channel (across all output channels)
        # A channel is "surviving" if at least one output channel kept it
        survival_count = prune_mask.sum(dim=0)  # [in_features]
        is_surviving = survival_count > 0

        # Filter to surviving channels only
        surviving_indices = torch.where(is_surviving)[0]

        if len(surviving_indices) == 0:
            print(f"    ⚠️ No surviving weights to correct!")
            return torch.tensor([], dtype=torch.long), None

        # Get L2 salience for surviving channels (this is the Wanda score metric)
        l2_surviving = l2_salience[surviving_indices]

        # Sort ASCENDING by salience (weakest first)
        sorted_values, sorted_order = torch.sort(l2_surviving, descending=False)
        sorted_indices = surviving_indices[sorted_order]

        num_surviving = len(surviving_indices)

        # Simple fixed-percentage selection
        num_to_select = max(1, int(self.percent_change * num_surviving))

        # Select WEAKEST weights (first num_to_select from sorted list)
        selected_indices = sorted_indices[:num_to_select]
        actual_num = len(selected_indices)

        selection_info = {
            'num_selected': actual_num,
            'num_surviving': num_surviving,
            'selection_pct': actual_num / num_surviving * 100 if num_surviving > 0 else 0,
            'min_salience': sorted_values[0].item() if len(sorted_values) > 0 else 0,
            'max_salience': sorted_values[actual_num-1].item() if actual_num > 0 and len(sorted_values) >= actual_num else 0,
        }

        if debug:
            print(f"    Surviving channels: {num_surviving}")
            print(f"    WEAKEST selection: {actual_num} positions ({selection_info['selection_pct']:.2f}%)")
            if len(selected_indices) > 0:
                print(f"    Salience range: [{selection_info['min_salience']:.6f}, {selection_info['max_salience']:.6f}]")

        return selected_indices, selection_info

    @torch.no_grad()
    def correct_weights_greedy(self, W_orig, W_pruned, js_mean,
                               selected_positions, prune_mask, debug=False):
        """
        Greedy iterative weight correction (prac_ho.py style).

        Key: Only corrects SURVIVING weakest weights (not pruned ones).
        Fixed magnitude correction with early stopping.

        Algorithm (greedy, position by position):
        1. Compute initial error: error[i] = (W_pruned[i,:] - W_orig[i,:]) · js_mean
        2. For each weakest position j (in order):
           a. For all channels: delta[i] = correction_magnitude * W_orig[i,j]
           b. Try +delta and -delta, compute error changes (vectorized)
           c. Apply the direction that reduces error for each channel
           d. Update error vector
           e. If improvement > 0: continue
           f. If no improvement or error close to 0: stop early

        Args:
            W_orig: Original weights [out_features, in_features]
            W_pruned: Pruned weights [out_features, in_features]
            js_mean: James-Stein mean [in_features]
            selected_positions: Indices of weakest positions (column indices)
            prune_mask: Boolean mask [out_features, in_features] (True = kept)
            debug: Print debug info

        Returns:
            W_corrected: Corrected weights
            correction_stats: Dictionary with statistics
        """
        # Store original dtype and convert everything to float32 for computation
        weight_dtype = W_orig.dtype

        # Convert to float32 for computation
        W_orig_f32 = W_orig.float()
        W_pruned_f32 = W_pruned.float()
        js_mean_f32 = js_mean.float()

        # Compute initial reconstruction errors for all channels
        # error[i] = (W_pruned[i,:] - W_orig[i,:]) · js_mean
        W_current = W_pruned_f32.clone()
        errors = torch.matmul(W_current - W_orig_f32, js_mean_f32)  # [out_features]
        error_current = errors.abs().sum().item()
        error_initial = error_current

        if len(selected_positions) == 0:
            if debug:
                print(f"    ⚠️ No candidates, no correction")
            return W_pruned.clone(), None

        # Greedy correction: process each candidate position
        num_applied = 0
        stopped_early = False
        position_improvements = []

        for idx, j in enumerate(selected_positions):
            j_item = j.item()

            # Only correct surviving weights at this position
            survivors_at_j = prune_mask[:, j]  # [out_features]
            if not survivors_at_j.any():
                continue

            # VECTORIZED: evaluate all channels at position j simultaneously
            # For each channel i, changing W[i,j] by ±delta changes error[i]

            # Compute delta values for all channels: [out_features]
            deltas = self.correction_magnitude * W_orig_f32[:, j]  # [out_features]

            # For W[i,j] → W[i,j] + delta[i]:
            # New error[i] = error[i] + delta[i] * js_mean[j]
            js_mean_j = js_mean_f32[j]

            # Error changes (vectorized across all channels)
            delta_error_pos = deltas * js_mean_j  # Change in error if we add delta
            delta_error_neg = -deltas * js_mean_j  # Change in error if we subtract delta

            # We want to reduce |error|, so:
            # If error > 0, we want negative delta_error (reduce error)
            # If error < 0, we want positive delta_error (reduce |error|)

            # Compute new absolute errors for both directions
            new_error_pos = (errors + delta_error_pos).abs()
            new_error_neg = (errors + delta_error_neg).abs()
            current_abs_error = errors.abs()

            # Determine which channels benefit from each direction
            # Only consider surviving weights at this position
            pos_improves = (new_error_pos < current_abs_error) & (new_error_pos <= new_error_neg) & survivors_at_j
            neg_improves = (new_error_neg < current_abs_error) & ~pos_improves & survivors_at_j

            # Compute total improvement
            improvement_pos = (current_abs_error[pos_improves] - new_error_pos[pos_improves]).sum().item() if pos_improves.any() else 0.0
            improvement_neg = (current_abs_error[neg_improves] - new_error_neg[neg_improves]).sum().item() if neg_improves.any() else 0.0
            total_improvement = improvement_pos + improvement_neg

            # Check if we should stop (error close to 0 or no improvement)
            if total_improvement <= 0:
                stopped_early = True
                if debug:
                    print(f"      Stopped at position {idx+1}/{len(selected_positions)}: no improvement")
                break

            # Check if error is already close to 0
            if error_current < 1e-6:
                stopped_early = True
                if debug:
                    print(f"      Stopped at position {idx+1}/{len(selected_positions)}: error close to 0")
                break

            # Apply corrections vectorized
            if pos_improves.any():
                # Apply positive corrections
                W_current[pos_improves, j] += deltas[pos_improves]
                errors[pos_improves] += delta_error_pos[pos_improves]

            if neg_improves.any():
                # Apply negative corrections
                W_current[neg_improves, j] -= deltas[neg_improves]
                errors[neg_improves] += delta_error_neg[neg_improves]

            error_current -= total_improvement
            num_applied += 1
            num_channels_corrected = pos_improves.sum().item() + neg_improves.sum().item()

            # Track for diagnostics
            position_improvements.append({
                'position': idx,
                'j': j_item,
                'improvement': total_improvement,
                'num_channels': num_channels_corrected,
                'error_after': error_current,
            })

            if debug and num_applied <= 5:
                print(f"      Position {idx+1}/{len(selected_positions)}: j={j_item}, "
                      f"channels={num_channels_corrected}, improvement={total_improvement:.6e}, "
                      f"error={error_current:.6e}")

        # Convert back to original dtype
        W_corrected = W_current.to(weight_dtype)

        # Final error
        error_final = error_current
        error_reduction = error_initial - error_final

        # Simple statistics
        avg_improvement = sum(p['improvement'] for p in position_improvements) / len(position_improvements) if position_improvements else 0
        avg_channels = sum(p['num_channels'] for p in position_improvements) / len(position_improvements) if position_improvements else 0

        correction_stats = {
            'error_before_mean': error_initial,
            'error_after_mean': error_final,
            'error_reduction_mean': error_reduction,
            'num_candidates': len(selected_positions),
            'num_applied': num_applied,
            'stopped_early': stopped_early,
            'apply_percentage': num_applied / len(selected_positions) * 100 if len(selected_positions) > 0 else 0,
            'avg_improvement_per_position': avg_improvement,
            'avg_channels_per_position': avg_channels,
        }

        if debug:
            print(f"    Error before: {error_initial:.6e}")
            print(f"    Error after: {error_final:.6e}")
            print(f"    Reduction: {error_reduction:.6e} ({error_reduction/error_initial*100 if error_initial > 0 else 0:.2f}%)")
            print(f"    Positions applied: {num_applied}/{len(selected_positions)}")
            print(f"    Avg improvement/position: {avg_improvement:.6e}")
            print(f"    Avg channels corrected/position: {avg_channels:.1f}")
            if stopped_early:
                print(f"    Status: Early stopping")
            else:
                print(f"    Status: Completed all candidate positions")

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
        Prune layer with optional greedy weakest weight correction.
        Selects and greedily corrects the weakest surviving weights (smallest Wanda scores).
        Uses fixed magnitude changes with early stopping.
        """
        if name not in self.activation_stats:
            print(f"  ⚠️  No activation stats for {name}, skipping")
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
        W_pruned, prune_mask, actual_sparsity = self.prune_weights_wanda(
            W_device, l2_salience_device, self.sparsity
        )

        # Step 2: Weakest weight correction (if enabled)
        if self.use_correction:
            # Select WEAKEST surviving positions (smallest Wanda scores)
            selected_positions, selection_info = self.select_weakest_positions(
                l2_salience_device, prune_mask, debug=debug
            )

            # Correct weights greedily (only surviving ones)
            if selection_info is not None and len(selected_positions) > 0:
                W_final, correction_stats = self.correct_weights_greedy(
                    W_device, W_pruned, js_mean_device, selected_positions,
                    prune_mask, debug=debug
                )
            else:
                W_final = W_pruned
                correction_stats = None
        else:
            W_final = W_pruned
            selection_info = None
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
                'selection_info': selection_info,
                'correction_stats': correction_stats,
            })

        if debug:
            print(f"  → Sparsity: {actual_sparsity*100:.2f}%")

        del W_device, l2_salience_device, js_mean_device, W_pruned, W_final
        torch.cuda.empty_cache()

    def get_hook(self, name):
        """
        Create hook for incremental activation statistics.

        Instead of storing all activations (O(samples × hidden_dim) memory),
        we accumulate running statistics (O(hidden_dim) memory).
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
                    'l2_sum': torch.zeros(hidden_dim, dtype=torch.float32),
                    'mean_sum': torch.zeros(hidden_dim, dtype=torch.float32),
                    'count': 0
                }

            # Update running statistics on CPU (to avoid GPU memory pressure)
            stats = self.activation_stats[name]
            stats['l2_sum'] += inp_flat.pow(2).sum(dim=0).cpu()
            stats['mean_sum'] += inp_flat.sum(dim=0).cpu()
            stats['count'] += num_tokens

            # No need to store inp - we've extracted what we need!
            del inp_flat

        return hook

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """
        Calibrate batch of layers using incremental statistics.

        Memory efficient: Accumulates statistics instead of storing activations.
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
                # Initialize with zeros if no data
                print(f"  ⚠️ No activations captured for {name}")
                self.activation_stats[name] = {
                    'l2_sum': None,
                    'mean_sum': None,
                    'count': 0
                }

    def prune_model_sequential(self, calibration_data, n_samples=500, layer_batch_size=16):
        """Batched sequential pruning with greedy weakest weight correction."""
        print("\n" + "=" * 80)
        print("BATCHED SEQUENTIAL PRUNING WITH GREEDY WEAKEST WEIGHT CORRECTION (XM Version)")
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

                    print(f"\nGreedy weakest weight correction statistics ({len(corrected_layers)} layers):")
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
        description="Wanda Pruning with Greedy Weakest Weight Correction (XM Version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--sparsity", type=float, default=0.5,
                       help="Target sparsity (0.5 = prune 50%%)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens per sample")
    parser.add_argument("--use-correction", action="store_true", default=True,
                       help="Enable weakest weight correction")
    parser.add_argument("--no-correction", dest="use_correction", action="store_false",
                       help="Disable weakest weight correction")
    parser.add_argument("--percent-change", type=float, default=0.05,
                       help="Percentage of weakest surviving weights to correct (default 5%%)")
    parser.add_argument("--correction-magnitude", type=float, default=0.01,
                       help="Fixed correction magnitude as fraction of original weight (default 1%%)")
    parser.add_argument("--output-dir", type=str, default="./pruned_models/model_prac_xm",
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
    print("Wanda Pruning with Greedy Weakest Weight Correction (XM Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Target Sparsity: {args.sparsity*100:.1f}%")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"Greedy weakest weight correction: {args.use_correction}")
    if args.use_correction:
        print(f"  Correction percent: {args.percent_change*100:.1f}% (weakest surviving weights)")
        print(f"  Correction magnitude: {args.correction_magnitude*100:.1f}%")
        print(f"  Method: Greedy iterative with early stopping")
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
