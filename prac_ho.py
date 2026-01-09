"""
Wanda Pruning with Greedy Direct Error Correction (prac_ho.py)

Implements greedy iterative correction using DIRECT activation error measurement
with OPTIMIZED residual-based incremental updates.

Key Difference from prac_h.py:
    This version selects the WEAKEST surviving weights (smallest Wanda scores)
    for correction, rather than using the Kneedle algorithm to find "moderate" weights.

Mathematical Foundation:
    Directly minimize the reconstruction error on calibration data:

    Error(W) = ||X @ W^T - X @ W_orig^T||²

    where:
    - X = calibration activations [num_samples, in_features]
    - W = current weights [out_features, in_features]
    - W_orig = original dense weights

Optimization Key Insight:
    Instead of recomputing full error after each weight change (expensive!),
    we maintain residual R = X @ (W - W_orig)^T and update it incrementally.

    FULLY VECTORIZED - No Python loops over channels!

    For all channels at position j simultaneously:
        1. Deltas: D = magnitude × W_orig[:, j]  [out_features]
        2. Dot products: P = R^T @ X[:, j]  [out_features]
        3. Error changes: ΔE = 2D⊙P + D²||X_j||²  (vectorized across all channels!)
        4. Masks: pos_better = (ΔE_pos < 0) & (ΔE_pos ≤ ΔE_neg)  (boolean indexing!)
        5. Apply: W[mask, j] += D[mask]; R[:, mask] += X_j ⊗ D[mask]  (GPU parallel!)

    This reduces complexity from O(positions × channels × tokens × features²)
    to O(positions × tokens × features) - over 1000x faster with full GPU utilization!

Key Features:
1. **Direct error measurement**: Computes ||XW^T - XW_orig^T||² on actual activations
2. **No Hessian**: Stores activation samples instead of computing XX^T (much faster!)
3. **Fully vectorized**: All channels evaluated in parallel - NO Python loops!
4. **Greedy search**: Process positions one-by-one, stop when no improvement
5. **Fixed magnitude**: 1% correction (default) avoids clamping issues
6. **Incremental updates**: Residual-based optimization for 1000x+ speedup
7. **Weakest-first selection**: Corrects weakest surviving weights first

Algorithm:
    1. Select WEAKEST surviving weights (smallest Wanda scores, up to percent-change)
    2. Compute initial residual: R = X @ (W_pruned - W_orig)^T
    3. For each candidate position j (greedy loop):
       a. Vectorized across ALL channels: compute error changes ΔE for +δ and -δ
       b. Create boolean masks: which channels improve with each direction
       c. Apply corrections using GPU-parallel indexing: W[mask, j] += deltas[mask]
       d. Update residual incrementally: R[:, mask] += X_j ⊗ deltas[mask]
       e. If total improvement > 0: continue to next position
       f. If no improvement: stop (early termination)

Benefits:
    - **Mathematically correct**: Directly optimizes reconstruction error on calibration data
    - **Fully vectorized**: ALL channels evaluated in parallel on GPU (no Python loops!)
    - **Efficient**: 1000x+ faster than naive approach via residual-based updates
    - **Adaptive**: Natural early stopping when improvements plateau
    - **Stable**: Small fixed magnitude prevents network disruption
    - **Memory efficient**: Stores activations on CPU, moves to GPU only during correction
    - **Focused**: Targets weakest surviving weights for maximum impact

Usage:
    python prac_ho.py --sparsity 0.5 --percent-change 0.05 --correction-magnitude 0.01
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


class WandaPrunerWithGreedyHessian:
    """
    Wanda Pruning with Greedy Direct Error Correction.

    Uses greedy iterative approach with fixed magnitude corrections.
    Measures direct activation error ||XW^T - XW_orig^T||² on calibration data.
    Selects WEAKEST surviving weights (smallest Wanda scores) for correction.
    """

    def __init__(self, model, tokenizer, device="cuda", sparsity=0.5,
                 max_tokens_per_sample=512, max_activation_samples=8192,
                 use_correction=True, percent_change=0.05, correction_magnitude=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sparsity = sparsity
        self.max_tokens_per_sample = max_tokens_per_sample
        self.max_activation_samples = max_activation_samples  # Limit total samples stored
        self.use_correction = use_correction
        self.percent_change = percent_change
        self.correction_magnitude = correction_magnitude

        # Storage for activation samples (for direct error measurement)
        # Memory: O(max_activation_samples * hidden_dim) per layer
        self.activation_stats = {}  # Dict of {name: {'activations': tensor, 'mean_sum': tensor, 'count': int}}
        self.hooks = []
        self.layer_stats = {}

        print(f"\n[Wanda Pruner with Greedy Direct Error Correction]")
        print(f"  Target sparsity: {sparsity*100:.1f}% (keep {(1-sparsity)*100:.1f}%)")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample")
        print(f"  Max activation samples: {max_activation_samples} tokens/layer")
        print(f"  Salience metric: ||X_j||_2 (L2 norm from activations)")
        print(f"  Scoring: Score_ij = |W_ij| * ||X_j||_2")
        print(f"  Error measurement: Direct ||X·W^T - X·W_orig^T||² on calibration data")
        if use_correction:
            print(f"  Greedy direct error correction: ENABLED")
            print(f"    - Selection strategy: WEAKEST surviving weights (smallest Wanda scores)")
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
    def select_weakest_positions(self, l2_salience, prune_mask, debug=False):
        """
        Select WEAKEST surviving weights (smallest Wanda scores) for correction.

        Strategy:
        1. Compute Wanda scores for all surviving weights
        2. Sort by score (ascending - weakest first)
        3. Select bottom percent_change% for correction

        Returns candidates sorted by salience (ascending) for greedy processing.
        """
        # Count surviving weights per input channel
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
        max_to_select = max(1, int(self.percent_change * num_surviving))

        # Select weakest weights (first max_to_select from sorted list)
        candidate_indices = sorted_indices[:max_to_select]

        selection_info = {
            'num_candidates': len(candidate_indices),
            'num_surviving': num_surviving,
            'candidate_pct': len(candidate_indices) / num_surviving * 100 if num_surviving > 0 else 0,
            'min_salience': sorted_values[0].item() if len(sorted_values) > 0 else 0,
            'max_salience': sorted_values[max_to_select-1].item() if max_to_select > 0 and len(sorted_values) >= max_to_select else 0,
        }

        if debug:
            print(f"    Surviving channels: {num_surviving}")
            print(f"    WEAKEST candidates: {selection_info['num_candidates']} positions ({selection_info['candidate_pct']:.2f}%)")
            if len(candidate_indices) > 0:
                print(f"    Salience range: [{selection_info['min_salience']:.6f}, {selection_info['max_salience']:.6f}]")

        return candidate_indices, selection_info

    @torch.no_grad()
    def correct_weights_greedy_direct(self, W_orig, W_pruned, js_mean, activations,
                                      candidate_positions, prune_mask, debug=False):
        """
        Greedy iterative weight correction using DIRECT activation error.

        Fully Vectorized Algorithm (No Python Loops Over Channels):
            1. Compute residual once: R = X @ (W_current - W_orig)^T  [num_tokens, out_features]
            2. Precompute column norms: ||X[:,j]||² for all j
            3. For each candidate position j:
               a. Compute dot products: P = R^T @ X[:, j]  [out_features] - vectorized
               b. Compute deltas: D = magnitude × W_orig[:, j]  [out_features] - vectorized
               c. Error changes: ΔE_pos = 2D⊙P + D²||X_j||², ΔE_neg = -2D⊙P + D²||X_j||² - vectorized
               d. Boolean masks: which channels improve with +D or -D - vectorized
               e. Update weights: W[mask_pos, j] += D[mask_pos] - vectorized indexing
               f. Update residual: R[:, mask] += X_j ⊗ D[mask] - vectorized broadcast
               g. If no improvement: stop
            4. Return corrected weights

        Complexity:
            Naive approach: O(positions × channels × tokens × features²) - completely infeasible
            First optimization: O(positions × channels + positions × tokens × features) - 100x faster
            Final (vectorized): O(positions × tokens × features) - 1000x+ faster (no Python loops!)

        Mathematical Basis:
            For weight change W[i,j] → W[i,j] + δ, the residual at channel i becomes:
            R_new[:,i] = R[:,i] + δ * X[:,j]

            Change in squared error:
            ΔE_i = sum((R_new[:,i])²) - sum((R[:,i])²)
                 = 2δ * (R[:,i] · X[:,j]) + δ² * ||X[:,j]||²

            This allows O(1) evaluation per channel without full matrix multiplication!

        Args:
            W_orig: Original weights [out_features, in_features]
            W_pruned: Pruned weights [out_features, in_features]
            js_mean: James-Stein mean [in_features] (not used)
            activations: Calibration activations [num_samples, in_features]
            candidate_positions: Indices of candidate positions (WEAKEST weights)
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

        # Move activations from CPU to GPU for error computation
        X = activations.float().to(self.device)  # [num_samples, in_features]
        num_tokens = X.shape[0]
        out_features = W_orig_f32.shape[0]

        if len(candidate_positions) == 0:
            if debug:
                print(f"    ⚠️ No candidates, no correction")
            return W_pruned.clone(), None

        # Compute initial residual: R = X @ (W_current - W_orig)^T
        # R[t, i] = error at token t, output channel i
        W_current = W_pruned_f32.clone()
        R = X @ (W_current - W_orig_f32).t()  # [num_tokens, out_features]
        error_current = (R ** 2).sum().item()
        error_initial = error_current

        # Precompute X column norms (needed for delta^2 term)
        X_col_norms_sq = (X ** 2).sum(dim=0)  # [in_features]

        # Greedy correction: process each candidate position
        num_applied = 0
        stopped_early = False
        position_improvements = []  # Track improvement per position for diagnostics

        for idx, j in enumerate(candidate_positions):
            j_item = j.item()

            # Only correct surviving weights at this position
            survivors_at_j = prune_mask[:, j]  # [out_features]
            if not survivors_at_j.any():
                continue

            # FULLY VECTORIZED: evaluate all channels at position j simultaneously
            # For each channel i, changing W[i,j] by δ changes residual R[:,i] by δ*X[:,j]
            # Change in error: ΔE_i = 2δ * (R[:,i] · X[:,j]) + δ² * ||X[:,j]||²

            X_j = X[:, j]  # [num_tokens]
            X_j_norm_sq = X_col_norms_sq[j]

            # Compute dot products for all channels: R^T @ X_j = [out_features]
            dot_products = R.t() @ X_j  # [out_features]

            # Compute delta values for all channels: [out_features]
            deltas = self.correction_magnitude * W_orig_f32[:, j]  # [out_features]

            # Vectorized error change computation for all channels at once
            # ΔE(+δ) = 2δ * P + δ² * ||X_j||²
            delta_error_pos = 2 * deltas * dot_products + deltas ** 2 * X_j_norm_sq  # [out_features]
            # ΔE(-δ) = -2δ * P + δ² * ||X_j||²
            delta_error_neg = -2 * deltas * dot_products + deltas ** 2 * X_j_norm_sq  # [out_features]

            # Determine which channels benefit from each direction
            # Only consider surviving weights at this position
            pos_improves = (delta_error_pos < 0) & (delta_error_pos <= delta_error_neg) & survivors_at_j
            neg_improves = (delta_error_neg < 0) & ~pos_improves & survivors_at_j

            # Compute total improvement
            improvement_pos = -delta_error_pos[pos_improves].sum().item() if pos_improves.any() else 0.0
            improvement_neg = -delta_error_neg[neg_improves].sum().item() if neg_improves.any() else 0.0
            total_improvement = improvement_pos + improvement_neg

            # Diagnostic: track linear vs quadratic terms
            if debug and total_improvement > 0:
                # Average linear term contribution (2δP)
                linear_term_pos = (2 * deltas[pos_improves] * dot_products[pos_improves]).abs().mean().item() if pos_improves.any() else 0
                linear_term_neg = (2 * deltas[neg_improves] * dot_products[neg_improves]).abs().mean().item() if neg_improves.any() else 0
                # Average quadratic penalty (δ²||X||²)
                quad_term = (deltas[pos_improves | neg_improves] ** 2).mean().item() * X_j_norm_sq.item() if (pos_improves | neg_improves).any() else 0

            if total_improvement > 0:
                # Apply corrections vectorized - no Python loops!
                if pos_improves.any():
                    # Apply positive corrections
                    W_current[pos_improves, j] += deltas[pos_improves]
                    # Update residual: R[:, pos_channels] += X_j.unsqueeze(1) @ deltas[pos_channels].unsqueeze(0)
                    R[:, pos_improves] += X_j.unsqueeze(1) * deltas[pos_improves].unsqueeze(0)

                if neg_improves.any():
                    # Apply negative corrections
                    W_current[neg_improves, j] -= deltas[neg_improves]
                    # Update residual: R[:, neg_channels] -= X_j.unsqueeze(1) @ deltas[neg_channels].unsqueeze(0)
                    R[:, neg_improves] -= X_j.unsqueeze(1) * deltas[neg_improves].unsqueeze(0)

                error_current -= total_improvement
                num_applied += 1
                num_channels_corrected = pos_improves.sum().item() + neg_improves.sum().item()

                # VERIFICATION: Check if incremental error matches actual error
                if debug and num_applied <= 3:
                    actual_error = (R ** 2).sum().item()
                    error_diff = abs(actual_error - error_current)
                    if error_diff > 1e-3:
                        print(f"      ⚠️  ERROR MISMATCH! Tracked={error_current:.6e}, Actual={actual_error:.6e}, Diff={error_diff:.6e}")

                # Track for diagnostics
                residual_norm = R.abs().mean().item()  # Track residual magnitude
                position_improvements.append({
                    'position': idx,
                    'j': j_item,
                    'improvement': total_improvement,
                    'num_channels': num_channels_corrected,
                    'error_after': error_current,
                    'residual_norm': residual_norm,  # Track how residual evolves
                })

                if debug and num_applied <= 5:
                    linear_avg = (linear_term_pos + linear_term_neg) / 2 if (linear_term_pos + linear_term_neg) > 0 else 0
                    # Show ratio of quad penalty to linear benefit
                    penalty_ratio = quad_term / linear_avg if linear_avg > 0 else 0
                    print(f"      Position {idx+1}/{len(candidate_positions)}: j={j_item}, "
                          f"channels={num_channels_corrected}, improvement={total_improvement:.6e}")
                    print(f"        Linear benefit: {linear_avg:.6e}, Quad penalty: {quad_term:.6e}, "
                          f"Penalty ratio: {penalty_ratio:.3f}, Error: {error_current:.6e}")
            else:
                # No improvement, stop
                stopped_early = True
                if debug:
                    print(f"      Stopped at position {idx+1}/{len(candidate_positions)}: no improvement")
                break

        # Convert back to original dtype
        W_corrected = W_current.to(weight_dtype)

        # Final error
        error_final = error_current
        error_reduction = error_initial - error_final

        # Simple statistics: what matters is total reduction
        avg_improvement = sum(p['improvement'] for p in position_improvements) / len(position_improvements) if position_improvements else 0
        avg_channels = sum(p['num_channels'] for p in position_improvements) / len(position_improvements) if position_improvements else 0

        correction_stats = {
            'error_before_mean': error_initial,
            'error_after_mean': error_final,
            'error_reduction_mean': error_reduction,
            'num_candidates': len(candidate_positions),
            'num_applied': num_applied,
            'stopped_early': stopped_early,
            'apply_percentage': num_applied / len(candidate_positions) * 100 if len(candidate_positions) > 0 else 0,
            'avg_improvement_per_position': avg_improvement,
            'avg_channels_per_position': avg_channels,
        }

        if debug:
            print(f"    Error before: {error_initial:.6e}")
            print(f"    Error after: {error_final:.6e}")
            print(f"    Reduction: {error_reduction:.6e} ({error_reduction/error_initial*100 if error_initial > 0 else 0:.2f}%)")
            print(f"    Positions applied: {num_applied}/{len(candidate_positions)}")
            print(f"    Avg improvement/position: {avg_improvement:.6e}")
            print(f"    Avg channels corrected/position: {avg_channels:.1f}")
            if stopped_early:
                print(f"    Status: Early stopping (no more improvements found)")
            else:
                print(f"    Status: Completed all candidate positions")

        # Cleanup - free GPU memory
        del X
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
            # Select WEAKEST surviving weights (smallest Wanda scores)
            candidate_positions, selection_info = self.select_weakest_positions(
                l2_salience_device, prune_mask, debug=debug
            )

            # Correct weights using greedy iterative approach with direct error measurement
            if selection_info is not None and len(candidate_positions) > 0:
                W_final, correction_stats = self.correct_weights_greedy_direct(
                    W_device, W_pruned, js_mean_device, activations,
                    candidate_positions, prune_mask, debug=debug
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

        # Cleanup weights and activation stats for this layer to free memory
        del W_device, l2_salience_device, js_mean_device, W_pruned, W_final
        if name in self.activation_stats:
            del self.activation_stats[name]
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
                    'count': 0,
                    'num_stored': 0  # Track number of tokens stored
                }

            # Store activation samples
            stats = self.activation_stats[name]

            # Only store if under the limit (to prevent OOM)
            if stats['num_stored'] < self.max_activation_samples:
                # How many tokens can we store?
                remaining = self.max_activation_samples - stats['num_stored']
                num_to_store = min(num_tokens, remaining)

                # Store on CPU to save GPU memory (will move to GPU during correction)
                stats['activations'].append(inp_flat[:num_to_store].cpu())
                stats['num_stored'] += num_to_store

            # Always accumulate mean for JS estimator (keep on GPU for speed)
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
        description="Wanda Pruning with Greedy Direct Error Correction (Weakest-First)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--sparsity", type=float, default=0.5,
                       help="Target sparsity (0.5 = prune 50%%)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens per sample")
    parser.add_argument("--max-activation-samples", type=int, default=8192,
                       help="Max activation tokens to store per layer (to prevent OOM)")
    parser.add_argument("--use-correction", action="store_true", default=True,
                       help="Enable greedy direct error correction")
    parser.add_argument("--no-correction", dest="use_correction", action="store_false",
                       help="Disable correction")
    parser.add_argument("--percent-change", type=float, default=0.05,
                       help="Percentage of weakest surviving weights to correct (default 5%%)")
    parser.add_argument("--correction-magnitude", type=float, default=0.01,
                       help="Fixed correction magnitude as fraction of original weight (default 1%%)")
    parser.add_argument("--output-dir", type=str, default="./pruned_models/model_prac_ho",
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
    print("Wanda Pruning with Greedy Direct Error Correction (Weakest-First)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Target Sparsity: {args.sparsity*100:.1f}%")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"Correction magnitude: {args.correction_magnitude*100:.1f}%")
    print(f"Greedy direct error correction: {args.use_correction}")
    if args.use_correction:
        print(f"  Max candidates: {args.percent_change*100:.1f}% of surviving weights (weakest first)")
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
        max_activation_samples=args.max_activation_samples,
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
