"""
Wanda Pruning with Full Hessian Correction

Implements optimal reconstruction correction using the true XX^T (full empirical covariance matrix).
This moves from a Diagonal Hessian approximation to a Full Hessian correction, equivalent to
solving the Normal Equations for the reconstruction error.

Mathematical Foundation:
    Objective: min_ΔW ||X W_dense^T - X (W_pruned + ΔW)^T||_2^2

    With full H = XX^T, the optimal correction is:
        ΔW_surviving = (W_dense - W_pruned) (XX^T) (XX^T)_surviving^(-1)

    Or equivalently, for surviving weights:
        H_sub @ ΔW = -error_vector
        where H_sub is the submatrix of H at selected positions

Key Differences from Diagonal (prac_xl.py):
1. Calibration: Accumulate full Hessian matrix H = X^T X [hidden_dim, hidden_dim]
2. Correction: Solve linear system H_sub @ ΔW = -error using Cholesky decomposition
3. Memory: O(hidden_dim^2) instead of O(hidden_dim), requires careful batching

Usage:
    python prac_full.py --sparsity 0.5 --percent-change 0.05 --damping 1e-5
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


class WandaPrunerWithFullHessian:
    """
    Wanda Pruning with Full Hessian Correction.

    Uses the true XX^T for optimal reconstruction correction.
    """

    def __init__(self, model, tokenizer, device="cuda", sparsity=0.5,
                 max_tokens_per_sample=512, use_correction=True,
                 knee_tolerance=0.0, offset_percent=0.0, percent_change=0.05,
                 max_correction_magnitude=0.05, damping=1e-5):
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
        self.damping = damping

        # Storage for full Hessian matrices
        # Memory: O(hidden_dim^2) per layer - stored on CPU to save GPU memory
        self.activation_stats = {}  # Dict of {name: {'hessian': tensor, 'mean_sum': tensor, 'count': int}}
        self.hooks = []
        self.layer_stats = {}

        print(f"\n[Wanda Pruner with Full Hessian Correction]")
        print(f"  Target sparsity: {sparsity*100:.1f}% (keep {(1-sparsity)*100:.1f}%)")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample")
        print(f"  Salience metric: ||X_j||_2 (L2 norm from full Hessian diagonal)")
        print(f"  Scoring: Score_ij = |W_ij| * ||X_j||_2")
        print(f"  Hessian: Full XX^T matrix [hidden_dim, hidden_dim]")
        print(f"  Damping: {damping} (for numerical stability)")
        if use_correction:
            print(f"  Full Hessian correction: ENABLED")
            print(f"    - Correction percent: {percent_change*100:.1f}% of weights")
            print(f"    - Max magnitude: {max_correction_magnitude*100:.1f}% of original weight")
            print(f"    - Solver: Cholesky decomposition")
        else:
            print(f"  Correction: DISABLED")
        print(f"  Special: lm_head is NOT pruned")

    @torch.no_grad()
    def get_activation_stats(self, name):
        """
        Compute L2 salience and James-Stein mean from full Hessian.

        Returns:
            tuple: (l2_salience, js_mean, raw_mean, hessian)
        """
        if name not in self.activation_stats:
            return None, None, None, None

        stats = self.activation_stats[name]

        # Check if we have any data
        if stats['count'] == 0:
            return None, None, None, None

        # L2 salience from Hessian diagonal: sqrt(H_jj / count)
        hessian = stats['hessian']
        l2_salience = torch.sqrt(torch.diag(hessian) / stats['count'])

        # Mean from accumulated sums
        raw_mean = stats['mean_sum'] / stats['count']

        # Apply James-Stein estimator
        js_mean = compute_james_stein_mean(raw_mean)

        return l2_salience, js_mean, raw_mean, hessian

    @torch.no_grad()
    def select_moderate_positions(self, js_mean, prune_mask, debug=False):
        """
        Select moderate positions among SURVIVING (non-pruned) weights.

        Same logic as diagonal version - only the correction step differs.
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

        # Sort descending
        sorted_values, sorted_order = torch.sort(abs_js_mean_surviving, descending=True)
        sorted_indices = surviving_indices[sorted_order]

        num_surviving = len(surviving_indices)

        # Apply Kneedle on first half
        first_half = sorted_values[:num_surviving // 2]
        if len(first_half) < 3:
            num_to_select = max(1, int(self.percent_change * num_surviving))
            selected_indices = sorted_indices[:num_to_select]
            knee_info = {
                'knee_idx': 0,
                'knee_value': sorted_values[0].item() if len(sorted_values) > 0 else 0,
                'start_idx': 0,
                'end_idx': num_to_select,
                'num_selected': num_to_select,
                'num_surviving': num_surviving,
                'selection_pct': num_to_select / num_surviving * 100,
            }
            if debug:
                print(f"    Moderate selection (few survivors): {num_to_select}/{num_surviving} ({knee_info['selection_pct']:.2f}%)")
            return selected_indices, knee_info

        knee_idx = find_knee_point(first_half, tolerance_offset=self.knee_tolerance)

        # Start position
        offset_indices = int(self.offset_percent * num_surviving)
        start_idx = knee_idx + offset_indices
        start_idx = max(0, min(start_idx, num_surviving - 1))

        # Number to select
        num_to_select = max(1, int(self.percent_change * num_surviving))
        end_idx = min(start_idx + num_to_select, num_surviving)

        # Adjust if needed
        if end_idx == num_surviving and start_idx > 0:
            start_idx = max(0, num_surviving - num_to_select)

        actual_num = end_idx - start_idx

        # Get selected indices
        selected_indices = sorted_indices[start_idx:end_idx]

        knee_info = {
            'knee_idx': knee_idx,
            'knee_value': sorted_values[knee_idx].item(),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'num_selected': actual_num,
            'num_surviving': num_surviving,
            'selection_pct': actual_num / num_surviving * 100 if num_surviving > 0 else 0,
        }

        if debug:
            print(f"    Surviving channels: {num_surviving}")
            print(f"    Moderate selection: {actual_num} positions ({knee_info['selection_pct']:.2f}%)")
            print(f"    Knee idx: {knee_idx}, value: {knee_info['knee_value']:.6f}")

        return selected_indices, knee_info

    @torch.no_grad()
    def correct_weights_full_hessian(self, W_orig, W_pruned, js_mean, hessian,
                                     selected_positions, prune_mask, debug=False):
        """
        Full Hessian weight correction using linear system solver.

        Mathematical approach:
            For each output channel i, we solve:
                H_sub @ ΔW[i, selected] = -error[i] * 1_vector

            where H_sub = hessian[selected][:, selected] + damping * I

        This is the optimal correction that minimizes reconstruction error
        under the assumption that only selected positions can be modified.

        Args:
            W_orig: Original weights [out_features, in_features]
            W_pruned: Pruned weights [out_features, in_features]
            js_mean: James-Stein mean [in_features]
            hessian: Full Hessian matrix [in_features, in_features]
            selected_positions: Indices of moderate positions
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
        js_mean_f32 = js_mean.float()

        # Move Hessian to device
        hessian_f32 = hessian.float().to(self.device)

        # Compute reconstruction errors for all channels
        W_diff = W_pruned_f32 - W_orig_f32
        errors = torch.matmul(W_diff, js_mean_f32)  # [out_features]

        if len(selected_positions) == 0:
            if debug:
                print(f"    ⚠️ No positions selected, no correction")
            return W_pruned.clone(), None

        # Extract Hessian submatrix for selected positions
        selected_positions_dev = selected_positions.to(self.device)
        H_sub = hessian_f32[selected_positions_dev][:, selected_positions_dev]  # [num_selected, num_selected]

        # Add damping for numerical stability: H_sub = H_sub + λI
        H_sub = H_sub + self.damping * torch.eye(len(selected_positions), device=self.device)

        # Vectorized solution for all output channels at once
        W_corrected = W_pruned_f32.clone()
        solve_failures = 0

        # Build RHS matrix: [out_features, num_selected]
        # Each row i has: -error[i] for all selected positions
        out_features = W_orig.shape[0]
        num_selected = len(selected_positions)
        RHS = torch.ones(out_features, num_selected, device=self.device) * (-errors.unsqueeze(1))

        try:
            # Solve: H_sub @ X = RHS^T
            # where X is [num_selected, out_features] and we want Delta = X^T
            # Using Cholesky decomposition for stability
            L = torch.linalg.cholesky(H_sub)
            # cholesky_solve(B, L) solves A @ X = B where A = L @ L^T
            # RHS.t() is [num_selected, out_features], so we solve for each column
            delta_W_all = torch.cholesky_solve(RHS.t(), L).t()
            # delta_W_all is now [out_features, num_selected]

        except RuntimeError as e:
            # If Cholesky fails, fall back to lstsq (more robust but slower)
            if debug:
                print(f"    ⚠️ Cholesky failed, using lstsq: {e}")
            delta_W_all = torch.linalg.lstsq(H_sub, RHS.t()).solution.t()
            solve_failures = 1

        # Apply magnitude constraint: |ΔW[i,j]| ≤ max_correction_magnitude * |W_orig[i,j]|
        # max_change: [out_features, num_selected]
        max_change = self.max_correction_magnitude * W_orig_f32[:, selected_positions].abs()
        delta_W_clamped = torch.clamp(delta_W_all, -max_change, max_change)

        # Only apply to surviving weights
        # survivors_mask: [out_features, num_selected]
        survivors_mask = prune_mask[:, selected_positions].float()
        delta_W_masked = delta_W_clamped * survivors_mask

        # Track clamping statistics
        was_clamped = ((delta_W_all.abs() > max_change) & (survivors_mask > 0))
        num_clamped = was_clamped.sum().item()
        total_corrections = (survivors_mask > 0).sum().item()

        # Apply correction (vectorized)
        W_corrected[:, selected_positions] += delta_W_masked

        # Convert back to original dtype
        W_corrected = W_corrected.to(weight_dtype)

        # Compute correction statistics
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
            'solve_failures': solve_failures,
        }

        if debug:
            print(f"    Error before: {correction_stats['error_before_mean']:.6f}")
            print(f"    Error after: {correction_stats['error_after_mean']:.6f}")
            print(f"    Reduction: {error_reduction:.6f}")
            print(f"    Clamped: {num_clamped}/{total_corrections} ({correction_stats['clamp_percentage']:.1f}%)")
            if solve_failures > 0:
                print(f"    Solve failures: {solve_failures} (used lstsq fallback)")

        # Cleanup
        del hessian_f32, H_sub
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
        Prune layer with optional full Hessian correction.
        """
        if name not in self.activation_stats:
            print(f"  ⚠️  No activation stats for {name}, skipping")
            return

        # Get activation statistics
        l2_salience, js_mean, _, hessian = self.get_activation_stats(name)
        if l2_salience is None:
            if debug:
                print(f"  DEBUG: No activation stats for {name}, skipping")
            return

        if debug:
            print(f"  DEBUG: Layer {name}")
            print(f"    L2 salience shape: {l2_salience.shape}")
            print(f"    JS mean range: [{js_mean.min():.6f}, {js_mean.max():.6f}]")
            print(f"    Hessian shape: {hessian.shape}")

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

        # Step 2: Full Hessian correction (if enabled)
        if self.use_correction:
            # Select moderate positions
            selected_positions, knee_info = self.select_moderate_positions(
                js_mean_device, prune_mask, debug=debug
            )

            # Correct weights using full Hessian
            if knee_info is not None and len(selected_positions) > 0:
                W_final, correction_stats = self.correct_weights_full_hessian(
                    W_device, W_pruned, js_mean_device, hessian,
                    selected_positions, prune_mask, debug=debug
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
        Create hook for accumulating full Hessian matrix.

        Key difference from diagonal version:
            Accumulates H = X^T X [hidden_dim, hidden_dim] instead of just diag(X^T X)

        Memory: O(hidden_dim^2) stored on CPU to avoid GPU OOM
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
                    'hessian': torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32),
                    'mean_sum': torch.zeros(hidden_dim, dtype=torch.float32),
                    'count': 0
                }

            # Accumulate full Hessian: H += X^T X
            # For memory efficiency, compute in batches if needed
            stats = self.activation_stats[name]

            # Compute X^T X incrementally on GPU, then move to CPU
            hessian_update = torch.matmul(inp_flat.t(), inp_flat)  # [hidden_dim, hidden_dim]
            stats['hessian'] += hessian_update.cpu()

            # Also accumulate mean for JS estimator
            stats['mean_sum'] += inp_flat.sum(dim=0).cpu()
            stats['count'] += num_tokens

            del inp_flat, hessian_update

        return hook

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """
        Calibrate batch of layers using full Hessian accumulation.

        Note: This requires O(hidden_dim^2) memory per layer batch.
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
        Batched sequential pruning with full Hessian correction.

        Note: Reduced default batch size to 4 due to O(hidden_dim^2) memory per layer.
        """
        print("\n" + "=" * 80)
        print("BATCHED SEQUENTIAL PRUNING WITH FULL HESSIAN CORRECTION")
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
                    clamp_pcts = [self.layer_stats[k]['correction_stats']['clamp_percentage']
                                 for k in corrected_layers]
                    solve_failures = [self.layer_stats[k]['correction_stats']['solve_failures']
                                     for k in corrected_layers]

                    print(f"\nFull Hessian correction statistics ({len(corrected_layers)} layers):")
                    print(f"  Mean error before: {np.mean(errors_before):.6f}")
                    print(f"  Mean error after: {np.mean(errors_after):.6f}")
                    print(f"  Mean reduction: {np.mean(reductions):.6f}")
                    print(f"  Mean clamp %: {np.mean(clamp_pcts):.1f}%")
                    print(f"  Total solve failures: {sum(solve_failures)} (used lstsq fallback)")

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
        description="Wanda Pruning with Full Hessian Correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--sparsity", type=float, default=0.5,
                       help="Target sparsity (0.5 = prune 50%%)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens per sample")
    parser.add_argument("--use-correction", action="store_true", default=True,
                       help="Enable full Hessian correction")
    parser.add_argument("--no-correction", dest="use_correction", action="store_false",
                       help="Disable correction")
    parser.add_argument("--knee-tolerance", type=float, default=0.0,
                       help="Tolerance for Kneedle algorithm")
    parser.add_argument("--offset-percent", type=float, default=0.0,
                       help="Offset from knee point (as fraction)")
    parser.add_argument("--percent-change", type=float, default=0.05,
                       help="Percentage of weights to correct (default 5%%)")
    parser.add_argument("--max-correction-magnitude", type=float, default=0.05,
                       help="Max correction magnitude as fraction of original weight (default 5%%)")
    parser.add_argument("--damping", type=float, default=1e-5,
                       help="Damping factor for Hessian (numerical stability)")
    parser.add_argument("--output-dir", type=str, default="./pruned_models/model_prac_full",
                       help="Output directory")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3",
                       help="Model path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--layer-batch-size", type=int, default=4,
                       help="Layers per batch (reduced for full Hessian)")
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
    print("Wanda Pruning with Full Hessian Correction")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Target Sparsity: {args.sparsity*100:.1f}%")
    print(f"Layer Batch Size: {args.layer_batch_size} (reduced for O(d²) memory)")
    print(f"Damping: {args.damping}")
    print(f"Full Hessian correction: {args.use_correction}")
    if args.use_correction:
        print(f"  Correction percent: {args.percent_change*100:.1f}%")
        print(f"  Max correction magnitude: {args.max_correction_magnitude*100:.1f}%")
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
    pruner = WandaPrunerWithFullHessian(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sparsity=args.sparsity,
        max_tokens_per_sample=args.max_tokens_per_sample,
        use_correction=args.use_correction,
        knee_tolerance=args.knee_tolerance,
        offset_percent=args.offset_percent,
        percent_change=args.percent_change,
        max_correction_magnitude=args.max_correction_magnitude,
        damping=args.damping
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
