"""
Standard Heuristic AWQ - XL Version with lm_head Special Handling

This version extends awq_sh_7b.py with special handling for large layers (lm_head).

Key Features:
- Same base as awq_sh_7b.py (Heuristic-Guided Global Greedy Rounding)
- SPECIAL: Splits lm_head into halves to avoid OOM
- L2 salience metric
- Outlier masking
- Batched sequential quantization

Special Handling for lm_head:
- lm_head is often very large (e.g., [vocab_size × hidden_dim])
- For models with large vocab (32k-128k), this causes OOM
- Solution: Process output dimension in two halves
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

# Try to import calibration utils, fallback if not present
try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    print("⚠️ calibration_utils not found. Using internal fallback loaders.")
    def get_c4_calibration_data(*args, **kwargs): raise NotImplementedError("Please provide calibration_utils.py")
    def get_wikitext2_calibration_data(*args, **kwargs): raise NotImplementedError("Please provide calibration_utils.py")

class StandardHeuristicAWQQuantizerXL:
    def __init__(self, model, tokenizer, device="cuda", bits=4, n_grid=20,
                 group_size=128, use_heuristic=True, outlier_percent=0.05, max_tokens_per_sample=512,
                 layer_batch_size=16, lmhead_chunks=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bits = bits
        self.n_grid = n_grid
        self.group_size = group_size
        self.use_heuristic = use_heuristic
        self.outlier_percent = outlier_percent
        self.max_tokens_per_sample = max_tokens_per_sample
        self.layer_batch_size = layer_batch_size
        self.lmhead_chunks = lmhead_chunks

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_scales = {}

        print(f"\n[Standard Heuristic AWQ Quantizer XL Initialized]")
        print(f"  Target bits: {bits}")
        print(f"  Group size: {group_size}")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample")
        print(f"  Layer batch size: {layer_batch_size}")
        print(f"  Use heuristic: {use_heuristic}")
        if use_heuristic:
            print(f"  Outlier protection: Top {outlier_percent*100:.1f}% ignored")
            print(f"  Quantization: HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        else:
            print(f"  Quantization: STANDARD GROUP-WISE ASYMMETRIC [0, {2**bits - 1}]")
        print(f"  Special: lm_head split into {lmhead_chunks} chunks to avoid OOM")

    def get_hook(self, name):
        """Create a hook function for a specific layer."""
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Subsample tokens if sequence is too long (memory optimization)
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                indices = torch.randperm(seq_len)[:self.max_tokens_per_sample]
                indices = indices.sort()[0]  # Keep temporal order
                inp = inp[:, indices, :]

            # Store on CPU to save GPU memory, use float32 for numerical stability in L2 computation
            # We need float32 because squaring float16 values can overflow to inf
            self.activation_data[name].append(inp.detach().cpu().float())
        return hook

    @torch.no_grad()
    def get_activation_stats(self, name):
        """Compute L2 salience (E[X²]) and raw mean (E[X]) using FLOAT32 precision."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None, None

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

        salience = (l2_sum / total_samples)
        raw_mean = (mean_sum / total_samples)

        return salience, raw_mean

    @torch.no_grad()
    def quantize_weight_heuristic_groupwise(self, W, group_activation_means, apply_heuristic=True):
        """
        Vectorized implementation of 'quantize_groupwise_global_greedy'.
        """
        out_features, in_features = W.shape
        device = W.device

        # --- 1. Pre-processing / Padding ---
        n_groups = (in_features + self.group_size - 1) // self.group_size
        padded_in_features = n_groups * self.group_size

        if padded_in_features > in_features:
            W_padded = torch.zeros(out_features, padded_in_features, device=device, dtype=W.dtype)
            W_padded[:, :in_features] = W
            act_padded = torch.zeros(padded_in_features, device=device, dtype=W.dtype)
            act_padded[:in_features] = group_activation_means
        else:
            W_padded = W
            act_padded = group_activation_means

        # Reshape to groups for scaling
        W_g = W_padded.reshape(out_features, n_groups, self.group_size)

        # Asymmetric Quantization Setup
        w_min = W_g.min(dim=2, keepdim=True)[0]
        w_max = W_g.max(dim=2, keepdim=True)[0]
        max_int = 2**self.bits - 1

        scale = (w_max - w_min) / max_int
        scale = scale.clamp(min=1e-8)
        zp = torch.round(-w_min / scale).clamp(0, max_int)

        # Expand to full size [out, padded_in]
        scale_flat = scale.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)
        zp_flat = zp.repeat(1, 1, self.group_size).reshape(out_features, padded_in_features)

        # --- 2. Initial Quantization ---
        W_div = W_padded / scale_flat
        W_int = torch.round(W_div + zp_flat).clamp(0, max_int)
        W_quant = (W_int - zp_flat) * scale_flat

        if not apply_heuristic:
            # Return early if simple rounding (no flips)
            W_dequant = (W_int - zp_flat) * scale_flat
            if padded_in_features > in_features: W_dequant = W_dequant[:, :in_features]
            # Return empty flip stats
            flip_stats = {
                'total': 0, 'per_channel_mean': 0, 'per_channel_median': 0,
                'per_channel_min': 0, 'per_channel_max': 0, 'per_channel_std': 0,
                'per_channel_p25': 0, 'per_channel_p75': 0, 'per_channel_p90': 0,
                'per_channel_p95': 0, 'per_channel_p99': 0, 'per_channel_zero_pct': 100
            }
            return W_dequant.to(W.dtype), flip_stats

        # --- 3. Global Greedy Heuristic (Vectorized) ---

        # A. Calculate Current Error
        W_diff = W_padded - W_quant
        current_error = (W_diff * act_padded.unsqueeze(0)).sum(dim=1) # [out_features]

        # B. Identify Flip Candidates
        flip_dir = torch.sign(W_div + zp_flat - W_int)
        flip_dir[flip_dir == 0] = 1.0
        flip_impacts = act_padded.unsqueeze(0) * flip_dir * scale_flat # [out, in]

        # C. Validity Masks
        target_sign = torch.sign(current_error).unsqueeze(1)
        valid_mask = (torch.sign(flip_impacts) == target_sign)

        w_int_proposed = W_int + flip_dir
        in_range = (w_int_proposed >= 0) & (w_int_proposed <= max_int)
        valid_mask = valid_mask & in_range

        # Outlier Masking
        k_outliers = int(padded_in_features * self.outlier_percent)
        k_outliers = min(k_outliers, act_padded.numel())
        if k_outliers > 0:
            _, outlier_indices = torch.topk(act_padded.abs(), k_outliers)
            is_outlier = torch.zeros(padded_in_features, dtype=torch.bool, device=device)
            is_outlier[outlier_indices] = True
            valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

        # --- 4. Sorting & Optimization ---
        rounding_costs = (W_div + zp_flat - W_int).abs()
        rounding_costs_masked = rounding_costs.clone()
        rounding_costs_masked[~valid_mask] = -1.0

        sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)
        sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
        sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
        sorted_impacts = sorted_impacts * sorted_validity

        cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
        residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
        error_unsqueezed = torch.abs(current_error).unsqueeze(1)
        all_residuals = torch.cat([error_unsqueezed, residuals], dim=1)
        best_k = torch.argmin(all_residuals, dim=1)

        # --- 5. Apply Flips ---
        idx_range = torch.arange(padded_in_features, device=device).unsqueeze(0)
        flip_mask_sorted = idx_range < best_k.unsqueeze(1)
        final_flips_sorted = flip_mask_sorted & (sorted_validity.bool())

        sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
        sorted_flip_dir[~final_flips_sorted] = 0.0

        W_int.scatter_add_(1, sorted_indices, sorted_flip_dir)
        W_int.clamp_(0, max_int)

        # --- 6. Compute Flip Statistics ---
        # Total flips
        num_flips_total = final_flips_sorted.sum().item()

        # Per-channel flip statistics (sum across output dimension)
        flips_per_channel = final_flips_sorted.sum(dim=0).float()  # [padded_in_features]

        # Only consider actual channels (not padding)
        if padded_in_features > in_features:
            flips_per_channel = flips_per_channel[:in_features]

        # Compute comprehensive statistics
        flip_stats = {
            'total': num_flips_total,
            'per_channel_mean': flips_per_channel.mean().item(),
            'per_channel_median': flips_per_channel.median().item(),
            'per_channel_min': flips_per_channel.min().item(),
            'per_channel_max': flips_per_channel.max().item(),
            'per_channel_std': flips_per_channel.std().item(),
            # Percentiles
            'per_channel_p25': torch.quantile(flips_per_channel, 0.25).item(),
            'per_channel_p75': torch.quantile(flips_per_channel, 0.75).item(),
            'per_channel_p90': torch.quantile(flips_per_channel, 0.90).item(),
            'per_channel_p95': torch.quantile(flips_per_channel, 0.95).item(),
            'per_channel_p99': torch.quantile(flips_per_channel, 0.99).item(),
            # Percentage with 0 flips
            'per_channel_zero_pct': (flips_per_channel == 0).float().mean().item() * 100
        }

        # --- 7. Dequantize & Return ---
        W_dequant = (W_int - zp_flat) * scale_flat

        if padded_in_features > in_features:
            W_dequant = W_dequant[:, :in_features]

        return W_dequant.to(W.dtype), flip_stats

    @torch.no_grad()
    def search_best_scale(self, name, module):
        """Grid search for optimal per-input-channel scaling factor."""
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience, raw_mean = self.get_activation_stats(name)
        if activation_salience is None:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience = activation_salience.to(self.device).to(module.weight.dtype)
        raw_mean = raw_mean.to(self.device).to(module.weight.dtype)

        # Subsample for speed
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        max_samples = min(2048, X_cpu.shape[0])
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)

        del X_cpu

        if X_search.dtype != module.weight.dtype:
            X_search = X_search.to(module.weight.dtype)

        W = module.weight.data
        Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        # Clamp min salience
        activation_salience = activation_salience.clamp(min=1e-5)

        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid
            scales = activation_salience.pow(alpha)

            W_scaled = W * scales.unsqueeze(0)
            scaled_act_mean = raw_mean / scales

            W_quant, _ = self.quantize_weight_heuristic_groupwise(
                W_scaled,
                scaled_act_mean,
                apply_heuristic=self.use_heuristic
            )

            W_recon = W_quant / scales.unsqueeze(0)
            Y_quant = torch.matmul(X_search, W_recon.t())
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

            del W_scaled, W_quant, W_recon, Y_quant, scales

        del X_search, Y_orig
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def search_best_scale_lmhead_half(self, name, module, out_start, out_end, debug=False):
        """
        Grid search for lm_head, processing only half of the output dimension.

        Args:
            out_start: Starting index of output dimension
            out_end: Ending index of output dimension
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        activation_salience, raw_mean = self.get_activation_stats(name)
        if activation_salience is None:
            if debug:
                print(f"  DEBUG: No activation salience for {name}, using default scales")
            in_features = module.weight.shape[1]
            return torch.ones(in_features).to(self.device), 0.0, 0.0

        if debug:
            print(f"  DEBUG: Processing lm_head rows {out_start}:{out_end}")
            print(f"  DEBUG: Salience shape={activation_salience.shape}, "
                  f"mean={activation_salience.mean():.6f}, max={activation_salience.max():.6f}")

        activation_salience = activation_salience.to(self.device).to(module.weight.dtype)
        raw_mean = raw_mean.to(self.device).to(module.weight.dtype)

        # Prepare calibration data - use fewer samples for lm_head
        X_list = self.activation_data[name]
        X_cpu = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0)

        max_samples = min(1024, X_cpu.shape[0])  # Reduced for lm_head
        if X_cpu.shape[0] > max_samples:
            indices = torch.randperm(X_cpu.shape[0])[:max_samples]
            X_search = X_cpu[indices].to(self.device)
        else:
            X_search = X_cpu.to(self.device)

        del X_cpu

        if X_search.dtype != module.weight.dtype:
            X_search = X_search.to(module.weight.dtype)

        # Get only the slice of weights we're processing
        W_full = module.weight.data
        W = W_full[out_start:out_end, :]  # Slice output dimension

        # Compute original output for this slice
        Y_orig = torch.matmul(X_search, W.t())

        best_error = float('inf')
        best_alpha = 0.0
        best_scales = torch.ones(W.shape[1], device=self.device)

        activation_salience = activation_salience.clamp(min=1e-5)

        # Grid search over α
        for grid_idx in range(self.n_grid + 1):
            alpha = grid_idx / self.n_grid
            scales = activation_salience.pow(alpha)

            W_scaled = W * scales.unsqueeze(0)
            scaled_act_mean = raw_mean / scales

            W_quant, _ = self.quantize_weight_heuristic_groupwise(
                W_scaled,
                scaled_act_mean,
                apply_heuristic=self.use_heuristic
            )

            W_recon = W_quant / scales.unsqueeze(0)
            Y_quant = torch.matmul(X_search, W_recon.t())
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha
                best_scales = scales.clone()

            del W_scaled, W_quant, W_recon, Y_quant, scales

        del X_search, Y_orig, W
        torch.cuda.empty_cache()

        return best_scales, best_alpha, best_error

    @torch.no_grad()
    def quantize_lmhead_half_by_half(self, name, module, debug=False, num_chunks=4):
        """
        Quantize lm_head by splitting it into N chunks along output dimension.
        This reduces peak memory usage significantly.

        Args:
            num_chunks: Number of chunks to split into (default: 4 for very large lm_heads)
        """
        print(f"\n  🔧 Special handling for {name} (split into {num_chunks} chunks)")

        W = module.weight.data
        original_dtype = W.dtype
        out_features, in_features = W.shape

        print(f"     Shape: {W.shape} ({W.numel() / 1e6:.1f}M parameters)")

        # Get activation stats once for all chunks
        _, raw_mean = self.get_activation_stats(name)
        if raw_mean is None:
            raw_mean = torch.zeros(in_features, device=self.device, dtype=W.dtype)
        else:
            raw_mean = raw_mean.to(self.device).to(W.dtype)

        # Calculate chunk boundaries
        chunk_size = out_features // num_chunks
        chunk_boundaries = [(i * chunk_size,
                            out_features if i == num_chunks - 1 else (i + 1) * chunk_size)
                           for i in range(num_chunks)]

        W_final_chunks = []
        chunk_stats = []

        # Process each chunk
        for chunk_idx, (start_idx, end_idx) in enumerate(chunk_boundaries):
            print(f"     Processing chunk {chunk_idx + 1}/{num_chunks}: rows {start_idx}-{end_idx}")

            # Grid search for this chunk
            best_scales, best_alpha, best_error = self.search_best_scale_lmhead_half(
                name, module, start_idx, end_idx, debug=(debug and chunk_idx == 0)
            )

            # Scale and quantize this chunk
            W_chunk = W[start_idx:end_idx, :]
            W_scaled = W_chunk * best_scales.unsqueeze(0)
            scaled_act_mean = raw_mean / best_scales

            W_quant, flip_stats = self.quantize_weight_heuristic_groupwise(
                W_scaled,
                scaled_act_mean,
                apply_heuristic=self.use_heuristic
            )
            W_final_chunk = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)

            W_final_chunks.append(W_final_chunk)
            chunk_stats.append({
                'alpha': best_alpha,
                'error': best_error,
                'scales': best_scales,
                'flip_stats': flip_stats
            })

            # Cleanup
            del W_chunk, W_scaled, W_quant, W_final_chunk, scaled_act_mean
            torch.cuda.empty_cache()

        # Combine all chunks
        W_final = torch.cat(W_final_chunks, dim=0)
        module.weight.data = W_final

        # Store average statistics
        avg_alpha = np.mean([s['alpha'] for s in chunk_stats])
        avg_error = np.mean([s['error'] for s in chunk_stats])

        # Aggregate flip statistics
        total_flips = sum([s['flip_stats']['total'] for s in chunk_stats])
        avg_per_channel_mean = np.mean([s['flip_stats']['per_channel_mean'] for s in chunk_stats])
        avg_per_channel_median = np.mean([s['flip_stats']['per_channel_median'] for s in chunk_stats])
        avg_per_channel_std = np.mean([s['flip_stats']['per_channel_std'] for s in chunk_stats])
        avg_per_channel_p25 = np.mean([s['flip_stats']['per_channel_p25'] for s in chunk_stats])
        avg_per_channel_p75 = np.mean([s['flip_stats']['per_channel_p75'] for s in chunk_stats])
        avg_per_channel_p90 = np.mean([s['flip_stats']['per_channel_p90'] for s in chunk_stats])
        avg_per_channel_p95 = np.mean([s['flip_stats']['per_channel_p95'] for s in chunk_stats])
        avg_per_channel_p99 = np.mean([s['flip_stats']['per_channel_p99'] for s in chunk_stats])
        avg_per_channel_zero_pct = np.mean([s['flip_stats']['per_channel_zero_pct'] for s in chunk_stats])

        # Build detailed stats dict
        stats_dict = {
            'scales': chunk_stats[0]['scales'].cpu(),  # Use first chunk's scales
            'alpha': avg_alpha,
            'error': avg_error,
            'flip_stats': {
                'total': total_flips,
                'per_channel_mean': avg_per_channel_mean,
                'per_channel_median': avg_per_channel_median,
                'per_channel_std': avg_per_channel_std,
                'per_channel_p25': avg_per_channel_p25,
                'per_channel_p75': avg_per_channel_p75,
                'per_channel_p90': avg_per_channel_p90,
                'per_channel_p95': avg_per_channel_p95,
                'per_channel_p99': avg_per_channel_p99,
                'per_channel_zero_pct': avg_per_channel_zero_pct
            }
        }
        for i, stat in enumerate(chunk_stats):
            stats_dict[f'alpha_chunk{i+1}'] = stat['alpha']
            stats_dict[f'error_chunk{i+1}'] = stat['error']
            stats_dict[f'flips_chunk{i+1}'] = stat['flip_stats']['total']

        self.layer_scales[name] = stats_dict

        # Print summary
        alpha_str = ', '.join([f'α_{i+1}={s["alpha"]:.4f}' for i, s in enumerate(chunk_stats)])
        error_str = ', '.join([f'err_{i+1}={s["error"]:.8f}' for i, s in enumerate(chunk_stats)])
        flips_str = ', '.join([f'flips_{i+1}={s["flip_stats"]["total"]:,}' for i, s in enumerate(chunk_stats)])
        print(f"     ✓ Done: {alpha_str}")
        print(f"             {error_str}")
        print(f"             {flips_str}")

        del W_final_chunks, chunk_stats, W_final, raw_mean
        torch.cuda.empty_cache()

    @torch.no_grad()
    def quantize_layer(self, name, module):
        """Apply Standard Heuristic AWQ Quantization."""
        best_scales, best_alpha, best_error = self.search_best_scale(name, module)

        W = module.weight.data
        original_dtype = W.dtype
        W_scaled = W * best_scales.unsqueeze(0)

        _, raw_mean = self.get_activation_stats(name)
        if raw_mean is not None:
            scaled_act_mean = (raw_mean.to(self.device).to(W.dtype) / best_scales)
        else:
            scaled_act_mean = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)

        W_quant, flip_stats = self.quantize_weight_heuristic_groupwise(
            W_scaled,
            scaled_act_mean,
            apply_heuristic=self.use_heuristic
        )

        W_final = (W_quant / best_scales.unsqueeze(0)).to(original_dtype)
        module.weight.data = W_final

        self.layer_scales[name] = {
            'scales': best_scales.cpu(),
            'alpha': best_alpha,
            'error': best_error,
            'flip_stats': flip_stats
        }

        del best_scales, scaled_act_mean, W_scaled, W_quant, W_final
        if name in self.activation_data:
            del self.activation_data[name]
        torch.cuda.empty_cache()
        gc.collect()

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """Calibrate a batch of layers simultaneously."""
        print(f"  Calibrating {len(layer_names_batch)} layers...")

        self.model.eval()
        handles = []

        # Register hooks for all layers in this batch
        for name, module in layer_names_batch:
            handle = module.register_forward_hook(self.get_hook(name))
            handles.append((name, handle))

        # Run calibration
        successful = 0
        with torch.no_grad():
            for text in tqdm(calibration_data[:n_samples], desc="  Calibration", leave=False):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    self.model(**inputs, use_cache=False, return_dict=True)
                    successful += 1

                    # Periodic cache clearing
                    if (successful + 1) % 32 == 0:
                        torch.cuda.empty_cache()
                except Exception:
                    continue

        # Remove hooks
        for _, handle in handles:
            handle.remove()

        torch.cuda.empty_cache()
        gc.collect()

    def quantize_model_sequential(self, calibration_data, n_samples=500):
        """Batched sequential quantization with special lm_head handling."""
        print("\n" + "=" * 80)
        print("Batched Sequential Quantization (XL Version)")
        print("=" * 80)
        print(f"  Strategy: Process {self.layer_batch_size} layers per batch")

        if HAS_PSUTIL:
            initial_ram = psutil.virtual_memory().percent
            print(f"  Initial System RAM: {initial_ram:.1f}%")

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        num_layers = len(layer_names)
        num_batches = (num_layers + self.layer_batch_size - 1) // self.layer_batch_size

        print(f"  Total layers: {num_layers}")
        print(f"  Total batches: {num_batches}")
        print("=" * 80)

        quantized_count = 0

        # Process in batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.layer_batch_size
            batch_end = min(batch_start + self.layer_batch_size, num_layers)
            batch_layers = layer_names[batch_start:batch_end]

            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Layers {batch_start}-{batch_end-1}")

            # Calibrate this batch
            self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

            # Quantize all layers in this batch
            print(f"  Quantizing {len(batch_layers)} layers...")
            for name, module in tqdm(batch_layers, desc="  Quantization", leave=False):
                try:
                    # Check if this is lm_head (special handling)
                    is_lmhead = 'lm_head' in name.lower() or name.endswith('lm_head')

                    if is_lmhead:
                        # Use chunked processing for lm_head
                        debug = (quantized_count < 2)
                        self.quantize_lmhead_half_by_half(name, module, debug=debug, num_chunks=self.lmhead_chunks)
                    else:
                        # Standard processing
                        self.quantize_layer(name, module)

                    quantized_count += 1

                except Exception as e:
                    print(f"\n⚠️  Error quantizing {name}: {e}")
                    continue

            # Clear activations for this batch
            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                ram_pct = psutil.virtual_memory().percent
                print(f"  Batch {batch_idx+1} complete. RAM: {ram_pct:.1f}%")

        print("\n" + "=" * 80)
        print("✓ Batched Sequential Quantization Complete")
        print(f"  Total layers quantized: {quantized_count}/{num_layers}")
        print("=" * 80)

        if self.layer_scales:
            alphas = [info['alpha'] for info in self.layer_scales.values()]
            num_flips_list = [info.get('num_flips', 0) for info in self.layer_scales.values()]

            print(f"\nOptimal α statistics:")
            print(f"  Mean: {np.mean(alphas):.3f}")
            print(f"  Median: {np.median(alphas):.3f}")

            if self.use_heuristic:
                # Collect flip statistics
                flip_totals = [info.get('flip_stats', {}).get('total', 0) for info in self.layer_scales.values()]
                per_ch_means = [info.get('flip_stats', {}).get('per_channel_mean', 0) for info in self.layer_scales.values()]
                per_ch_medians = [info.get('flip_stats', {}).get('per_channel_median', 0) for info in self.layer_scales.values()]
                per_ch_stds = [info.get('flip_stats', {}).get('per_channel_std', 0) for info in self.layer_scales.values()]
                per_ch_p25s = [info.get('flip_stats', {}).get('per_channel_p25', 0) for info in self.layer_scales.values()]
                per_ch_p75s = [info.get('flip_stats', {}).get('per_channel_p75', 0) for info in self.layer_scales.values()]
                per_ch_p90s = [info.get('flip_stats', {}).get('per_channel_p90', 0) for info in self.layer_scales.values()]
                per_ch_p95s = [info.get('flip_stats', {}).get('per_channel_p95', 0) for info in self.layer_scales.values()]
                per_ch_p99s = [info.get('flip_stats', {}).get('per_channel_p99', 0) for info in self.layer_scales.values()]
                per_ch_zero_pcts = [info.get('flip_stats', {}).get('per_channel_zero_pct', 0) for info in self.layer_scales.values()]

                total_flips = np.sum(flip_totals)
                mean_flips_per_layer = np.mean(flip_totals)
                median_flips_per_layer = np.median(flip_totals)
                min_flips = np.min(flip_totals)
                max_flips = np.max(flip_totals)

                avg_per_ch_mean = np.mean(per_ch_means)
                avg_per_ch_median = np.mean(per_ch_medians)
                avg_per_ch_std = np.mean(per_ch_stds)
                avg_per_ch_p25 = np.mean(per_ch_p25s)
                avg_per_ch_p75 = np.mean(per_ch_p75s)
                avg_per_ch_p90 = np.mean(per_ch_p90s)
                avg_per_ch_p95 = np.mean(per_ch_p95s)
                avg_per_ch_p99 = np.mean(per_ch_p99s)
                avg_per_ch_zero_pct = np.mean(per_ch_zero_pcts)

                print(f"\nWeight Flipping Statistics:")
                print(f"  Total flips across all layers: {int(total_flips):,}")
                print(f"  Mean flips per layer: {mean_flips_per_layer:,.1f}")
                print(f"  Median flips per layer: {median_flips_per_layer:,.1f}")
                print(f"  Min: {int(min_flips):,} | Max: {int(max_flips):,}")
                print(f"\n  Per-Channel Statistics (averaged across layers):")
                print(f"    Mean flips per channel: {avg_per_ch_mean:.2f}")
                print(f"    Median flips per channel: {avg_per_ch_median:.2f}")
                print(f"    Std dev: {avg_per_ch_std:.2f}")
                print(f"\n    Percentiles:")
                print(f"      25th: {avg_per_ch_p25:.2f} | 50th: {avg_per_ch_median:.2f} | 75th: {avg_per_ch_p75:.2f}")
                print(f"      90th: {avg_per_ch_p90:.2f} | 95th: {avg_per_ch_p95:.2f} | 99th: {avg_per_ch_p99:.2f}")
                print(f"\n    Channels with 0 flips: {avg_per_ch_zero_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--use-heuristic", action="store_true", default=True,
                       help="Enable heuristic rounding (default: True)")
    parser.add_argument("--no-heuristic", dest="use_heuristic", action="store_false",
                       help="Disable heuristic rounding")
    parser.add_argument("--outlier-percent", type=float, default=0.05,
                       help="Percent of outliers to ignore (default: 0.05)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens to store per sample (default: 2048)")
    parser.add_argument("--layer-batch-size", type=int, default=16,
                       help="Number of layers to process per batch (default: 16)")
    parser.add_argument("--lmhead-chunks", type=int, default=4,
                       help="Number of chunks to split lm_head into (default: 4, higher = less memory)")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/model_awq_sh_xl")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3",
                       help="Model name or local path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"],
                       help="Calibration dataset (default: c4)")
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache",
                       help="Directory to cache calibration data (default: ./calibration_cache)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use model path from args
    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Standard Heuristic AWQ (XL Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Group size: {args.group_size}")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"Use heuristic: {args.use_heuristic}")
    print(f"Special: lm_head split into halves")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Fix for Llama/Mistral models lacking pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16 for better numerical stability
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load calibration data
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)

    quantizer = StandardHeuristicAWQQuantizerXL(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=4,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_heuristic=args.use_heuristic,
        outlier_percent=args.outlier_percent,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks
    )

    # Use batched sequential quantization (optimal memory/speed balance)
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
