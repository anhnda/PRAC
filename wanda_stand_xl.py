"""
Wanda Pruning Implementation with L2 Salience
ADAPTED FOR: Extra Large Models (XL) - Special handling for large layers like lm_head

Key Features:
- Structured magnitude pruning based on activation salience
- Uses E[X²] (L2 norm) for activation salience
- Scoring: Score_ij = |W_ij| * E[X²]_j
- Per-row (output channel) top-p% selection
- Batched sequential processing for memory efficiency

Algorithm:
1. Compute per-input-channel salience: s[j] = E[X[:, j]²] (L2 norm)
2. Score each weight: Score_ij = |W_ij| * s[j]
3. For each output channel (row i):
   - Keep top p% highest scoring weights
   - Zero out the remaining weights
4. Save pruned model

Special Handling for lm_head:
- lm_head is often very large (e.g., [vocab_size × hidden_dim])
- For models with large vocab (32k-128k), this causes OOM
- Solution: Process output dimension in chunks
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
    print("   Install with: pip install psutil")

# Import your calibration utils (assuming they exist in the same folder)
from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data

class WandaPruner:
    """
    Wanda Pruning with L2 Salience.
    Special handling for large layers (lm_head).
    """

    def __init__(self, model, tokenizer, device="cuda", sparsity=0.5, max_tokens_per_sample=512, lmhead_chunks=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sparsity = sparsity  # Percentage of weights to PRUNE (e.g., 0.5 means keep 50%, prune 50%)
        self.max_tokens_per_sample = max_tokens_per_sample  # Subsample to save memory
        self.lmhead_chunks = lmhead_chunks

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_stats = {}

        print(f"\n[Wanda Pruner Initialized - XL Version]")
        print(f"  Target sparsity: {sparsity*100:.1f}% (keep {(1-sparsity)*100:.1f}%)")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample (memory optimization)")
        print(f"  Salience metric: E[X²] (L2 norm)")
        print(f"  Scoring: Score_ij = |W_ij| * E[X²]_j")
        print(f"  Special: lm_head split into {lmhead_chunks} chunks to avoid OOM")


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

        # Accumulate L2 salience on CPU to save GPU VRAM
        salience_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            # Use pow(2) for L2 Norm
            # Ensure we're working with float32 for numerical stability
            x_flat = x_flat.float()
            salience_sum += x_flat.pow(2).sum(dim=0)

        salience = salience_sum / total_samples
        return salience

    @torch.no_grad()
    def prune_weights_wanda(self, W, salience, sparsity):
        """
        Wanda pruning: Score_ij = |W_ij| * salience_j
        For each row (output channel), keep top (1-sparsity)% weights, zero out the rest.

        Args:
            W: Weight matrix [out_features, in_features]
            salience: Per-input-channel salience [in_features]
            sparsity: Percentage of weights to prune (e.g., 0.5 = prune 50%)

        Returns:
            Pruned weight matrix
        """
        out_features, in_features = W.shape

        # Compute scores: Score_ij = |W_ij| * salience_j
        # Broadcasting: W.abs() is [out, in], salience is [in] -> [out, in]
        scores = W.abs() * salience.unsqueeze(0)

        # For each output channel (row), determine threshold
        num_to_keep = int(in_features * (1 - sparsity))

        # Create mask
        mask = torch.zeros_like(W, dtype=torch.bool)

        for i in range(out_features):
            # Get scores for this row
            row_scores = scores[i]

            # Get indices of top-k elements
            if num_to_keep > 0:
                _, top_indices = torch.topk(row_scores, num_to_keep, largest=True)
                mask[i, top_indices] = True

        # Apply mask
        W_pruned = W * mask.float()

        # Calculate actual sparsity
        actual_sparsity = (W_pruned == 0).float().mean().item()

        return W_pruned, actual_sparsity

    @torch.no_grad()
    def prune_layer(self, name, module, debug=False):
        """
        Prune a single layer using Wanda method.
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            print(f"  ⚠️  No activation data for {name}, skipping pruning")
            return

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            if debug:
                print(f"  DEBUG: No activation salience for {name}, skipping")
            return

        if debug:
            print(f"  DEBUG: Got salience for {name}, shape={activation_salience.shape}, "
                  f"mean={activation_salience.mean():.6f}, max={activation_salience.max():.6f}")

        W = module.weight.data
        original_dtype = W.dtype

        # Move to device for computation
        W_device = W.to(self.device)
        salience_device = activation_salience.to(self.device)

        # Apply Wanda pruning
        W_pruned, actual_sparsity = self.prune_weights_wanda(W_device, salience_device, self.sparsity)

        # Update module weights
        module.weight.data = W_pruned.to(original_dtype)

        # Store statistics
        self.layer_stats[name] = {
            'target_sparsity': self.sparsity,
            'actual_sparsity': actual_sparsity,
            'salience_mean': activation_salience.mean().item(),
            'salience_max': activation_salience.max().item()
        }

        if debug:
            print(f"  → Target sparsity: {self.sparsity*100:.1f}%, Actual: {actual_sparsity*100:.2f}%")

    @torch.no_grad()
    def prune_lmhead_chunked(self, name, module, debug=False, num_chunks=4):
        """
        Prune lm_head by splitting it into N chunks along output dimension.
        This reduces peak memory usage significantly.

        Args:
            num_chunks: Number of chunks to split into (default: 4 for very large lm_heads)
        """
        print(f"\n  🔧 Special handling for {name} (split into {num_chunks} chunks)")

        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            print(f"  ⚠️  No activation data for {name}, skipping pruning")
            return

        # Get L2 activation salience
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            print(f"  ⚠️  No activation salience for {name}, skipping")
            return

        W = module.weight.data
        original_dtype = W.dtype
        out_features, in_features = W.shape

        print(f"     Shape: {W.shape} ({W.numel() / 1e6:.1f}M parameters)")

        # Calculate chunk boundaries
        chunk_size = out_features // num_chunks
        chunk_boundaries = [(i * chunk_size,
                            out_features if i == num_chunks - 1 else (i + 1) * chunk_size)
                           for i in range(num_chunks)]

        W_pruned_chunks = []
        chunk_stats = []

        salience_device = activation_salience.to(self.device)

        # Process each chunk
        for chunk_idx, (start_idx, end_idx) in enumerate(chunk_boundaries):
            print(f"     Processing chunk {chunk_idx + 1}/{num_chunks}: rows {start_idx}-{end_idx}")

            # Get chunk of weights
            W_chunk = W[start_idx:end_idx, :].to(self.device)

            # Apply Wanda pruning to this chunk
            W_pruned_chunk, actual_sparsity = self.prune_weights_wanda(W_chunk, salience_device, self.sparsity)

            W_pruned_chunks.append(W_pruned_chunk.to(original_dtype))
            chunk_stats.append({
                'actual_sparsity': actual_sparsity
            })

            # Cleanup
            del W_chunk, W_pruned_chunk
            torch.cuda.empty_cache()

        # Combine all chunks
        W_final = torch.cat(W_pruned_chunks, dim=0)
        module.weight.data = W_final

        # Store average statistics
        avg_sparsity = np.mean([s['actual_sparsity'] for s in chunk_stats])

        self.layer_stats[name] = {
            'target_sparsity': self.sparsity,
            'actual_sparsity': avg_sparsity,
            'salience_mean': activation_salience.mean().item(),
            'salience_max': activation_salience.max().item(),
        }

        for i, stat in enumerate(chunk_stats):
            self.layer_stats[name][f'sparsity_chunk{i+1}'] = stat['actual_sparsity']

        # Print summary
        sparsity_str = ', '.join([f's_{i+1}={s["actual_sparsity"]*100:.2f}%' for i, s in enumerate(chunk_stats)])
        print(f"     ✓ Done: {sparsity_str}")

        del W_pruned_chunks, chunk_stats, W_final
        torch.cuda.empty_cache()

    def calibrate_layer_batch(self, layer_names_batch, calibration_data, n_samples=500):
        """
        Calibrate a BATCH of layers simultaneously.
        """
        # Clear any previous activation data
        self.activation_data = {}

        # Register hooks for ALL layers in this batch
        handles = []
        for name, module in layer_names_batch:
            handle = module.register_forward_hook(self.get_hook(name))
            handles.append((name, handle))

        # Run calibration data through model ONCE for all layers in batch
        successful_passes = 0
        with torch.no_grad():
            for i, text in enumerate(calibration_data[:n_samples]):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    _ = self.model(**inputs, use_cache=False, return_dict=True)
                    successful_passes += 1
                    del inputs

                    # Aggressive cleanup
                    if (i + 1) % 10 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                except Exception as e:
                    if i == 0:
                        print(f"\n⚠️  Forward pass error: {str(e)[:100]}")
                    continue

        # Remove all hooks
        for name, handle in handles:
            handle.remove()

        if successful_passes == 0:
            print(f"\n❌ FATAL: No successful forward passes for batch!")

        # Verify activations were captured
        for name, _ in layer_names_batch:
            if name not in self.activation_data:
                # Some layers might not be called in every pass (rare for Linear)
                self.activation_data[name] = []

    def get_hook(self, name):
        """Create a hook function for a specific layer."""
        def hook(_module, input, _output):
            if name not in self.activation_data:
                self.activation_data[name] = []
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Subsample tokens if sequence is too long
            if inp.dim() == 3 and inp.shape[1] > self.max_tokens_per_sample:
                seq_len = inp.shape[1]
                # Random subsample is better for coverage
                indices = torch.randperm(seq_len, device=inp.device)[:self.max_tokens_per_sample]
                indices = indices.sort()[0]
                inp = inp[:, indices, :]

            # Store activation on CPU (use float32 for numerical stability in L2 computation)
            inp_stored = inp.detach().cpu().float().clone()
            self.activation_data[name].append(inp_stored)
            del inp
        return hook

    def prune_model_sequential(self, calibration_data, n_samples=500, layer_batch_size=16):
        """
        BATCHED SEQUENTIAL PRUNING with special handling for lm_head.
        """
        print("\n" + "=" * 80)
        print("BATCHED SEQUENTIAL PRUNING (XL Version)")
        print("=" * 80)

        if HAS_PSUTIL:
            initial_ram = psutil.virtual_memory().percent
            print(f"Initial System RAM: {initial_ram:.1f}%")

        layer_names = [(name, module) for name, module in self.model.named_modules()
                       if isinstance(module, nn.Linear)]

        print(f"\nFound {len(layer_names)} linear layers to prune")
        print(f"Batch size: {layer_batch_size} layers per batch")
        num_batches = (len(layer_names) + layer_batch_size - 1) // layer_batch_size
        print(f"Total batches: {num_batches}")

        pruned_count = 0
        skipped_count = 0

        # Process layers in batches
        for batch_idx in range(num_batches):
            batch_start = batch_idx * layer_batch_size
            batch_end = min(batch_start + layer_batch_size, len(layer_names))
            batch_layers = layer_names[batch_start:batch_end]

            print(f"\n{'='*60}")
            print(f"Batch {batch_idx + 1}/{num_batches}: Layers {batch_start}-{batch_end-1}")
            print(f"{'='*60}")

            # STEP 1: Calibrate this BATCH
            self.calibrate_layer_batch(batch_layers, calibration_data, n_samples)

            # STEP 2: Prune each layer in the batch
            for idx_in_batch, (name, module) in enumerate(tqdm(batch_layers, desc=f"Pruning Batch {batch_idx+1}")):
                try:
                    # Check if this is lm_head (special handling)
                    is_lmhead = 'lm_head' in name.lower() or name.endswith('lm_head')

                    if is_lmhead:
                        # Use chunked processing for lm_head
                        debug = (pruned_count < 2)
                        self.prune_lmhead_chunked(name, module, debug=debug, num_chunks=self.lmhead_chunks)
                    else:
                        # Standard processing for other layers
                        # Debug output for first few layers
                        if pruned_count < 2:
                            print(f"\nDEBUG Layer {pruned_count}: {name}")
                            self.prune_layer(name, module, debug=True)
                        else:
                            self.prune_layer(name, module)

                    pruned_count += 1

                except Exception as e:
                    print(f"\n⚠️  Error pruning {name}: {e}")
                    skipped_count += 1
                    continue

            # STEP 3: Clear activations
            self.activation_data = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                ram_pct = psutil.virtual_memory().percent
                print(f"Batch {batch_idx+1} complete. RAM: {ram_pct:.1f}%")

        print(f"\n✅ Sequential Pruning Complete!")
        print(f"   Total layers pruned: {pruned_count}/{len(layer_names)}")

        if self.layer_stats:
            sparsities = [info['actual_sparsity'] for info in self.layer_stats.values()]
            print(f"\nActual sparsity statistics:")
            print(f"  Mean: {np.mean(sparsities)*100:.2f}%")
            print(f"  Median: {np.median(sparsities)*100:.2f}%")
            print(f"  Min: {np.min(sparsities)*100:.2f}%")
            print(f"  Max: {np.max(sparsities)*100:.2f}%")

        # Final cleanup
        self.activation_data = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def load_wikitext2_simple(n_samples=128):
    from datasets import load_dataset
    print(f"Loading WikiText-2 (simple/fast approach)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
    return texts[:n_samples]

def main():
    parser = argparse.ArgumentParser(
        description="Wanda Pruning with L2 Salience for XL Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--sparsity", type=float, default=0.5,
                       help="Target sparsity (0.5 = prune 50%% of weights)")
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048,
                       help="Max tokens to store per sample. Lower this if OOM.")
    parser.add_argument("--output-dir", type=str, default="./pruned_models/model_wanda_xl",
                       help="Output directory")
    parser.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3",
                       help="Model name or local path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calib-dataset", type=str, default="c4",
                       choices=["c4", "wikitext2", "wikitext2-simple"],
                       help="Calibration dataset")
    parser.add_argument("--layer-batch-size", type=int, default=16,
                       help="Number of layers to calibrate simultaneously. "
                            "XL models require smaller batches due to larger hidden dim.")
    parser.add_argument("--lmhead-chunks", type=int, default=4,
                       help="Number of chunks to split lm_head into (default: 4, higher = less memory)")
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache",
                       help="Directory to cache calibration data (default: ./calibration_cache)")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Use model path from args
    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Wanda Pruning with L2 Salience (XL Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Target Sparsity: {args.sparsity*100:.1f}%")
    print(f"Layer Batch Size: {args.layer_batch_size}")
    print(f"Special: lm_head split into {args.lmhead_chunks} chunks")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Mistral/Llama fix: Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load calibration data
    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        calib_texts = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calib_texts = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)

    # Initialize pruner
    pruner = WandaPruner(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sparsity=args.sparsity,
        max_tokens_per_sample=args.max_tokens_per_sample,
        lmhead_chunks=args.lmhead_chunks
    )

    # Batched sequential pruning
    pruner.prune_model_sequential(calib_texts, n_samples=args.n_calib,
                                  layer_batch_size=args.layer_batch_size)

    # Save model
    print(f"\nSaving pruned model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("PRUNING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
