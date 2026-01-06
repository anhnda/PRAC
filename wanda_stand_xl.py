"""
Wanda Pruning Implementation with L2 Norm Salience
ADAPTED FOR: Extra Large Models (XL) - Special handling for large layers like lm_head

Key Features:
- Structured magnitude pruning based on activation salience
- Uses L2 norm for activation salience
- Scoring: Score_ij = |W_ij| * ||X_j||_2
- Per-row (output channel) top-p% selection
- Layer-by-layer processing with output propagation

Algorithm (per layer):
1. Run forward pass through current layer (unpruned) - capture inputs
2. Compute per-input-channel salience: s[j] = ||X_j||_2 (L2 norm across all tokens)
3. Score each weight: Score_ij = |W_ij| * s[j]
4. For each output channel (row i):
   - Keep top p% highest scoring weights
   - Zero out the remaining weights
5. Run forward pass through pruned layer - get outputs
6. Outputs become inputs for next layer

Special Handling:
- lm_head layer is NOT pruned (keeps original weights)
- This preserves output quality and vocabulary distribution
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
    Wanda Pruning with L2 Norm Salience.
    Special handling for large layers (lm_head).
    """

    def __init__(self, model, tokenizer, device="cuda", sparsity=0.5, max_tokens_per_sample=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sparsity = sparsity  # Percentage of weights to PRUNE (e.g., 0.5 means keep 50%, prune 50%)
        self.max_tokens_per_sample = max_tokens_per_sample  # Subsample to save memory

        # Storage for activations
        self.activation_data = {}
        self.hooks = []
        self.layer_stats = {}

        print(f"\n[Wanda Pruner Initialized - XL Version]")
        print(f"  Target sparsity: {sparsity*100:.1f}% (keep {(1-sparsity)*100:.1f}%)")
        print(f"  Token subsampling: {max_tokens_per_sample} tokens/sample (memory optimization)")
        print(f"  Salience metric: ||X_j||_2 (L2 norm across all tokens)")
        print(f"  Scoring: Score_ij = |W_ij| * ||X_j||_2")
        print(f"  Special: lm_head is NOT pruned (keeps original weights)")


    @torch.no_grad()
    def get_activation_salience_l2(self, name):
        """
        Compute per-input-channel activation salience using L2 norm: ||X_j||_2

        Where X_j is the j-th input channel across all calibration tokens.
        ||X_j||_2 = sqrt(sum(X_j^2))
        """
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None

        X_list = self.activation_data[name]
        in_features = X_list[0].shape[-1]

        # Accumulate squared values for L2 norm on CPU to save GPU VRAM
        squared_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1])
            # Ensure we're working with float32 for numerical stability
            x_flat = x_flat.float()
            # Accumulate sum of squares
            squared_sum += x_flat.pow(2).sum(dim=0)

        # L2 norm: sqrt of sum of squares
        salience = torch.sqrt(squared_sum)
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

        # VECTORIZED: Get top-k indices for all rows at once
        # scores shape: [out_features, in_features]
        # topk returns: values [out_features, num_to_keep], indices [out_features, num_to_keep]
        if num_to_keep > 0:
            _, top_indices = torch.topk(scores, num_to_keep, dim=1, largest=True)

            # Create mask using scatter: set selected indices to True
            mask = torch.zeros_like(W, dtype=torch.bool)
            mask.scatter_(1, top_indices, True)
        else:
            # If num_to_keep is 0, prune everything
            mask = torch.zeros_like(W, dtype=torch.bool)

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

        # Get L2 norm activation salience
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

    def get_transformer_layers(self):
        """
        Get transformer layers in execution order.
        Works for Llama, Mistral, and similar architectures.
        """
        # Try to find the transformer layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LlamaForCausalLM, MistralForCausalLM
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2, GPT-Neo
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            # GPT-NeoX
            return self.model.gpt_neox.layers
        else:
            raise ValueError("Could not find transformer layers. Model architecture not supported.")

    def prepare_calibration_inputs(self, calibration_data, n_samples=500):
        """
        Tokenize calibration data once and prepare inputs.
        """
        print(f"\nTokenizing {n_samples} calibration samples...")
        input_ids_list = []
        attention_mask_list = []

        for i, text in enumerate(tqdm(calibration_data[:n_samples], desc="Tokenizing")):
            try:
                inputs = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                input_ids_list.append(inputs['input_ids'])
                # Convert attention_mask to float to match model dtype
                attention_mask_list.append(inputs['attention_mask'].float())
            except Exception as e:
                if i == 0:
                    print(f"\n⚠️  Tokenization error: {str(e)[:100]}")
                continue

        print(f"Successfully tokenized {len(input_ids_list)} samples")
        return input_ids_list, attention_mask_list

    def get_layer_inputs(self, layer_module, inputs, attention_mask, position_ids=None, position_embeddings=None):
        """
        Run inputs through a transformer layer and capture the input to linear sublayers.
        Returns the layer's inputs (for pruning) and outputs (for next layer).
        """
        # Storage for capturing inputs to linear layers
        linear_inputs = {}

        def capture_hook(name):
            def hook(module, inp, output):
                if isinstance(inp, tuple):
                    inp_tensor = inp[0]
                else:
                    inp_tensor = inp
                # Store on CPU
                if name not in linear_inputs:
                    linear_inputs[name] = []
                linear_inputs[name].append(inp_tensor.detach().cpu().float().clone())
            return hook

        # Register hooks on all linear layers in this transformer layer
        handles = []
        for name, module in layer_module.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(capture_hook(name))
                handles.append(handle)

        # Run forward pass through this layer
        with torch.no_grad():
            # Ensure all inputs are on the same device and have correct dtype
            inputs = inputs.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                # Convert to match model dtype (bfloat16)
                if attention_mask.dtype != inputs.dtype:
                    attention_mask = attention_mask.to(inputs.dtype)
            if position_ids is not None:
                position_ids = position_ids.to(self.device)

            # Prepare arguments based on what's available
            if position_embeddings is not None:
                # For Mistral/Llama that need pre-computed position_embeddings
                outputs = layer_module(
                    inputs,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings
                )
            elif position_ids is not None:
                # For models that compute embeddings internally from position_ids
                outputs = layer_module(
                    inputs,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
            else:
                # Fallback
                outputs = layer_module(inputs, attention_mask=attention_mask)

            # Extract hidden states (first element of tuple for most models)
            if isinstance(outputs, tuple):
                layer_output = outputs[0]
            else:
                layer_output = outputs

        # Remove hooks
        for handle in handles:
            handle.remove()

        return linear_inputs, layer_output

    def prune_model_sequential(self, calibration_data, n_samples=500):
        """
        LAYER-BY-LAYER SEQUENTIAL PRUNING with output propagation.

        Flow for each layer:
        1. Run forward pass through current layer (unpruned) - capture inputs
        2. Prune linear sublayers based on input activations
        3. Run forward pass through pruned layer - get outputs
        4. Outputs become inputs for next layer
        """
        print("\n" + "=" * 80)
        print("LAYER-BY-LAYER SEQUENTIAL PRUNING (XL Version)")
        print("=" * 80)

        if HAS_PSUTIL:
            initial_ram = psutil.virtual_memory().percent
            print(f"Initial System RAM: {initial_ram:.1f}%")

        # Get transformer layers in execution order
        try:
            transformer_layers = self.get_transformer_layers()
            print(f"\nFound {len(transformer_layers)} transformer layers")
        except ValueError as e:
            print(f"\n❌ Error: {e}")
            return

        # Prepare calibration inputs (tokenize once)
        input_ids_list, attention_mask_list = self.prepare_calibration_inputs(calibration_data, n_samples)

        if len(input_ids_list) == 0:
            print("\n❌ No valid calibration samples!")
            return

        # Get embeddings and setup for RoPE
        print("\nComputing embeddings and setting up RoPE...")
        with torch.no_grad():
            # Get embedding layer and rotary_emb
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                embed_layer = self.model.model.embed_tokens
                # Get rotary_emb - could be at model level or layer level
                if hasattr(self.model.model, 'rotary_emb'):
                    rotary_emb = self.model.model.rotary_emb
                else:
                    # Try to get from first layer
                    try:
                        rotary_emb = transformer_layers[0].self_attn.rotary_emb
                    except:
                        rotary_emb = None
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                embed_layer = self.model.transformer.wte
                rotary_emb = None
            else:
                print("❌ Could not find embedding layer")
                return

            # Compute embeddings and position_ids for all samples
            embeddings_list = []
            position_ids_list = []
            for input_ids, attention_mask in tqdm(zip(input_ids_list, attention_mask_list),
                                                   desc="Embedding", total=len(input_ids_list)):
                input_ids_device = input_ids.to(self.device)
                attention_mask_device = attention_mask.to(self.device)

                # 1. Get initial token embeddings
                emb = embed_layer(input_ids_device)
                embeddings_list.append(emb.detach().cpu())  # Store on CPU to save VRAM

                # 2. Compute correct position_ids (crucial for padded data)
                position_ids = attention_mask_device.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask_device == 0, 1)
                position_ids_list.append(position_ids.detach().cpu())

                del input_ids_device, attention_mask_device

        print(f"Embeddings computed for {len(embeddings_list)} samples")
        print(f"RoPE available: {rotary_emb is not None}")

        # Process each transformer layer sequentially
        pruned_count = 0
        skipped_count = 0

        for layer_idx, layer_module in enumerate(tqdm(transformer_layers, desc="Pruning Layers")):
            print(f"\n{'='*60}")
            print(f"Layer {layer_idx}/{len(transformer_layers)-1}")
            print(f"{'='*60}")

            # Collect inputs for all calibration samples through this layer
            all_linear_inputs = {}
            next_layer_inputs = []

            for sample_idx, (hidden_states, attention_mask, pos_ids) in enumerate(
                zip(embeddings_list, attention_mask_list, position_ids_list)):
                # Move to device
                hidden_states = hidden_states.to(self.device)
                attention_mask = attention_mask.to(self.device)
                pos_ids = pos_ids.to(self.device)

                # Compute position_embeddings for this sample if RoPE is available
                if rotary_emb is not None:
                    # Compute cos/sin embeddings
                    position_embeddings = rotary_emb(hidden_states, pos_ids)
                else:
                    position_embeddings = None

                # Run layer forward pass and capture linear layer inputs
                linear_inputs, layer_output = self.get_layer_inputs(
                    layer_module, hidden_states, attention_mask,
                    position_ids=pos_ids,
                    position_embeddings=position_embeddings
                )

                # Accumulate inputs for each linear sublayer
                for sublayer_name, inp_list in linear_inputs.items():
                    if sublayer_name not in all_linear_inputs:
                        all_linear_inputs[sublayer_name] = []
                    all_linear_inputs[sublayer_name].extend(inp_list)

                # Store output for next layer
                next_layer_inputs.append(layer_output.detach())

                # Cleanup
                del layer_output
                if (sample_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()

            # Prune each linear sublayer in this transformer layer
            for sublayer_name, module in layer_module.named_modules():
                if not isinstance(module, nn.Linear):
                    continue

                # Check if this is lm_head (skip - but lm_head shouldn't be in transformer layers)
                is_lmhead = 'lm_head' in sublayer_name.lower()
                if is_lmhead:
                    print(f"  ⏭️  Skipping {sublayer_name} (keeping original weights)")
                    skipped_count += 1
                    continue

                # Get activations for this sublayer
                if sublayer_name in all_linear_inputs:
                    self.activation_data[sublayer_name] = all_linear_inputs[sublayer_name]

                    # Prune this sublayer
                    debug = (pruned_count < 2)
                    if debug:
                        print(f"\n  DEBUG Sublayer: {sublayer_name}")
                    self.prune_layer(sublayer_name, module, debug=debug)
                    pruned_count += 1

            # Update embeddings_list with outputs from this layer for next iteration
            embeddings_list = next_layer_inputs

            # Cleanup
            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()

            if HAS_PSUTIL:
                ram_pct = psutil.virtual_memory().percent
                print(f"Layer {layer_idx} complete. RAM: {ram_pct:.1f}%")

        # Handle lm_head separately (skip pruning)
        print(f"\n{'='*60}")
        print("Final Layer: lm_head")
        print(f"{'='*60}")
        print("  ⏭️  Skipping lm_head (keeping original weights)")
        skipped_count += 1

        print(f"\n✅ Sequential Pruning Complete!")
        print(f"   Total layers pruned: {pruned_count}")
        print(f"   Skipped layers: {skipped_count}")

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
        description="Wanda Pruning with L2 Norm Salience for XL Models",
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
    print("Wanda Pruning with L2 Norm Salience (XL Version)")
    print(f"Target Model: {model_name}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Target Sparsity: {args.sparsity*100:.1f}%")
    print(f"Processing: Layer-by-layer with output propagation")
    print(f"Special: lm_head is NOT pruned (keeps original weights)")
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
        max_tokens_per_sample=args.max_tokens_per_sample
    )

    # Layer-by-layer sequential pruning
    pruner.prune_model_sequential(calib_texts, n_samples=args.n_calib)

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
