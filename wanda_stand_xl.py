"""
Wanda Pruning Implementation with L2 Norm Salience
ADAPTED FOR: Extra Large Models (XL) - Mistral/Llama Compatible

Key Features:
- Structured magnitude pruning based on activation salience
- Uses L2 norm for activation salience: Score_ij = |W_ij| * ||X_j||_2
- Layer-by-layer sequential processing
- Robust RoPE (Rotary Embedding) handling for Mistral
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
import gc

# Import your calibration utils
from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class WandaPruner:
    def __init__(self, model, tokenizer, device="cuda", sparsity=0.5, max_tokens_per_sample=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sparsity = sparsity
        self.max_tokens_per_sample = max_tokens_per_sample
        self.activation_data = {}
        self.layer_stats = {}

        print(f"\n[Wanda Pruner Initialized]")
        print(f"  Target sparsity: {sparsity*100:.1f}%")
        print(f"  Device: {device}")

    @torch.no_grad()
    def get_activation_salience_l2(self, name):
        if name not in self.activation_data or len(self.activation_data[name]) == 0:
            return None
        
        # Stack on CPU to avoid OOM
        X_list = self.activation_data[name]
        in_features = X_list[0].shape[-1]
        squared_sum = torch.zeros(in_features)

        for x in X_list:
            x_flat = x.reshape(-1, x.shape[-1]).float()
            squared_sum += x_flat.pow(2).sum(dim=0)

        return torch.sqrt(squared_sum)

    @torch.no_grad()
    def prune_weights_wanda(self, W, salience, sparsity):
        out_features, in_features = W.shape
        scores = W.abs() * salience.unsqueeze(0)
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
        activation_salience = self.get_activation_salience_l2(name)
        if activation_salience is None:
            return

        W = module.weight.data
        original_dtype = W.dtype

        # Prune
        W_pruned, actual_sparsity = self.prune_weights_wanda(
            W.to(self.device), 
            activation_salience.to(self.device), 
            self.sparsity
        )
        module.weight.data = W_pruned.to(original_dtype)

        self.layer_stats[name] = {'actual_sparsity': actual_sparsity}

    def get_transformer_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        else:
            raise ValueError("Could not find transformer layers.")

    def prepare_calibration_inputs(self, calibration_data, n_samples=500):
        print(f"\nTokenizing {n_samples} samples...")
        input_ids_list = []
        attention_mask_list = []
        for text in tqdm(calibration_data[:n_samples]):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids_list.append(inputs['input_ids'])
            attention_mask_list.append(inputs['attention_mask'])
        return input_ids_list, attention_mask_list

    def get_layer_inputs(self, layer_module, inputs, attention_mask, position_ids=None, position_embeddings=None):
        """
        Run forward pass of a single layer and capture inputs to linear layers.
        Handles the tricky argument passing for Mistral/Llama.
        """
        linear_inputs = {}

        def capture_hook(name):
            def hook(module, inp, output):
                if isinstance(inp, tuple): inp = inp[0]
                if name not in linear_inputs: linear_inputs[name] = []
                linear_inputs[name].append(inp.detach().cpu().float().clone())
            return hook

        handles = []
        for name, module in layer_module.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(capture_hook(name)))

        # Move to device
        inputs = inputs.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device).to(inputs.dtype)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)
        
        # Prepare RoPE tuple on device
        if position_embeddings is not None:
            cos, sin = position_embeddings
            position_embeddings = (cos.to(self.device), sin.to(self.device))

        # Build kwargs dynamically
        kwargs = {'attention_mask': attention_mask}
        if position_ids is not None:
            kwargs['position_ids'] = position_ids
        if position_embeddings is not None:
            kwargs['position_embeddings'] = position_embeddings

        with torch.no_grad():
            try:
                outputs = layer_module(inputs, **kwargs)
            except TypeError as e:
                # If explicit position_embeddings fails, try without
                # (But your error suggests it IS required, so this fallback likely won't trigger)
                if 'position_embeddings' in kwargs:
                    del kwargs['position_embeddings']
                outputs = layer_module(inputs, **kwargs)

        if isinstance(outputs, tuple): outputs = outputs[0]

        for h in handles: h.remove()
        return linear_inputs, outputs

    def prune_model_sequential(self, calibration_data, n_samples=500):
        """
        ORIGINAL WANDA APPROACH:
        1. Collect activations from UNPRUNED model (one full pass)
        2. Prune all layers based on collected activations

        This prevents error accumulation from pruned layers.
        """
        print("\n" + "="*50 + "\nWANDA PRUNING (Original Paper Method)\n" + "="*50)

        transformer_layers = self.get_transformer_layers()
        input_ids_list, attention_mask_list = self.prepare_calibration_inputs(calibration_data, n_samples)

        # ------------------------------------------------------------------
        # STEP 1: COLLECT ALL ACTIVATIONS (UNPRUNED MODEL)
        # ------------------------------------------------------------------
        print("\nCollecting activations from unpruned model...")
        all_activations = {}  # {layer_name: [activation tensors]}

        def register_hooks_for_all_layers():
            """Register hooks on ALL linear layers at once"""
            hooks = []
            for layer_idx, layer_module in enumerate(transformer_layers):
                for sublayer_name, module in layer_module.named_modules():
                    if isinstance(module, nn.Linear):
                        full_name = f"layer_{layer_idx}.{sublayer_name}"

                        def make_hook(name):
                            def hook(m, inp, out):
                                if isinstance(inp, tuple):
                                    inp = inp[0]
                                if name not in all_activations:
                                    all_activations[name] = []
                                # Store on CPU immediately to save VRAM
                                all_activations[name].append(inp.detach().cpu().float().clone())
                            return hook

                        handle = module.register_forward_hook(make_hook(full_name))
                        hooks.append(handle)
            return hooks

        # Register all hooks
        hooks = register_hooks_for_all_layers()

        # Run calibration samples through UNPRUNED model
        print(f"Running {len(input_ids_list)} calibration samples through model...")
        with torch.no_grad():
            for idx, (input_ids, attention_mask) in enumerate(tqdm(zip(input_ids_list, attention_mask_list),
                                                                     total=len(input_ids_list),
                                                                     desc="Calibration")):
                inputs = {
                    'input_ids': input_ids.to(self.device),
                    'attention_mask': attention_mask.to(self.device)
                }
                _ = self.model(**inputs)

                # Aggressive cleanup every 10 samples
                if (idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()

        # Remove all hooks
        for handle in hooks:
            handle.remove()

        print(f"✅ Collected activations for {len(all_activations)} linear layers")

        # ------------------------------------------------------------------
        # STEP 2: PRUNE ALL LAYERS
        # ------------------------------------------------------------------
        print("\nPruning layers based on collected activations...")
        pruned_count = 0

        for layer_idx, layer_module in enumerate(tqdm(transformer_layers, desc="Pruning")):
            for sublayer_name, module in layer_module.named_modules():
                if isinstance(module, nn.Linear):
                    full_name = f"layer_{layer_idx}.{sublayer_name}"

                    # Skip lm_head
                    if "lm_head" in sublayer_name.lower():
                        continue

                    # Get activations for this layer
                    if full_name in all_activations:
                        self.activation_data[full_name] = all_activations[full_name]

                        # Prune
                        debug = (pruned_count < 2)
                        self.prune_layer(full_name, module, debug=debug)
                        pruned_count += 1

                        # Free memory
                        del all_activations[full_name]

            # Cleanup after each layer
            torch.cuda.empty_cache()
            gc.collect()

        print(f"✅ Pruned {pruned_count} layers")

        # Final cleanup
        self.activation_data = {}
        all_activations.clear()
        torch.cuda.empty_cache()
        gc.collect()

        print("Pruning Complete.")

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