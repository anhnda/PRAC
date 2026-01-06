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
        print("\n" + "="*50 + "\nSEQUENTIAL PRUNING START\n" + "="*50)
        
        transformer_layers = self.get_transformer_layers()
        input_ids_list, attention_mask_list = self.prepare_calibration_inputs(calibration_data, n_samples)

        # ------------------------------------------------------------------
        # STEP 1: FIND ROTARY EMBEDDING MODULE (CRITICAL)
        # ------------------------------------------------------------------
        rotary_emb = None
        
        # Check Location 1: Model level (Llama-3 style)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'rotary_emb'):
            rotary_emb = self.model.model.rotary_emb
        
        # Check Location 2: First layer attention (Mistral/Llama-2 style)
        elif hasattr(transformer_layers[0], 'self_attn') and hasattr(transformer_layers[0].self_attn, 'rotary_emb'):
            rotary_emb = transformer_layers[0].self_attn.rotary_emb
            
        if rotary_emb is None:
            raise ValueError(
                "❌ CRITICAL ERROR: Could not find 'rotary_emb' module.\n"
                "This script requires access to RoPE to calculate position embeddings manually.\n"
                "Please check if 'transformer_layers[0].self_attn.rotary_emb' exists."
            )
        
        print(f"✅ Found Rotary Embedding Module: {type(rotary_emb)}")

        # ------------------------------------------------------------------
        # STEP 2: PRE-COMPUTE EMBEDDINGS
        # ------------------------------------------------------------------
        print("Computing initial embeddings...")
        embed_layer = self.model.model.embed_tokens
        
        embeddings_list = []
        position_ids_list = []
        
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            # Embeddings
            emb = embed_layer(input_ids.to(self.device)).detach().cpu()
            embeddings_list.append(emb)
            
            # Position IDs (handling padding)
            mask_dev = attention_mask.to(self.device)
            pos_ids = mask_dev.long().cumsum(-1) - 1
            pos_ids.masked_fill_(mask_dev == 0, 1)
            position_ids_list.append(pos_ids.detach().cpu())

        # ------------------------------------------------------------------
        # STEP 3: LAYER LOOP
        # ------------------------------------------------------------------
        for i, layer_module in enumerate(tqdm(transformer_layers, desc="Layers")):
            all_linear_inputs = {}
            next_layer_inputs = []

            for hidden_states, attention_mask, pos_ids in zip(embeddings_list, attention_mask_list, position_ids_list):
                
                # 1. Generate RoPE for this batch
                # MistralRotaryEmbedding expects (x, seq_len)
                seq_len = hidden_states.shape[1]
                
                # Create dummy tensor on device to generate cos/sin
                dummy = hidden_states.to(self.device)
                with torch.no_grad():
                    cos, sin = rotary_emb(dummy, seq_len=seq_len)
                    position_embeddings = (cos, sin)
                
                # 2. Run Layer
                linear_inputs, output = self.get_layer_inputs(
                    layer_module, 
                    hidden_states, 
                    attention_mask, 
                    position_ids=pos_ids,
                    position_embeddings=position_embeddings
                )

                # Accumulate
                for name, data in linear_inputs.items():
                    if name not in all_linear_inputs: all_linear_inputs[name] = []
                    all_linear_inputs[name].extend(data)
                
                next_layer_inputs.append(output.detach().cpu())
                del dummy

            # 3. Prune this layer
            for name, module in layer_module.named_modules():
                if isinstance(module, nn.Linear) and "lm_head" not in name:
                    self.activation_data[name] = all_linear_inputs.get(name, [])
                    self.prune_layer(name, module)
            
            # Cleanup
            embeddings_list = next_layer_inputs
            self.activation_data = {}
            torch.cuda.empty_cache()
            gc.collect()

        print("Pruning Complete.")
        
        # Save
        os.makedirs("pruned_model", exist_ok=True)
        self.model.save_pretrained("pruned_model")
        self.tokenizer.save_pretrained("pruned_model")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/Mistral-7B-v0.3")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    pruner = WandaPruner(model, tokenizer)
    texts = get_c4_calibration_data(tokenizer, n_samples=128, seqlen=2048)
    pruner.prune_model_sequential(texts, n_samples=128)

if __name__ == "__main__":
    main()