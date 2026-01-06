# AWQ-GQA Integration Summary

## Overview
`awq_gqa_xl.py` combines AWQ quantization (from `awq_js_xl.py`) with GQA-specific ReFlip refinement (from `fast_quantize_qkv.py`) to provide enhanced quantization for models with Group-Query Attention.

## Pipeline

### Step 1: AWQ Quantization (All Layers)
- Apply James-Stein Heuristic AWQ to all linear layers
- Use L2 salience metric for importance
- Apply heuristic-guided global greedy rounding
- **For GQA layers**: Preserve additional artifacts for refinement

### Step 2: GQA ReFlip Refinement (Attention Layers Only)
- Detect Q_proj, K_proj, V_proj layers
- Apply attention-score-aware error correction
- Use ReFlip strategy to minimize attention score errors
- Update only the Query projections with refined weights

## Key Implementation Details

### AWQGQAQuantizer Class Extensions

**New Storage:**
```python
self.gqa_original_weights = {}  # Original FP weights before quantization
self.gqa_quant_artifacts = {}   # INT4, scales, zp from AWQ
self.gqa_activations = {}       # Preserved activations for ReFlip
```

**Overridden Methods:**

1. **`quantize_weight_heuristic_groupwise_extended`**
   - Returns INT4 representation in addition to dequantized weights
   - Enables preservation of quantization artifacts for GQA layers

2. **`quantize_layer`**
   - Saves original weights before quantization (GQA layers only)
   - Uses extended quantization method for GQA layers
   - Stores INT4, scales, and zero points

3. **`calibrate_layer_batch`**
   - Preserves activations for GQA layers
   - Prevents deletion of activation data needed for ReFlip

4. **`quantize_model_sequential`**
   - Calls parent AWQ quantization
   - Applies GQA ReFlip refinement if enabled

**New Methods:**

5. **`apply_gqa_reflip_refinement`**
   - Groups attention layers (Q, K, V projections)
   - Iterates through attention blocks
   - Calls ReFlip refinement for each group

6. **`refine_attention_group`**
   - Prepares data: activations, original weights, AWQ weights
   - Re-quantizes AWQ output to get clean INT4 representation
   - Reshapes 2D weights to multi-head 3D format
   - Calls `quantize_qkv_reflip_fast` from fast_quantize_qkv.py
   - Updates model weights with refined Query projections

## Usage

### Basic Usage
```bash
python awq_gqa_xl.py \
    --model-path openbmb/MiniCPM-2B-sft-bf16 \
    --output-dir ./quantized_models/minicpm_awq_gqa \
    --n-calib 128 \
    --apply-gqa-reflip
```

### With Custom ReFlip Parameters
```bash
python awq_gqa_xl.py \
    --model-path openbmb/MiniCPM-2B-sft-bf16 \
    --output-dir ./quantized_models/minicpm_awq_gqa \
    --n-calib 128 \
    --apply-gqa-reflip \
    --gqa-critical-dim-pct 0.15 \
    --gqa-max-flip-pct 0.05
```

### AWQ Only (No ReFlip)
```bash
python awq_gqa_xl.py \
    --model-path openbmb/MiniCPM-2B-sft-bf16 \
    --output-dir ./quantized_models/minicpm_awq \
    --n-calib 128
```

## Parameters

### GQA ReFlip Parameters
- `--apply-gqa-reflip`: Enable ReFlip refinement (default: False)
- `--gqa-critical-dim-pct`: Percentage of moderate dimensions for error redistribution (default: 0.15)
- `--gqa-max-flip-pct`: Maximum percentage of weights to flip per dimension (default: 0.05)

### AWQ Parameters
- `--model-path`: Model name or path
- `--output-dir`: Output directory for quantized model
- `--n-calib`: Number of calibration samples (default: 128)
- `--n-grid`: Grid search points for alpha (default: 20)
- `--group-size`: Group size for quantization (default: 128)
- `--layer-batch-size`: Layers per batch (default: 50)
- `--use-heuristic`: Enable heuristic flip correction (default: True)
- `--calib-dataset`: Calibration dataset (choices: c4, wikitext2, wikitext2-simple; default: c4)

## Architecture Notes

### Weight Shape Handling
- **Model Storage**: Weights stored as 2D `[out_features, in_features]`
- **ReFlip Expects**: 3D format `[num_heads, head_dim, hidden_dim]`
- **Current Implementation**: Treats entire weight matrix as single head (simplified)
- **Future Enhancement**: Parse model config to detect actual num_heads and reshape properly

### GQA Layer Detection
Layers detected as GQA if name contains:
- `q_proj`, `k_proj`, `v_proj`
- `query`, `key`, `value`

Common patterns:
- `model.layers.0.self_attn.q_proj`
- `model.layers.0.self_attn.k_proj`
- `model.layers.0.self_attn.v_proj`

### ReFlip Integration Strategy

**Why Re-quantize AWQ Output?**
AWQ stores dequantized weights in the model. To apply ReFlip:
1. AWQ quantizes: `FP → INT4 → FP (scaled)`
2. For ReFlip, we need INT4 representation
3. Solution: Re-quantize the AWQ output to get clean INT4
4. Apply ReFlip corrections to INT4 values
5. Dequantize and store back to model

**Alternative Considered:**
Preserve AWQ's actual INT4 values during quantization. However, AWQ applies additional scaling (`W * awq_scales`) before quantization, so the INT4 values represent scaled weights. Re-quantizing the final output ensures we're working with the actual model weights.

## Expected Benefits

1. **AWQ Benefits** (All Layers):
   - L2 salience-based importance weighting
   - Optimal per-input-channel scaling (α search)
   - Heuristic-guided rounding correction
   - ~14-28GB memory usage with batched sequential

2. **ReFlip Benefits** (GQA Layers):
   - Attention-score-aware error correction
   - Redistributes errors to moderate dimensions
   - Greedy flip selection with automatic stopping
   - Typical improvements: 14-86% better attention scores

3. **Combined Benefits**:
   - AWQ handles general weight quantization efficiently
   - ReFlip specifically targets attention degradation
   - Complementary: AWQ for capacity, ReFlip for attention quality

## Output

Quantized model saved to `--output-dir`:
```
./quantized_models/minicpm_awq_gqa/
├── config.json
├── pytorch_model.bin      # Dequantized FP16/BF16 weights
└── tokenizer files
```

**Note**: Weights stored in dequantized format for research purposes. For deployment, would need INT4 storage format (e.g., AutoAWQ's gemm kernels).

## Limitations & Future Work

1. **Single-Head Treatment**: Current implementation treats entire weight matrix as one head
   - **Fix**: Parse `model.config.num_attention_heads` and reshape properly
   - Impact: Would enable per-head error correction

2. **GQA vs MHA**: Doesn't distinguish between GQA (grouped K/V) and MHA (full K/V)
   - **Fix**: Check `num_key_value_heads` from config
   - Impact: Would enable proper GQA handling (Llama 3+)

3. **V_proj Not Refined**: Currently only refines Q_proj
   - **Consideration**: V_proj affects output but not attention scores
   - Could extend to refine V_proj based on output error

4. **Memory Overhead**: Preserves activations and original weights for GQA layers
   - **Optimization**: Could process attention groups immediately after quantization
   - Would reduce peak memory usage

## Testing

After quantization, evaluate with:
```bash
python compare_awq_heuristic.py \
    --model-awq ./quantized_models/minicpm_awq \
    --model-heuristic ./quantized_models/minicpm_awq_gqa \
    --n-samples 2000
```

Compare perplexity across:
- WikiText-2 test (in-distribution)
- C4 validation (cross-dataset)
- AG News test (cross-dataset)
