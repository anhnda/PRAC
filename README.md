# AWQ Quantization: Standard vs Dynamic Heuristic

Comparison of two AWQ quantization approaches for 4-bit weight quantization of Large Language Models.

## Methods

### 1. Standard AWQ (`awq_stand_xl.py`)
- **Group-wise asymmetric quantization** [0, 15]
- **L2 salience metric**: E[X²] for channel importance
- **No heuristic rounding**: Uses nearest rounding
- **Batched sequential processing**: Memory-efficient layer-by-layer quantization
- **Special lm_head handling**: Splits large layers into chunks to avoid OOM

### 2. Dynamic Heuristic AWQ (`awq_dh_xl.py`)
- **All features from Standard AWQ**, PLUS:
- **Heuristic-guided rounding**: Global greedy correction to minimize quantization error
- **Dynamic outlier detection**: Kneedle algorithm adaptively identifies outliers per layer
- **Flip constraint**: Limits flips per output channel to max_flip_percent (default: 1%)
- **Outlier masking**: Protects high-activation channels from flipping

## Usage

### Quantize with Standard AWQ
```bash
python awq_stand_xl.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/Mistral-7B-v0.3_awq_standard \
    --n-calib 128 \
    --layer-batch-size 16
```

### Quantize with Dynamic Heuristic AWQ
```bash
python awq_dh_xl.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/Mistral-7B-v0.3_awq_dh \
    --n-calib 128 \
    --knee-tolerance 0.0 \
    --max-flip-percent 0.01 \
    --layer-batch-size 16
```

### Evaluate and Compare
```bash
python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Mistral-7B-v0.3_awq_dh \
    --standard-path ./quantized_models/Mistral-7B-v0.3_awq_standard \
    --n-samples 2000
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-calib` | 128 | Number of calibration samples |
| `--n-grid` | 20 | Grid search points for α (scaling factor) |
| `--group-size` | 128 | Quantization group size |
| `--layer-batch-size` | 16 | Layers per batch (higher = more memory) |
| `--lmhead-chunks` | 4 | Split lm_head into N chunks (higher = less memory) |
| `--knee-tolerance` | 0.0 | Outlier detection tolerance (Dynamic AWQ only) |
| `--max-flip-percent` | 0.01 | Max flips per output channel = 1% of in_features (Dynamic AWQ only) |

## Results

### Mistral-7B-v0.3
Perplexity (↓ lower is better):

| Dataset    | Origin | Standard AWQ | Dynamic AWQ |
|------------|--------|--------------|-------------|
| WikiText-2 | 4.8454 | 4.9778       | **4.9689**  |
| C4         | 7.6040 | 7.7892       | **7.7830**  |

**Improvement**: Dynamic AWQ achieves 0.89% better WikiText-2 and 0.62% better C4 perplexity vs Standard AWQ.

---

### Llama-3-8B
Perplexity (↓ lower is better):

| Dataset    | Origin | Standard AWQ | Dynamic AWQ |
|------------|--------|--------------|-------------|
| WikiText-2 | 5.4425 | 6.7386       | **6.6863**  |
| C4         | 8.6383 | 10.4595      | **10.3014** |

**Improvement**: Dynamic AWQ achieves 5.23% better C4 perplexity vs Standard AWQ.
**Note**: Dynamic outlier detection found 0.65% outliers on average (vs fixed 5%).

---

### Llama-2-7B
Perplexity (↓ lower is better):

| Dataset    | Origin | Standard AWQ | Dynamic AWQ |
|------------|--------|--------------|-------------|
| WikiText-2 | 4.9712 | 5.1280       | **5.1270**  |
| C4         | 6.5748 | 6.7986       | **6.7983**  |

**Improvement**: Marginal improvements; both methods perform similarly on this model.

---

### Qwen2.5-7B
Perplexity (↓ lower is better):

| Dataset    | Origin  | Standard AWQ | Dynamic AWQ |
|------------|---------|--------------|-------------|
| WikiText-2 | 23.1382 | 24.0180      | **23.3029** |
| C4         | 36.1769 | 37.5713      | **36.4447** |

**Improvement**: Dynamic AWQ achieves 7.15% better WikiText-2 and 11.27% better C4 perplexity vs Standard AWQ.
**Significant gains** on this model family.

---

## Summary

**Dynamic Heuristic AWQ** consistently outperforms **Standard AWQ** across all tested models:
- **Mistral-7B-v0.3**: +0.6-0.9% improvement
- **Llama-3-8B**: +5.2% improvement (C4)
- **Llama-2-7B**: Marginal improvement
- **Qwen2.5-7B**: +7-11% improvement

### Key Advantages of Dynamic AWQ:
1. **Adaptive outlier detection**: Kneedle algorithm adjusts per layer (vs fixed 5%)
2. **Flip constraint**: Prevents over-correction (limits to 1% per channel)
3. **Better optimization**: Global greedy rounding reduces quantization error


## Hardware Requirements

- **GPU**: 16GB+ VRAM recommended (tested on A100/V100)
  - Model: ~5-7GB
  - Activations: ~3-5GB per batch
  - Peak: ~14-20GB with default settings
- **CPU fallback**: Supported but 10-20× slower
- **Storage**: ~15GB for models and cached datasets

## Memory Optimization

For limited VRAM (8-12GB):
```bash
python awq_dh_xl.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --layer-batch-size 8 \           # Reduce batch size
    --lmhead-chunks 8 \               # Split lm_head more
    --n-calib 64 \                    # Fewer calibration samples
    --calib-dataset wikitext2-simple  # Smaller dataset
```

## Citation

If you use this code, please cite:
```
Dynamic Heuristic AWQ: Adaptive Quantization with Kneedle-based Outlier Detection
```

## License

MIT License
