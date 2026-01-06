# AWQ_SH Batched Sequential Quantization

This document describes the batched sequential quantization implementation in `awq_sh.py`.

## What Changed

The `awq_sh.py` script now uses the same batched sequential quantization strategy as `gw_awq_asym_l2.py` for optimal memory/speed balance.

### Key Changes

1. **Added `layer_batch_size` parameter** (default: 50)
   - Controls how many layers are processed simultaneously
   - Memory usage: `batch_size × 280 MB`
   - Example: 50 layers = ~14 GB

2. **New Methods Added:**
   - `get_hook(name)` - Creates hooks for individual layers
   - `calibrate_layer_batch(...)` - Calibrates a batch of layers
   - `quantize_model_sequential(...)` - Main batched sequential quantization

3. **Dtype Preservation:**
   - `quantize_layer()` now preserves original dtype (critical for sequential processing)
   - Prevents float32/float16 mismatch errors during forward passes

4. **Updated Calibration Data:**
   - Changed from `seqlen=2048` to `seqlen=512` (4x memory reduction)
   - Added `wikitext2-simple` option for variable-length sequences

5. **Main Flow Changed:**
   - **Old:** `calibrate(all_layers)` → `quantize_model()`
   - **New:** `quantize_model_sequential(calibration_data)`

## Usage

### Recommended (Default Settings)

```bash
python awq_sh.py \
  --calib-dataset wikitext2-simple \
  --n-calib 128 \
  --layer-batch-size 50
```

**Expected:**
- Memory: ~14 GB
- Time: ~6-12 minutes
- Batches: 6 (281 layers ÷ 50 per batch)

### For Limited Memory (8GB systems)

```bash
python awq_sh.py \
  --calib-dataset wikitext2-simple \
  --n-calib 128 \
  --layer-batch-size 20
```

**Expected:**
- Memory: ~6 GB
- Time: ~10-15 minutes
- Batches: 15 (281 layers ÷ 20 per batch)

### For High Memory Systems (64GB+)

```bash
python awq_sh.py \
  --calib-dataset c4 \
  --n-calib 256 \
  --layer-batch-size 100
```

**Expected:**
- Memory: ~28 GB
- Time: ~4-8 minutes
- Batches: 3 (281 layers ÷ 100 per batch)

### With Heuristic Disabled (Standard AWQ)

```bash
python awq_sh.py \
  --calib-dataset wikitext2-simple \
  --n-calib 128 \
  --layer-batch-size 50 \
  --no-heuristic
```

This disables the heuristic rounding, making it equivalent to standard group-wise AWQ.

## Memory Formula

```
Memory ≈ model_size + (batch_size × 280 MB)

Example with batch_size=50:
= 5 GB (model) + (50 × 0.28 GB)
= 5 GB + 14 GB
= ~19 GB peak (during calibration)
```

After each batch is quantized and cleared, memory drops back to ~8 GB.

## Speed Formula

```
Calibration runs = ceil(num_layers / batch_size)

Example with 281 layers, batch_size=50:
= ceil(281 / 50)
= 6 calibration runs

Total time ≈ 6 runs × 1-2 min/run = 6-12 minutes
```

## Expected Output

```
[Standard Heuristic AWQ Quantizer Initialized]
  Target bits: 4
  Group size: 128
  Token subsampling: 512 tokens/sample (memory optimization)
  Layer batch size: 50 (batched sequential quantization)
  Use heuristic: True
  Outlier protection: Top 5.0% ignored
  Quantization: HEURISTIC-GUIDED GROUP-WISE ASYMMETRIC [0, 15]

================================================================================
Batched Sequential Quantization
================================================================================
  Strategy: Process 50 layers per batch
  Memory: ~14.0 GB per batch
  Total layers: 281
  Total batches: 6
================================================================================

[Batch 1/6] Layers 0-49
  Calibrating 50 layers...
  Calibration: 100%|████████████████████| 128/128 [01:23<00:00,  1.54it/s]
  Quantizing 50 layers...
  Quantization: 100%|██████████████████| 50/50 [00:15<00:00,  3.21layer/s]
  [Batch 1/6] GPU: 5.23GB allocated, 14.12GB reserved

[Batch 2/6] Layers 50-99
  Calibrating 50 layers...
  Calibration: 100%|████████████████████| 128/128 [01:22<00:00,  1.55it/s]
  Quantizing 50 layers...
  Quantization: 100%|██████████████████| 50/50 [00:14<00:00,  3.42layer/s]
  [Batch 2/6] GPU: 5.18GB allocated, 14.10GB reserved

...

[Batch 6/6] Layers 250-280
  Calibrating 31 layers...
  Calibration: 100%|████████████████████| 128/128 [01:20<00:00,  1.59it/s]
  Quantizing 31 layers...
  Quantization: 100%|██████████████████| 31/31 [00:09<00:00,  3.28layer/s]
  [Batch 6/6] GPU: 5.21GB allocated, 14.15GB reserved

================================================================================
✓ Batched Sequential Quantization Complete
================================================================================
```

## Comparison with Old Approach

| Metric | Old Batch | New Batched Sequential |
|--------|-----------|----------------------|
| Memory | 75 GB (OOM) | 14 GB ✅ |
| Speed | 2 min (if not OOM) | 6-12 min ✅ |
| Max Samples | 16 (OOM above) | 128+ ✅ |
| Quality | Good | Better (error propagation aware) |

## Differences from gw_awq_asym_l2.py

While both use the same batched sequential strategy, `awq_sh.py` includes:

1. **Heuristic rounding** (can be disabled with `--no-heuristic`)
2. **Outlier masking** (controlled via `--outlier-percent`)
3. **Float16 precision** (vs bfloat16 in gw_awq_asym_l2.py)
4. **Random sampling** for grid search (vs sequential)

## Troubleshooting

### Still getting OOM?

**Solutions:**
1. Reduce batch size: `--layer-batch-size 20` (or 10, or even 1)
2. Use wikitext2-simple: `--calib-dataset wikitext2-simple`
3. Reduce samples: `--n-calib 64`

### Seeing all α=0.0?

This indicates a dtype mismatch bug. The fix has been applied:
- `quantize_layer()` now preserves `original_dtype`
- Should not occur with current implementation

### Want to use old batch mode?

Not recommended (will OOM), but you can:
```python
quantizer.calibrate(calib_texts, n_samples=args.n_calib)
quantizer.quantize_model()
```

Replace `quantize_model_sequential()` call in main() with the above.

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-calib` | 128 | Number of calibration samples |
| `--n-grid` | 20 | Grid search points for α |
| `--group-size` | 128 | Quantization group size |
| `--use-heuristic` | True | Enable heuristic rounding |
| `--no-heuristic` | - | Disable heuristic (standard AWQ) |
| `--outlier-percent` | 0.05 | Top X% activations to ignore |
| `--max-tokens-per-sample` | 512 | Token subsampling for memory |
| `--layer-batch-size` | 50 | Layers per batch (memory control) |
| `--calib-dataset` | c4 | Calibration dataset |
| `--output-dir` | ./quantized_models/minicpm_awq_sh | Output directory |
| `--seed` | 42 | Random seed |

## Recommended Configurations

### Production Quality (30GB RAM)
```bash
python awq_sh.py \
  --calib-dataset c4 \
  --n-calib 128 \
  --layer-batch-size 50
```

### Fast Iteration (16GB RAM)
```bash
python awq_sh.py \
  --calib-dataset wikitext2-simple \
  --n-calib 64 \
  --layer-batch-size 30
```

### Minimal Memory (8GB RAM)
```bash
python awq_sh.py \
  --calib-dataset wikitext2-simple \
  --n-calib 64 \
  --layer-batch-size 10
```

### Maximum Quality (64GB RAM)
```bash
python awq_sh.py \
  --calib-dataset c4 \
  --n-calib 256 \
  --layer-batch-size 100
```

---

**Updated:** 2025-12-13
