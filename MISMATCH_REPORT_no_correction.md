# Mismatch Report: prac_ho.py vs prac_full.py with --no-correction

## Summary
**When `--no-correction` is set, prac_ho.py and prac_full.py produce DIFFERENT results** due to a discrepancy in how they compute L2 salience for Wanda pruning.

## The Problem

### prac_ho.py Behavior
```python
# 1. Stores activation samples (limited to max_activation_samples=8192 tokens)
if stats['num_stored'] < self.max_activation_samples:
    remaining = self.max_activation_samples - stats['num_stored']
    num_to_store = min(num_tokens, remaining)
    stats['activations'].append(inp_flat[:num_to_store].cpu())
    stats['num_stored'] += num_to_store

# 2. Computes L2 salience from STORED activations only
activations = stats['activations']  # Only first 8192 tokens!
l2_salience = torch.sqrt((activations ** 2).mean(dim=0))
```

**Result**: L2 salience computed from **first 8192 tokens only**

### prac_full.py Behavior
```python
# 1. Accumulates Hessian for ALL tokens (no limit)
stats['hessian'] += torch.matmul(inp_flat.t(), inp_flat)  # ALL tokens!
stats['count'] += num_tokens

# 2. Computes L2 salience from full Hessian diagonal
l2_salience = torch.sqrt(torch.diag(hessian) / stats['count'])
```

**Result**: L2 salience computed from **ALL tokens** (typically ~262K tokens)

## Impact

With typical calibration settings:
- `n_samples` = 128 (or 500)
- `max_tokens_per_sample` = 2048
- **Total tokens processed** = 128 × 2048 = 262,144 tokens

| Implementation | Tokens used for L2 salience | Percentage |
|----------------|----------------------------|------------|
| prac_ho.py     | 8,192                     | 3.1%       |
| prac_full.py   | 262,144                   | 100%       |

### Consequences
1. **Different L2 salience values** → Different Wanda scores
2. **Different weights pruned** → Different pruned models
3. **Different perplexities** even with `--no-correction`

## Root Cause

In `prac_ho.py`, the `max_activation_samples` limit was added to prevent OOM when storing activations for the greedy correction phase. However, this limit also affects the L2 salience computation, which is used for Wanda pruning.

## Why This Matters

When users run:
```bash
python prac_ho.py --no-correction --sparsity 0.5
python prac_full.py --no-correction --sparsity 0.5
```

They would expect **identical results** (both should just do Wanda pruning), but they will get different pruned models!

## Recommended Fix

### Option 1: Store all activations for L2 salience (memory-intensive)
Remove the `max_activation_samples` limit for L2 salience computation:
```python
# Always store for L2 salience
stats['activations'].append(inp_flat.cpu())

# But limit correction samples separately
if stats['num_stored_for_correction'] < self.max_activation_samples:
    stats['correction_activations'].append(inp_flat[:remaining].cpu())
```

### Option 2: Compute L2 salience incrementally (recommended)
Like prac_full.py, accumulate X^T X diagonal incrementally:
```python
# Initialize
if name not in self.activation_stats:
    self.activation_stats[name] = {
        'activations': [],  # For correction only
        'squared_sum': torch.zeros(hidden_dim, dtype=torch.float32, device=inp.device),  # For L2 salience
        'mean_sum': torch.zeros(hidden_dim, dtype=torch.float32, device=inp.device),
        'count': 0,
        'num_stored': 0
    }

# Accumulate for L2 salience (ALL tokens, no limit)
stats['squared_sum'] += (inp_flat ** 2).sum(dim=0)

# Store activations for correction (limited)
if stats['num_stored'] < self.max_activation_samples:
    ...

# Compute L2 salience from accumulated sums
l2_salience = torch.sqrt(stats['squared_sum'] / stats['count'])
```

### Option 3: Document the difference
If the difference is intentional, clearly document that:
- `prac_ho.py` uses fewer samples for efficiency
- Results will differ from `prac_full.py` even with `--no-correction`
- Users should not compare across implementations

## Verification Test

To verify the mismatch:
```bash
# Run both with same settings
python prac_ho.py --no-correction --sparsity 0.5 --seed 42 --output-dir ./test_ho
python prac_full.py --no-correction --sparsity 0.5 --seed 42 --output-dir ./test_full

# Compare weights (should be identical but won't be)
python compare_models.py ./test_ho ./test_full
```

## Recommendation

Implement **Option 2** to ensure both scripts produce identical results when `--no-correction` is used, while maintaining memory efficiency for the correction phase.
