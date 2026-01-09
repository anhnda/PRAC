# Fix Summary: prac_ho.py L2 Salience Computation

## Problem
`prac_ho.py` and `prac_full.py` produced **different results** even when `--no-correction` was used, due to different L2 salience computations.

### Root Cause
- **prac_ho.py**: Computed L2 salience from only **first 8,192 stored tokens** (limited by `max_activation_samples`)
- **prac_full.py**: Computed L2 salience from **ALL tokens** (~262K with default settings)

This caused different Wanda scores → different pruned weights → different models.

## Solution
Modified `prac_ho.py` to compute L2 salience from ALL tokens while keeping memory-efficient correction:

### Changes Made

#### 1. Updated Hook to Accumulate Squared Sum
```python
# NEW: Accumulate squared_sum for ALL tokens (no limit)
stats['squared_sum'] += (inp_flat ** 2).sum(dim=0)

# Keep limited samples for correction phase only
if stats['num_stored'] < self.max_activation_samples:
    stats['activations'].append(inp_flat[:num_to_store].cpu())
```

#### 2. Updated L2 Salience Computation
```python
# OLD (WRONG): Only used first 8192 stored tokens
l2_salience = torch.sqrt((activations ** 2).mean(dim=0))

# NEW (CORRECT): Uses ALL tokens via accumulated squared_sum
l2_salience = torch.sqrt(stats['squared_sum'] / stats['count'])
```

#### 3. Updated Storage Structure
```python
self.activation_stats[name] = {
    'activations': [],        # LIMITED samples for correction
    'squared_sum': tensor,    # ALL tokens for L2 salience
    'mean_sum': tensor,       # ALL tokens for JS estimator
    'count': int,
    'num_stored': int
}
```

## Benefits

### 1. Consistency
- ✅ `prac_ho.py --no-correction` now produces **identical results** to `prac_full.py --no-correction`
- ✅ Both implementations use the same Wanda pruning baseline

### 2. Memory Efficiency
- ✅ L2 salience: O(hidden_dim) memory - uses ALL tokens via incremental accumulation
- ✅ Correction: O(max_activation_samples × hidden_dim) - limited for memory efficiency
- ✅ Much better than full Hessian: O(hidden_dim²)

### 3. Accuracy
- ✅ L2 salience now based on ALL calibration data (not just 3%)
- ✅ More accurate weight importance estimates
- ✅ Better pruning decisions

## Verification

Run `verify_fix.py` to confirm the fix:
```bash
python verify_fix.py
```

All tests pass:
- ✅ L2 salience computation matches prac_full.py (within floating point precision)
- ✅ Wanda pruning produces consistent results
- ✅ Incremental accumulation matches batch computation

## Impact on Users

### Before Fix
```bash
# These produced DIFFERENT models
python prac_ho.py --no-correction --sparsity 0.5
python prac_full.py --no-correction --sparsity 0.5
```

### After Fix
```bash
# These now produce IDENTICAL models
python prac_ho.py --no-correction --sparsity 0.5
python prac_full.py --no-correction --sparsity 0.5
```

## Technical Details

### L2 Salience Computation

Both methods are mathematically equivalent:

**prac_ho.py (new)**:
```
l2_salience[j] = sqrt(sum_i(X[i,j]²) / N)
```

**prac_full.py**:
```
Hessian = X^T @ X
l2_salience[j] = sqrt(diag(Hessian)[j] / N)
                = sqrt(sum_i(X[i,j]²) / N)
```

The tiny difference (2.6e-6) in verification is just floating point rounding from different computation orders - perfectly acceptable!

### Memory Comparison

For hidden_dim = 4096, with 262K tokens:

| Method | L2 Salience Memory | Correction Memory | Total |
|--------|-------------------|-------------------|-------|
| prac_full.py | 64 MB (Hessian) | 64 MB | 128 MB |
| prac_ho.py (old) | 128 MB (stored) | 128 MB | 256 MB |
| prac_ho.py (new) | 16 KB (squared_sum) | 128 MB | 128 MB |

The fix makes `prac_ho.py` as memory-efficient as `prac_full.py` for L2 salience!

## Files Modified

1. **prac_ho.py**:
   - Updated `__init__` docstring
   - Updated `get_activation_stats()` to use `squared_sum`
   - Updated `get_hook()` to accumulate `squared_sum`
   - Updated `prune_layer()` to handle None activations
   - Updated all docstrings and comments

2. **MISMATCH_REPORT_no_correction.md**: Created to document the original problem

3. **verify_fix.py**: Created to verify the fix works correctly

4. **FIX_SUMMARY.md**: This document

## Recommendations

### For Users
- Re-run any experiments that used `prac_ho.py --no-correction` to get more accurate baselines
- The new version should give slightly better results (more accurate L2 salience)

### For Developers
- Always verify that `--no-correction` produces identical results across implementations
- Use incremental accumulation for statistics when possible (memory efficient)
- Test with verification scripts before deploying

## Conclusion

The fix ensures `prac_ho.py` and `prac_full.py` are consistent when `--no-correction` is used, while maintaining memory efficiency. This was a critical bug fix that improves both accuracy and consistency of the implementation.
