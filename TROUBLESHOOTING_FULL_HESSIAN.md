# Troubleshooting Full Hessian Correction

## Problem: Perplexity Increasing After Correction

If you see perplexity **increasing** (5.89 → 5.90 → 5.92), the correction is making things worse. This can happen for several reasons.

---

## Root Causes & Solutions

### 1. **TOP Weights May Be Too Stable** ⭐ MOST LIKELY

**Symptoms:**
- Small error values (~1e-6)
- Low clamping percentage (<1%)
- Small correction magnitudes
- Perplexity consistently increases

**Why:** TOP weights are already optimally positioned for the pruned network. Modifying them disturbs a stable equilibrium.

**Solution:** Use MODERATE selection instead:
```bash
python prac_full.py --selection-strategy moderate --sparsity 0.5
```

### 2. **Clamping Too Aggressive**

**Symptoms:**
- High clamping percentage (>20%)
- Large correction magnitudes before clamping
- Error doesn't decrease much

**Why:** The 5% magnitude limit prevents the Hessian solution from reaching the optimal correction.

**Solutions:**
```bash
# Increase magnitude limit to 10%
python prac_full.py --max-correction-magnitude 0.10

# Or remove clamping by setting to very high value
python prac_full.py --max-correction-magnitude 1.0
```

### 3. **Damping Too Small**

**Symptoms:**
- Solve failures (uses lstsq fallback)
- Very large correction magnitudes
- Numerical instability

**Why:** Hessian is nearly singular, needs more regularization.

**Solutions:**
```bash
# Increase damping from 1e-5 to 1e-4
python prac_full.py --damping 1e-4

# Or even more conservative
python prac_full.py --damping 1e-3
```

### 4. **Correcting Too Few Weights**

**Symptoms:**
- Only 5% of weights corrected
- Error reduction exists but small
- Perplexity slightly worse

**Why:** Not enough degrees of freedom to meaningfully fix the error.

**Solutions:**
```bash
# Increase correction percentage to 10%
python prac_full.py --percent-change 0.10

# Or 20% for aggressive correction
python prac_full.py --percent-change 0.20
```

---

## Recommended Experiments

### Experiment 1: MODERATE Selection (Default Now)
```bash
python prac_full.py \
  --sparsity 0.5 \
  --selection-strategy moderate \
  --percent-change 0.05 \
  --damping 1e-5 \
  --max-correction-magnitude 0.05
```

**Expected:** Better than TOP selection, as it avoids disturbing the most important weights.

### Experiment 2: More Conservative (Higher Damping)
```bash
python prac_full.py \
  --sparsity 0.5 \
  --selection-strategy moderate \
  --percent-change 0.05 \
  --damping 1e-4 \
  --max-correction-magnitude 0.10
```

**Expected:** Smaller corrections, more stable, less likely to harm perplexity.

### Experiment 3: Correct More Weights
```bash
python prac_full.py \
  --sparsity 0.5 \
  --selection-strategy moderate \
  --percent-change 0.15 \
  --damping 1e-5 \
  --max-correction-magnitude 0.10
```

**Expected:** More aggressive correction, may help or hurt depending on whether the Hessian solution is accurate.

### Experiment 4: No Correction Baseline
```bash
python prac_full.py \
  --sparsity 0.5 \
  --no-correction
```

**Expected:** This gives you the baseline Wanda performance without any correction.

---

## Interpreting Debug Output

```
Error before (full H): 0.000003
Error after (full H): 0.000004
Reduction: -0.000001 (-33.33%)
Correction magnitude: mean=0.000001, max=0.000005
Clamped: 3749/928778 (0.4%)
```

### Good Signs:
- ✅ **Positive reduction** (error after < error before)
- ✅ **Reasonable clamping** (5-20%)
- ✅ **No solve failures**
- ✅ **Moderate correction magnitudes** (not too large)

### Bad Signs:
- ❌ **Negative reduction** (error increases)
- ❌ **Very high clamping** (>50%) - solution is constrained
- ❌ **Zero reduction** - corrections having no effect
- ❌ **Solve failures** - numerical issues

---

## Why Full Hessian Might Not Always Help

### Theory vs Practice Gap

**Theory Says:**
- Full Hessian captures feature correlations
- Optimal second-order correction
- Should always be better than diagonal

**Practice Says:**
- **Assumption violations:** Hessian estimated from limited calibration data (128 samples)
- **Stability issues:** Correcting high-salience weights can destabilize the network
- **Local optima:** Pruned network may have found a different local minimum
- **Magnitude limits:** Clamping prevents reaching the theoretical optimum

### When Diagonal Works Better

Diagonal approximation may be more robust when:
1. Calibration data is limited
2. Features are approximately independent (low correlation)
3. Small corrections are needed (first-order is sufficient)
4. Network has already converged to a stable state

### When Full Hessian Works Better

Full Hessian helps when:
1. Strong feature correlations exist
2. Enough calibration data (>500 samples ideally)
3. Correcting many weights (>10%)
4. Pruning is aggressive (>60% sparsity)

---

## Comparison Matrix

| Configuration | Strategy | Damping | Magnitude Limit | When to Use |
|--------------|----------|---------|-----------------|-------------|
| Conservative | moderate | 1e-4 | 0.05 | Default, safest |
| Standard | moderate | 1e-5 | 0.10 | Good middle ground |
| Aggressive | moderate | 1e-5 | 0.20 | High sparsity (>60%) |
| Experimental | top | 1e-5 | 0.10 | For comparison only |

---

## Quick Diagnostic Checklist

1. **Is error reducing?**
   - Yes → Good! Check if perplexity improves
   - No → Try moderate selection or higher damping

2. **Is clamping high (>20%)?**
   - Yes → Increase max_correction_magnitude
   - No → Keep as is

3. **Any solve failures?**
   - Yes → Increase damping (1e-4 or 1e-3)
   - No → Keep as is

4. **Perplexity still worse?**
   - Compare with --no-correction baseline
   - Try diagonal method (prac_xl.py) instead
   - Consider that full Hessian may not help for this model/sparsity

---

## Fallback: Use Diagonal Method

If Full Hessian consistently hurts perplexity:

```bash
# Use the diagonal approximation instead
python prac_xl.py --sparsity 0.5 --percent-change 0.05
```

The diagonal method:
- ✅ More stable
- ✅ Less sensitive to hyperparameters
- ✅ Faster (16 layers/batch vs 4)
- ❌ Ignores feature correlations
- ❌ Less optimal theoretically

---

## Summary

**Start with this:**
```bash
python prac_full.py \
  --sparsity 0.5 \
  --selection-strategy moderate \
  --percent-change 0.05 \
  --damping 1e-5 \
  --max-correction-magnitude 0.05
```

**If perplexity increases, try:**
1. Increase damping to 1e-4
2. Increase magnitude limit to 0.10
3. Compare with --no-correction baseline
4. Fall back to diagonal method (prac_xl.py)

The fact that correction can make things worse is **not a bug** - it's a reminder that theoretical optimality doesn't always translate to practical improvement, especially with approximate Hessians and limited calibration data.
