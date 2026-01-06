import torch
import numpy as np
import pandas as pd


def quantize_nearest_greedy_global(x, w, n_bits=4, outlier_percent=0.05):
    """Previous method: Single scale for the entire vector."""
    d = x.shape[0]
    max_val = w.abs().max()
    if max_val == 0: return w

    q_max = 2 ** (n_bits - 1) - 1
    scale = max_val / q_max

    # 1. Baseline
    w_div = w / scale
    w_int = torch.round(w_div).clamp(-2 ** (n_bits - 1), 2 ** (n_bits - 1) - 1)
    w_quant = w_int * scale

    # 2. Error & Candidates
    current_error = torch.dot(x, w - w_quant)
    if abs(current_error) < 1e-6: return w_quant

    target_sign = torch.sign(current_error)
    flip_dir = torch.sign(w_div - w_int);
    flip_dir[flip_dir == 0] = 1.0
    flip_impacts = x * flip_dir * scale

    # Filter Outliers
    k_outliers = int(d * outlier_percent)
    outlier_mask = torch.zeros(d, dtype=torch.bool, device=x.device)
    if k_outliers > 0:
        outlier_mask[torch.topk(x.abs(), k_outliers).indices] = True

    valid_mask = (~outlier_mask) & (torch.sign(flip_impacts) == target_sign)

    # Check bounds for proposed flip
    w_int_prop = w_int + flip_dir
    in_range = (w_int_prop >= -2 ** (n_bits - 1)) & (w_int_prop <= 2 ** (n_bits - 1) - 1)
    valid_mask = valid_mask & in_range

    if not valid_mask.any(): return w_quant

    # 3. Sort & Optimize
    valid_idx = torch.nonzero(valid_mask).squeeze()
    if valid_idx.ndim == 0: valid_idx = valid_idx.unsqueeze(0)

    candidate_costs = (w_div - w_int).abs()[valid_idx]  # Priority: closest to boundary
    sorted_indices = torch.argsort(candidate_costs, descending=True)
    sorted_impacts = flip_impacts[valid_idx][sorted_indices]

    cumsum = torch.cumsum(sorted_impacts, dim=0)
    best_k = torch.argmin(torch.abs(current_error - cumsum))

    if best_k == 0: return w_quant

    indices_to_flip = valid_idx[sorted_indices][:best_k]
    w_int[indices_to_flip] += flip_dir[indices_to_flip].long()

    return w_int * scale


def quantize_groupwise_global_greedy(x, w, group_size=128, n_bits=4, outlier_percent=0.05):
    """
    1. Calculate scales per group.
    2. Quantize per group.
    3. Collect flip candidates GLOBALLY (across all groups).
    4. Sort globally by cost and correct the TOTAL error.
    """
    d = x.shape[0]
    assert d % group_size == 0, f"Dimension {d} not divisible by group_size {group_size}"
    num_groups = d // group_size

    # Reshape to (num_groups, group_size)
    w_groups = w.view(num_groups, group_size)
    x_groups = x.view(num_groups, group_size)

    # --- 1. Group-wise Setup ---
    q_max = 2 ** (n_bits - 1) - 1

    # Max per group (keepdims=True for broadcasting)
    max_vals = w_groups.abs().max(dim=1, keepdim=True).values
    max_vals[max_vals == 0] = 1e-9  # Avoid div/0
    scales = max_vals / q_max  # Shape: (num_groups, 1)

    # Broadcast scales to full shape
    scales_expanded = scales.repeat(1, group_size).view(d)  # Flatten back to D

    # --- 2. Initial Group-wise Rounding ---
    w_div = w / scales_expanded
    w_int = torch.round(w_div).clamp(-2 ** (n_bits - 1), 2 ** (n_bits - 1) - 1)
    w_quant = w_int * scales_expanded

    # --- 3. Global Error Calculation ---
    # We look at the SINGLE scalar error of the dot product
    current_error = torch.dot(x, w - w_quant)

    if abs(current_error) < 1e-6: return w_quant

    # --- 4. Global Candidate Selection ---
    target_sign = torch.sign(current_error)

    # Directions
    flip_dir = torch.sign(w_div - w_int)
    flip_dir[flip_dir == 0] = 1.0

    # Impact: x * sign * specific_group_scale
    flip_impacts = x * flip_dir * scales_expanded

    # Outlier Mask (Global logic applied to whole vector)
    k_outliers = int(d * outlier_percent)
    outlier_mask = torch.zeros(d, dtype=torch.bool, device=x.device)
    if k_outliers > 0:
        outlier_mask[torch.topk(x.abs(), k_outliers).indices] = True

    # Validity Mask
    valid_mask = (~outlier_mask) & (torch.sign(flip_impacts) == target_sign)

    # Range check
    w_int_prop = w_int + flip_dir
    in_range = (w_int_prop >= -2 ** (n_bits - 1)) & (w_int_prop <= 2 ** (n_bits - 1) - 1)
    valid_mask = valid_mask & in_range

    if not valid_mask.any(): return w_quant

    # --- 5. Global Sorting & Optimization ---
    valid_indices = torch.nonzero(valid_mask).squeeze()
    if valid_indices.ndim == 0: valid_indices = valid_indices.unsqueeze(0)

    # Cost is distance to boundary
    candidate_costs = (w_div - w_int).abs()[valid_indices]

    # Sort ALL candidates from ALL groups together
    sorted_indices = torch.argsort(candidate_costs, descending=True)
    sorted_impacts = flip_impacts[valid_indices][sorted_indices]

    # Find best K
    cumsum = torch.cumsum(sorted_impacts, dim=0)
    best_k = torch.argmin(torch.abs(current_error - cumsum))

    if best_k == 0: return w_quant

    # Apply Flips
    indices_to_flip = valid_indices[sorted_indices][:best_k]
    w_int[indices_to_flip] += flip_dir[indices_to_flip].long()

    return w_int * scales_expanded


def load_and_compare(csv_path='data.csv', group_size=128):
    try:
        df = pd.read_csv(csv_path, sep=',')
        df.columns = [c.strip() for c in df.columns]

        if 'E[Xs]' not in df.columns or 'W_scaled' not in df.columns:
            print("Columns missing.")
            return

        x = torch.from_numpy(df['E[Xs]'].values).float()
        w = torch.from_numpy(df['W_scaled'].values).float()

        # Handle size divisibility
        orig_d = x.shape[0]
        remainder = orig_d % group_size
        if remainder != 0:
            pad_len = group_size - remainder
            x = torch.cat([x, torch.zeros(pad_len)])
            w = torch.cat([w, torch.zeros(pad_len)])
            print(f"Warning: Padded vector from {orig_d} to {x.shape[0]} to fit group_size {group_size}")

        print(f"Data Loaded. D={x.shape[0]}. Group Size={group_size}")

        # 1. Non-Group-Wise Proposed
        w_global_prop = quantize_nearest_greedy_global(x, w)
        err_global_prop = torch.dot(x, w - w_global_prop).item()

        # 2. Group-Wise Nearest (Baseline)
        # Re-calc manually for clarity
        w_groups = w.view(-1, group_size)
        scales = w_groups.abs().max(dim=1, keepdim=True).values / 7.0
        scales[scales == 0] = 1e-9
        w_gw_int = torch.round(w_groups / scales).clamp(-8, 7)
        w_gw_nearest = (w_gw_int * scales).view(-1)
        err_gw_nearest = torch.dot(x, w - w_gw_nearest).item()

        # 3. Group-Wise Proposed (Global TopK)
        w_gw_prop = quantize_groupwise_global_greedy(x, w, group_size=group_size)
        err_gw_prop = torch.dot(x, w - w_gw_prop).item()

        print("\n--- RESULTS (Absolute Dot Product Error) ---")
        print(f"1. Non-Group Global Proposed:  {abs(err_global_prop):.6f}")
        print(f"2. Group-Wise Nearest (Base):  {abs(err_gw_nearest):.6f}")
        print(f"3. Group-Wise Proposed (New):  {abs(err_gw_prop):.6f}")

        best_err = min(abs(err_global_prop), abs(err_gw_nearest), abs(err_gw_prop))

        print("\n--- COMPARISON ---")
        if abs(err_gw_prop) < abs(err_gw_nearest):
            print(f"New Method vs GW Nearest: {abs(err_gw_nearest) / abs(err_gw_prop):.2f}x improvement")

        if abs(err_gw_prop) < abs(err_global_prop):
            print(f"New Method vs Non-Group:  {abs(err_global_prop) / abs(err_gw_prop):.2f}x improvement")
        else:
            print(
                f"New Method vs Non-Group:  {abs(err_gw_prop) / abs(err_global_prop):.2f}x worse (Group constraints limit flexibility)")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    load_and_compare('data10.csv', group_size=128)