"""
Test script to demonstrate why larger magnitude can lead to smaller total reduction.

This creates a simple synthetic example to show the effect.
"""

import torch

def test_greedy_correction():
    """Demonstrate magnitude vs. total reduction trade-off."""

    # Synthetic setup
    torch.manual_seed(42)
    num_tokens = 1000
    num_channels = 100
    num_positions = 20

    # Create synthetic activations
    X = torch.randn(num_tokens, num_positions)

    # Create synthetic weights (original and pruned)
    W_orig = torch.randn(num_channels, num_positions)
    W_pruned = W_orig.clone()
    # Simulate pruning: zero out some weights
    mask = torch.rand(num_channels, num_positions) < 0.5
    W_pruned[mask] = 0

    # Initial residual
    R_init = X @ (W_pruned - W_orig).t()
    error_init = (R_init ** 2).sum().item()

    print("="*80)
    print("Testing Greedy Correction with Different Magnitudes")
    print("="*80)
    print(f"Initial error: {error_init:.6e}\n")

    # Test different magnitudes
    for magnitude in [0.005, 0.01, 0.02, 0.04]:
        print(f"\n{'='*80}")
        print(f"Magnitude: {magnitude*100:.1f}%")
        print(f"{'='*80}")

        W_current = W_pruned.clone()
        R = R_init.clone()
        error_current = error_init

        # Precompute column norms
        X_col_norms_sq = (X ** 2).sum(dim=0)

        positions_applied = 0
        total_reduction = 0

        # Greedy loop over positions
        for j in range(num_positions):
            X_j = X[:, j]
            X_j_norm_sq = X_col_norms_sq[j]

            # Compute dot products
            dot_products = R.t() @ X_j  # [num_channels]

            # Compute deltas
            deltas = magnitude * W_orig[:, j]

            # Error changes
            delta_error_pos = 2 * deltas * dot_products + deltas ** 2 * X_j_norm_sq
            delta_error_neg = -2 * deltas * dot_products + deltas ** 2 * X_j_norm_sq

            # Which channels improve?
            pos_improves = delta_error_pos < 0
            neg_improves = (delta_error_neg < 0) & ~pos_improves

            # Total improvement
            improvement_pos = -delta_error_pos[pos_improves].sum().item() if pos_improves.any() else 0
            improvement_neg = -delta_error_neg[neg_improves].sum().item() if neg_improves.any() else 0
            total_improvement = improvement_pos + improvement_neg

            if total_improvement > 0:
                # Apply corrections
                if pos_improves.any():
                    W_current[pos_improves, j] += deltas[pos_improves]
                    R[:, pos_improves] += X_j.unsqueeze(1) * deltas[pos_improves].unsqueeze(0)

                if neg_improves.any():
                    W_current[neg_improves, j] -= deltas[neg_improves]
                    R[:, neg_improves] -= X_j.unsqueeze(1) * deltas[neg_improves].unsqueeze(0)

                error_current -= total_improvement
                positions_applied += 1
                total_reduction += total_improvement

                if positions_applied <= 3:
                    # Compute linear vs quad terms
                    linear_pos = (2 * deltas[pos_improves] * dot_products[pos_improves]).abs().mean().item() if pos_improves.any() else 0
                    linear_neg = (2 * deltas[neg_improves] * dot_products[neg_improves]).abs().mean().item() if neg_improves.any() else 0
                    quad = (deltas[pos_improves | neg_improves] ** 2).mean().item() * X_j_norm_sq.item() if (pos_improves | neg_improves).any() else 0
                    linear_avg = (linear_pos + linear_neg) / 2

                    print(f"  Position {j+1}: improvement={total_improvement:.6e}, "
                          f"linear={linear_avg:.6e}, quad={quad:.6e}, ratio={quad/linear_avg:.3f}")
            else:
                print(f"  Position {j+1}: EARLY STOP (no improvement)")
                break

        print(f"\nResults for magnitude={magnitude*100:.1f}%:")
        print(f"  Positions applied: {positions_applied}/{num_positions}")
        print(f"  Total reduction: {total_reduction:.6e}")
        print(f"  Final error: {error_current:.6e}")
        print(f"  Reduction %: {total_reduction/error_init*100:.2f}%")

        # Verify
        actual_error = (R ** 2).sum().item()
        print(f"  Verification: tracked={error_current:.6e}, actual={actual_error:.6e}, "
              f"diff={abs(actual_error - error_current):.6e}")

if __name__ == "__main__":
    test_greedy_correction()
