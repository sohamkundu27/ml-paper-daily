"""Tests for categorical discrete diffusion forward process."""

import numpy as np
from discrete_diffusion import CategoricalDiffusion


def test_forward_process_basic():
    """Test that forward process produces valid samples."""
    num_classes = 5
    num_steps = 10
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    x0 = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int32)
    x_t = diffusion.forward(x0, t=5)

    # Check shape is preserved
    assert x_t.shape == x0.shape
    # Check all values are valid class indices
    assert np.all((x_t >= 0) & (x_t < num_classes))
    print("✓ test_forward_process_basic passed")


def test_transition_matrices_valid():
    """Test that transition matrices are valid probability distributions."""
    num_classes = 4
    num_steps = 20
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    # Each row should sum to 1 (columns for each x_0 class)
    for t in range(num_steps + 1):
        Q_t = diffusion.q_matrix[t]
        row_sums = np.sum(Q_t, axis=0)  # Sum over x_t dimension
        np.testing.assert_allclose(
            row_sums,
            np.ones(num_classes),
            rtol=1e-6,
            err_msg=f"Transition matrix at t={t} does not form valid distribution"
        )
    print("✓ test_transition_matrices_valid passed")


def test_convergence_to_uniform():
    """Test that late timesteps converge to uniform distribution."""
    num_classes = 6
    num_steps = 100
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    # At final step, all classes should have nearly equal probability
    Q_final = diffusion.q_matrix[-1]

    # Each column (for any x_0) should be approximately uniform
    for cls in range(num_classes):
        probs = Q_final[:, cls]
        expected_uniform = np.ones(num_classes) / num_classes
        np.testing.assert_allclose(
            probs,
            expected_uniform,
            atol=0.01,
            err_msg=f"Class {cls} distribution not uniform at final step"
        )
    print("✓ test_convergence_to_uniform passed")


def test_alpha_schedule():
    """Test that alpha schedule decreases monotonically."""
    num_classes = 3
    num_steps = 50
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    alpha = diffusion.alpha
    # Should decrease (or stay same) from 1.0 to nearly 0
    assert alpha[0] == 1.0
    assert alpha[-1] < 0.01
    assert np.all(np.diff(alpha) <= 0), "Alpha schedule should be non-increasing"
    print("✓ test_alpha_schedule passed")


def test_forward_batch():
    """Test batch forward sampling with different timesteps."""
    num_classes = 4
    num_steps = 15
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    batch_size = 3
    seq_len = 4
    x0 = np.random.randint(0, num_classes, (batch_size, seq_len), dtype=np.int32)
    timesteps = np.array([0, 7, 15], dtype=np.int32)

    x_t = diffusion.forward_batch(x0, timesteps)

    assert x_t.shape == x0.shape
    assert np.all((x_t >= 0) & (x_t < num_classes))
    print("✓ test_forward_batch passed")


def test_early_vs_late_noise():
    """Test that later timesteps have higher probability of being non-zero."""
    num_classes = 10
    num_steps = 50
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    x0 = np.zeros((1000, 10), dtype=np.int32)  # All zeros

    # Sample at early and late timesteps
    x_t_early = diffusion.forward(x0, t=5)
    x_t_late = diffusion.forward(x0, t=45)

    # At early timestep, should stay mostly in class 0
    early_prob_zero = np.mean(x_t_early == 0)

    # At late timestep, should be closer to uniform
    late_prob_zero = np.mean(x_t_late == 0)

    assert early_prob_zero > late_prob_zero, "Early timesteps should have more mass at original class"
    print(f"✓ test_early_vs_late_noise passed (P(x_t=0|x_0=0): early={early_prob_zero:.3f}, late={late_prob_zero:.3f})")


def test_forward_with_rehash_basic():
    """Test that rehashing produces valid samples with corruption mask."""
    num_classes = 5
    num_steps = 10
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    x0 = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int32)
    x_t, mask = diffusion.forward_with_rehash(x0, t=5, num_corrupts=1)

    # Check shape is preserved
    assert x_t.shape == x0.shape
    assert mask.shape == x0.shape
    # Check all values are valid class indices
    assert np.all((x_t >= 0) & (x_t < num_classes))
    # Check mask is boolean
    assert mask.dtype == bool
    # Check at least one position was corrupted (with high probability)
    assert np.sum(mask) >= 1, "At least one position should be corrupted"
    print("✓ test_forward_with_rehash_basic passed")


def test_forward_with_rehash_num_corrupts():
    """Test that num_corrupts parameter controls corruption amount."""
    num_classes = 6
    num_steps = 20
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    x0 = np.array([[0, 1, 2, 3, 4]] * 10, dtype=np.int32)  # All same

    # Test with different num_corrupts values
    for num_corrupts in [0, 1, 2, 3]:
        x_t, mask = diffusion.forward_with_rehash(x0, t=10, num_corrupts=num_corrupts)

        # Count total corruptions
        corrupted_count = np.sum(mask)

        # Should have exactly num_corrupts corruptions per sample
        expected_total = num_corrupts * x0.shape[0]
        assert corrupted_count == expected_total, \
            f"Expected {expected_total} corruptions, got {corrupted_count}"

    print("✓ test_forward_with_rehash_num_corrupts passed")


def test_forward_batch_with_rehash():
    """Test batch forward sampling with rehashing and different timesteps."""
    num_classes = 4
    num_steps = 15
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    batch_size = 5
    seq_len = 6
    x0 = np.random.randint(0, num_classes, (batch_size, seq_len), dtype=np.int32)
    timesteps = np.array([0, 3, 7, 10, 15], dtype=np.int32)
    num_corrupts = 2

    x_t, masks = diffusion.forward_batch_with_rehash(x0, timesteps, num_corrupts)

    assert x_t.shape == x0.shape
    assert masks.shape == x0.shape
    assert np.all((x_t >= 0) & (x_t < num_classes))
    assert masks.dtype == bool
    # Each sample should have exactly num_corrupts corruptions
    for i in range(batch_size):
        assert np.sum(masks[i]) == num_corrupts
    print("✓ test_forward_batch_with_rehash passed")


def test_rehash_path_diversity():
    """Test that rehashing creates diverse paths through noise space."""
    num_classes = 5
    num_steps = 50
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    # Generate multiple rehashed samples from same x0
    x0 = np.zeros((1, 10), dtype=np.int32)
    num_samples = 100
    num_corrupts = 1

    # Collect samples at mid-timestep
    samples = []
    for _ in range(num_samples):
        x_t, _ = diffusion.forward_with_rehash(x0, t=25, num_corrupts=num_corrupts)
        samples.append(x_t.flatten())

    samples = np.array(samples)

    # Compute entropy of each position across samples
    # Higher entropy = more diversity
    entropies = []
    for pos in range(samples.shape[1]):
        unique, counts = np.unique(samples[:, pos], return_counts=True)
        probs = counts / len(samples)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    mean_entropy = np.mean(entropies)

    # At a moderate timestep, we should have non-zero entropy (diversity)
    # but not maximum (still somewhat constrained to x0 = 0)
    assert mean_entropy > 0.1, f"Expected diversity, got mean entropy {mean_entropy}"
    assert mean_entropy < np.log(num_classes), \
        f"Entropy should be less than log(num_classes)={np.log(num_classes)}"
    print(f"✓ test_rehash_path_diversity passed (mean entropy: {mean_entropy:.3f})")


def test_rehash_unmasked_positions_unchanged():
    """Test that unmasked positions remain unchanged from x0."""
    num_classes = 5
    num_steps = 10
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    x0 = np.array([[1, 2, 3, 4]], dtype=np.int32)
    x_t, mask = diffusion.forward_with_rehash(x0, t=5, num_corrupts=1)

    # Check that unmasked positions are unchanged
    unmasked = ~mask
    np.testing.assert_array_equal(
        x_t[unmasked],
        x0[unmasked],
        err_msg="Unmasked positions should remain unchanged from x0"
    )
    print("✓ test_rehash_unmasked_positions_unchanged passed")


if __name__ == "__main__":
    test_forward_process_basic()
    test_transition_matrices_valid()
    test_convergence_to_uniform()
    test_alpha_schedule()
    test_forward_batch()
    test_early_vs_late_noise()
    test_forward_with_rehash_basic()
    test_forward_with_rehash_num_corrupts()
    test_forward_batch_with_rehash()
    test_rehash_path_diversity()
    test_rehash_unmasked_positions_unchanged()
    print("\n✓ All tests passed!")
