"""Tests for categorical discrete diffusion forward and reverse process."""

import numpy as np
import torch
from discrete_diffusion import CategoricalDiffusion, SimpleDenoiser


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


def test_simple_denoiser_forward():
    """Test that SimpleDenoiser forward pass works."""
    num_classes = 5
    seq_len = 4
    batch_size = 2

    denoiser = SimpleDenoiser(num_classes=num_classes, seq_len=seq_len, hidden_dim=32)

    x_t = torch.randint(0, num_classes, (batch_size, seq_len))
    t = torch.tensor([10, 20])

    logits = denoiser(x_t, t)

    # Check output shape
    assert logits.shape == (batch_size, seq_len, num_classes)
    # Check logits are finite
    assert torch.all(torch.isfinite(logits))
    print("✓ test_simple_denoiser_forward passed")


def test_simple_denoiser_with_scalar_timestep():
    """Test denoiser with scalar timestep input."""
    num_classes = 4
    seq_len = 3
    batch_size = 2

    denoiser = SimpleDenoiser(num_classes=num_classes, seq_len=seq_len)

    x_t = torch.randint(0, num_classes, (batch_size, seq_len))
    t = torch.tensor(15)

    logits = denoiser(x_t, t)

    assert logits.shape == (batch_size, seq_len, num_classes)
    assert torch.all(torch.isfinite(logits))
    print("✓ test_simple_denoiser_with_scalar_timestep passed")


def test_reverse_kernel_basic():
    """Test reverse kernel produces valid samples."""
    num_classes = 5
    num_steps = 20
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    x_t = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int32)
    x_0_pred = np.array([[1, 2, 0], [4, 0, 1]], dtype=np.int32)  # Some predictions

    x_t_minus_1 = diffusion.reverse_kernel(x_t, t=10, x_0_pred=x_0_pred)

    # Check shape preserved
    assert x_t_minus_1.shape == x_t.shape
    # Check all values are valid class indices
    assert np.all((x_t_minus_1 >= 0) & (x_t_minus_1 < num_classes))
    print("✓ test_reverse_kernel_basic passed")


def test_reverse_kernel_at_low_t():
    """Test reverse kernel near end of diffusion process."""
    num_classes = 4
    num_steps = 50
    diffusion = CategoricalDiffusion(num_classes, num_steps)

    # At high timestep (close to full noise), reverse should have more entropy
    x_t = np.random.randint(0, num_classes, (10, 5), dtype=np.int32)
    x_0_pred = np.random.randint(0, num_classes, (10, 5), dtype=np.int32)

    x_t_minus_1 = diffusion.reverse_kernel(x_t, t=45, x_0_pred=x_0_pred)

    assert x_t_minus_1.shape == x_t.shape
    assert np.all((x_t_minus_1 >= 0) & (x_t_minus_1 < num_classes))
    print("✓ test_reverse_kernel_at_low_t passed")


def test_full_sampling_loop():
    """Test complete reverse sampling loop from noised to clean sample."""
    num_classes = 4
    seq_len = 5
    num_steps = 15
    batch_size = 2

    diffusion = CategoricalDiffusion(num_classes, num_steps)

    # Create a simple denoiser
    denoiser = SimpleDenoiser(num_classes=num_classes, seq_len=seq_len, hidden_dim=32)

    # Random noisy sample to start
    x_T = np.random.randint(0, num_classes, (batch_size, seq_len), dtype=np.int32)

    # Run sampling
    x_0_sampled = diffusion.sample_with_rehash_sampler(denoiser, x_T, device="cpu")

    # Check output is valid
    assert x_0_sampled.shape == x_T.shape
    assert np.all((x_0_sampled >= 0) & (x_0_sampled < num_classes))
    print("✓ test_full_sampling_loop passed")


def test_synthetic_classification_task():
    """Test end-to-end diffusion on simple synthetic data.

    Scenario: We have K simple patterns (one-hot like for K categories).
    We corrupt them to noise, then try to recover with denoiser.
    """
    num_classes = 5
    seq_len = 6
    num_steps = 20
    batch_size = 8

    diffusion = CategoricalDiffusion(num_classes, num_steps)
    denoiser = SimpleDenoiser(num_classes=num_classes, seq_len=seq_len, hidden_dim=48)

    # Create synthetic "clean" data: random category labels
    x_0_clean = np.random.randint(0, num_classes, (batch_size, seq_len), dtype=np.int32)

    # Forward diffusion: corrupt to high noise
    x_T, _ = diffusion.forward_with_rehash(x_0_clean, t=num_steps - 1, num_corrupts=seq_len)

    # Reverse: recover from noise
    x_0_recovered = diffusion.sample_with_rehash_sampler(denoiser, x_T, device="cpu")

    # Sanity checks:
    # 1. Recovered should be valid tokens
    assert x_0_recovered.shape == x_0_clean.shape
    assert np.all((x_0_recovered >= 0) & (x_0_recovered < num_classes))

    # 2. Recovered should NOT be identical to original (untrained network gives random predictions)
    # But it should at least be in valid range, which we already checked
    print(f"✓ test_synthetic_classification_task passed")
    print(f"  Original shape: {x_0_clean.shape}")
    print(f"  Recovered shape: {x_0_recovered.shape}")
    print(f"  Sample original: {x_0_clean[0]}")
    print(f"  Sample recovered: {x_0_recovered[0]}")


def test_denoise_step_output():
    """Test the internal _denoise_step function."""
    num_classes = 5
    seq_len = 4
    num_steps = 15
    batch_size = 2

    diffusion = CategoricalDiffusion(num_classes, num_steps)
    denoiser = SimpleDenoiser(num_classes=num_classes, seq_len=seq_len, hidden_dim=32)

    x_t = np.random.randint(0, num_classes, (batch_size, seq_len), dtype=np.int32)
    t = 10

    x_0_pred = diffusion._denoise_step(denoiser, x_t, t, device="cpu")

    # Check output
    assert x_0_pred.shape == x_t.shape
    assert x_0_pred.dtype == np.int32
    assert np.all((x_0_pred >= 0) & (x_0_pred < num_classes))
    print("✓ test_denoise_step_output passed")


if __name__ == "__main__":
    # Pass 1 & 2 tests (forward process and rehashing)
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

    # Pass 3 tests (reverse process and denoiser)
    test_simple_denoiser_forward()
    test_simple_denoiser_with_scalar_timestep()
    test_reverse_kernel_basic()
    test_reverse_kernel_at_low_t()
    test_denoise_step_output()
    test_full_sampling_loop()
    test_synthetic_classification_task()

    print("\n✓ All tests passed!")
