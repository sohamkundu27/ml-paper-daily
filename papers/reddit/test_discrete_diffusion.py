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


if __name__ == "__main__":
    test_forward_process_basic()
    test_transition_matrices_valid()
    test_convergence_to_uniform()
    test_alpha_schedule()
    test_forward_batch()
    test_early_vs_late_noise()
    print("\n✓ All tests passed!")
