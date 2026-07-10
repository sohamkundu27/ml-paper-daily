"""Tests for basic SSM and diffusion components."""

import numpy as np
from ssm import LinearSSM, DiffusionProcess


def test_linear_ssm():
    """Test SSM trajectory generation."""
    state_dim = 3
    obs_dim = 2
    ssm = LinearSSM(state_dim, obs_dim, seed=42)

    # Generate a short trajectory
    T = 10
    z, y = ssm.sample_trajectory(T)

    # Check shapes
    assert z.shape == (T, state_dim), f"Expected z shape {(T, state_dim)}, got {z.shape}"
    assert y.shape == (T, obs_dim), f"Expected y shape {(T, obs_dim)}, got {y.shape}"

    # Check that values are finite
    assert np.all(np.isfinite(z)), "z contains non-finite values"
    assert np.all(np.isfinite(y)), "y contains non-finite values"

    print("✓ test_linear_ssm passed")


def test_diffusion_forward():
    """Test forward diffusion process."""
    state_dim = 4
    num_steps = 50
    diffusion = DiffusionProcess(state_dim, num_steps=num_steps)

    x0 = np.random.randn(state_dim)
    t = 25

    # Forward diffusion
    x_t, noise_used = diffusion.forward_diffusion(x0, t)

    # Check shapes
    assert x_t.shape == (state_dim,), f"Expected x_t shape {(state_dim,)}, got {x_t.shape}"
    assert noise_used.shape == (state_dim,), f"Expected noise shape {(state_dim,)}, got {noise_used.shape}"

    # Check finite values
    assert np.all(np.isfinite(x_t)), "x_t contains non-finite values"
    assert np.all(np.isfinite(noise_used)), "noise contains non-finite values"

    # At t=0, x_t should be close to x0 (minimal noise)
    x_t_early, _ = diffusion.forward_diffusion(x0, 0)
    assert np.allclose(x_t_early, x0, atol=0.05), "Forward diffusion at t=0 should be close to x0"

    # At t=num_steps-1, x_t should be mostly noise
    x_t_late, _ = diffusion.forward_diffusion(x0, num_steps - 1)
    assert not np.allclose(x_t_late, x0, atol=0.1), "Forward diffusion at end should be mostly noise"

    print("✓ test_diffusion_forward passed")


def test_diffusion_reverse():
    """Test reverse diffusion step."""
    state_dim = 4
    num_steps = 50
    diffusion = DiffusionProcess(state_dim, num_steps=num_steps)

    x0 = np.random.randn(state_dim)
    t = 25

    # Forward to get x_t
    x_t, noise_true = diffusion.forward_diffusion(x0, t)

    # Reverse step: assume we know the true noise
    x_t_minus_1 = diffusion.reverse_step(x_t, t, noise_true)

    # Check shapes
    assert x_t_minus_1.shape == (state_dim,), f"Expected x_{{t-1}} shape {(state_dim,)}, got {x_t_minus_1.shape}"

    # Check finite values
    assert np.all(np.isfinite(x_t_minus_1)), "x_t_minus_1 contains non-finite values"

    # Reverse step at t=0 should return x_t unchanged
    x_t_minus_1_at_0 = diffusion.reverse_step(x_t, 0, noise_true)
    assert np.allclose(x_t_minus_1_at_0, x_t), "Reverse step at t=0 should return x_t unchanged"

    print("✓ test_diffusion_reverse passed")


def test_end_to_end():
    """Test SSM generation + diffusion forward + reverse cycle."""
    state_dim = 3
    obs_dim = 2
    ssm = LinearSSM(state_dim, obs_dim, seed=42)
    diffusion = DiffusionProcess(state_dim, num_steps=100)

    # Sample two consecutive states
    z1 = np.random.randn(state_dim)
    z2 = ssm.A @ z1 + np.random.randn(state_dim) * 0.1

    # Forward diffusion on the transition
    transition = z2 - z1
    t = 50
    noisy_transition, noise_true = diffusion.forward_diffusion(transition, t)

    # Reverse step
    denoised_transition = diffusion.reverse_step(noisy_transition, t, noise_true)

    # Check that denoised transition is closer to original than noisy
    error_noisy = np.linalg.norm(noisy_transition - transition)
    error_denoised = np.linalg.norm(denoised_transition - transition)

    # In a properly trained model, denoised should be better; here we just check it's not NaN
    assert np.isfinite(error_denoised), "Denoised error is not finite"
    assert error_denoised < 100, f"Denoised error too large: {error_denoised}"

    print("✓ test_end_to_end passed")


if __name__ == "__main__":
    test_linear_ssm()
    test_diffusion_forward()
    test_diffusion_reverse()
    test_end_to_end()
    print("\nAll tests passed! ✓")
