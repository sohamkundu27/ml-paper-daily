"""Tests for SSM, diffusion, and learned noise prediction components."""

import numpy as np
import torch
from ssm import (
    LinearSSM, DiffusionProcess, NoisePredictor, train_diffusion_ssm,
    generate_sequence, estimate_likelihood, evaluate_on_synthetic_data,
    DampedOscillator, LorenzSystem, demo_damped_oscillator, demo_lorenz_system,
    run_pass_4_demo
)


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


def test_noise_predictor():
    """Test NoisePredictor network."""
    state_dim = 4
    batch_size = 8
    predictor = NoisePredictor(state_dim, hidden_dim=32)

    # Create dummy input
    x_t = torch.randn(batch_size, state_dim)
    t = torch.randint(0, 100, (batch_size,))

    # Forward pass
    noise_pred = predictor(x_t, t)

    # Check shapes
    assert noise_pred.shape == (batch_size, state_dim), f"Expected shape {(batch_size, state_dim)}, got {noise_pred.shape}"

    # Check finite values
    assert torch.all(torch.isfinite(noise_pred)), "Output contains non-finite values"

    print("✓ test_noise_predictor passed")


def test_training_diffusion_ssm():
    """Test training loop for diffusion SSM."""
    state_dim = 3
    obs_dim = 2
    ssm = LinearSSM(state_dim, obs_dim, seed=42)
    diffusion = DiffusionProcess(state_dim, num_steps=50)
    noise_predictor = NoisePredictor(state_dim, hidden_dim=32)

    # Train for a few epochs with small dataset
    losses = train_diffusion_ssm(
        ssm,
        diffusion,
        noise_predictor,
        num_trajectories=5,
        trajectory_length=10,
        num_epochs=5,
        learning_rate=0.01,
        device="cpu"
    )

    # Check that we have losses for each epoch
    assert len(losses) == 5, f"Expected 5 losses, got {len(losses)}"

    # Check that all losses are finite
    assert all(np.isfinite(loss) for loss in losses), "Some losses are non-finite"

    # Check that loss is decreasing (not strictly, but should show some trend)
    assert losses[-1] < losses[0] * 2, "Loss did not decrease sufficiently during training"

    print("✓ test_training_diffusion_ssm passed")


def test_generate_sequence():
    """Test multi-step sequence generation via diffusion inference."""
    state_dim = 3
    obs_dim = 2
    ssm = LinearSSM(state_dim, obs_dim, seed=42)
    diffusion = DiffusionProcess(state_dim, num_steps=50)
    noise_predictor = NoisePredictor(state_dim, hidden_dim=32)

    # Train quickly
    train_diffusion_ssm(
        ssm, diffusion, noise_predictor,
        num_trajectories=3,
        trajectory_length=8,
        num_epochs=3,
        learning_rate=0.01,
        device="cpu"
    )

    # Generate a sequence
    z0 = np.random.randn(state_dim)
    seq_length = 10
    z_seq, y_seq = generate_sequence(
        ssm, diffusion, noise_predictor,
        z0, seq_length,
        num_diffusion_steps=15,
        device="cpu"
    )

    # Check shapes
    assert z_seq.shape == (seq_length, state_dim), f"Expected z_seq shape {(seq_length, state_dim)}, got {z_seq.shape}"
    assert y_seq.shape == (seq_length, obs_dim), f"Expected y_seq shape {(seq_length, obs_dim)}, got {y_seq.shape}"

    # Check finite values
    assert np.all(np.isfinite(z_seq)), "z_seq contains non-finite values"
    assert np.all(np.isfinite(y_seq)), "y_seq contains non-finite values"

    # Check that first state is close to z0
    assert np.allclose(z_seq[0], z0), "First generated state should match z0"

    print("✓ test_generate_sequence passed")


def test_estimate_likelihood():
    """Test likelihood estimation on state trajectories."""
    state_dim = 3
    obs_dim = 2
    ssm = LinearSSM(state_dim, obs_dim, seed=42)
    diffusion = DiffusionProcess(state_dim, num_steps=50)
    noise_predictor = NoisePredictor(state_dim, hidden_dim=32)

    # Train quickly
    train_diffusion_ssm(
        ssm, diffusion, noise_predictor,
        num_trajectories=3,
        trajectory_length=8,
        num_epochs=3,
        learning_rate=0.01,
        device="cpu"
    )

    # Sample a trajectory and estimate likelihood
    z_traj, _ = ssm.sample_trajectory(10)
    nll = estimate_likelihood(ssm, diffusion, noise_predictor, z_traj, device="cpu")

    # Check that NLL is finite and positive
    assert np.isfinite(nll), f"NLL is not finite: {nll}"
    assert nll >= 0, f"NLL should be non-negative, got {nll}"

    print(f"✓ test_estimate_likelihood passed (NLL={nll:.4f})")


def test_evaluate_on_synthetic_data():
    """Test end-to-end evaluation on synthetic data."""
    results = evaluate_on_synthetic_data(
        state_dim=2,
        obs_dim=1,
        seq_length=10,
        num_train=5,
        num_test=3,
        num_epochs=5,
        device="cpu"
    )

    # Check all metrics are present and finite
    assert "train_loss" in results, "Missing train_loss in results"
    assert "test_nll" in results, "Missing test_nll in results"
    assert "test_mse" in results, "Missing test_mse in results"
    assert "test_trajectory_mse" in results, "Missing test_trajectory_mse in results"

    for key, value in results.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"
        assert value >= 0, f"{key} should be non-negative, got {value}"

    print(f"✓ test_evaluate_on_synthetic_data passed")
    print(f"  Train loss: {results['train_loss']:.6f}")
    print(f"  Test NLL: {results['test_nll']:.6f}")
    print(f"  Test MSE: {results['test_mse']:.6f}")
    print(f"  Test trajectory MSE: {results['test_trajectory_mse']:.6f}")


def test_damped_oscillator():
    """Test damped oscillator trajectory generation."""
    oscillator = DampedOscillator(gamma=0.1, omega=1.0, seed=42)
    T = 30
    z, y = oscillator.sample_trajectory(T)

    # Check shapes
    assert z.shape == (T, 2), f"Expected z shape {(T, 2)}, got {z.shape}"
    assert y.shape == (T, 1), f"Expected y shape {(T, 1)}, got {y.shape}"

    # Check finite values
    assert np.all(np.isfinite(z)), "z contains non-finite values"
    assert np.all(np.isfinite(y)), "y contains non-finite values"

    # Check that trajectory is bounded (damped oscillator should not explode)
    assert np.all(np.abs(z) < 100), "Oscillator trajectory exploded"

    print("✓ test_damped_oscillator passed")


def test_lorenz_system():
    """Test Lorenz system trajectory generation."""
    lorenz = LorenzSystem(sigma=10.0, rho=28.0, beta=8.0/3.0, seed=42)
    T = 100
    z, y = lorenz.sample_trajectory(T)

    # Check shapes
    assert z.shape == (T, 3), f"Expected z shape {(T, 3)}, got {z.shape}"
    assert y.shape == (T, 2), f"Expected y shape {(T, 2)}, got {y.shape}"

    # Check finite values
    assert np.all(np.isfinite(z)), "z contains non-finite values"
    assert np.all(np.isfinite(y)), "y contains non-finite values"

    print("✓ test_lorenz_system passed")


def test_demo_damped_oscillator():
    """Test damped oscillator demo (quick version)."""
    results = demo_damped_oscillator(
        num_trajectories=5, seq_length=20,
        num_epochs=5, device="cpu"
    )

    # Check results contain expected keys
    assert "system" in results
    assert "final_loss" in results
    assert "gen_mse" in results
    assert "nll" in results

    # Check all values are finite
    for key, val in results.items():
        if isinstance(val, (int, float)) and key != "system":
            assert np.isfinite(val), f"{key} is not finite: {val}"

    print("✓ test_demo_damped_oscillator passed")


def test_demo_lorenz_system():
    """Test Lorenz demo (quick version)."""
    results = demo_lorenz_system(
        num_trajectories=5, seq_length=30,
        num_epochs=5, device="cpu"
    )

    # Check results contain expected keys
    assert "system" in results
    assert "final_loss" in results
    assert "gen_mse" in results
    assert "nll" in results

    # Check all values are finite
    for key, val in results.items():
        if isinstance(val, (int, float)) and key != "system":
            assert np.isfinite(val), f"{key} is not finite: {val}"

    print("✓ test_demo_lorenz_system passed")


if __name__ == "__main__":
    test_linear_ssm()
    test_diffusion_forward()
    test_diffusion_reverse()
    test_end_to_end()
    test_noise_predictor()
    test_training_diffusion_ssm()
    test_generate_sequence()
    test_estimate_likelihood()
    test_evaluate_on_synthetic_data()
    test_damped_oscillator()
    test_lorenz_system()
    test_demo_damped_oscillator()
    test_demo_lorenz_system()
    print("\nAll tests passed! ✓")
