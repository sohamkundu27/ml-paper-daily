import torch
import numpy as np
from diffusion_forcing import cosine_schedule, get_alpha_beta, Denoiser, DiffusionForcing


def test_cosine_schedule():
    """Test noise schedule properties."""
    # At t=0, alpha should be 1 (no noise)
    alpha_0 = cosine_schedule(0)
    assert np.isclose(alpha_0, 1.0, atol=0.01), f"Expected alpha(0) ≈ 1, got {alpha_0}"

    # At t=1, alpha should be ~0 (full noise)
    alpha_1 = cosine_schedule(1)
    assert alpha_1 < 0.1, f"Expected alpha(1) << 1, got {alpha_1}"

    # Should be monotonically decreasing
    ts = np.linspace(0, 1, 100)
    alphas = [cosine_schedule(t) for t in ts]
    for i in range(len(alphas) - 1):
        assert alphas[i] >= alphas[i + 1], "Schedule should be monotonically decreasing"

    print("✓ Cosine schedule test passed")


def test_alpha_beta():
    """Test alpha and beta properties."""
    t = 0.5
    alpha, beta = get_alpha_beta(t)

    # Alpha + beta should equal 1
    assert np.isclose(alpha + beta, 1.0), "Alpha + Beta should equal 1"

    # Both should be in [0, 1]
    assert 0 <= alpha <= 1, f"Alpha {alpha} out of range"
    assert 0 <= beta <= 1, f"Beta {beta} out of range"

    print("✓ Alpha-beta properties test passed")


def test_denoiser_forward():
    """Test denoiser network forward pass."""
    token_dim = 16
    batch_size = 4
    seq_len = 8

    denoiser = Denoiser(token_dim)

    # Create random inputs
    x_t = torch.randn(batch_size, seq_len, token_dim)
    t = torch.ones(batch_size) * 0.5

    # Forward pass
    pred = denoiser(x_t, t)

    # Check output shape
    assert pred.shape == (batch_size, seq_len, token_dim), \
        f"Expected shape {(batch_size, seq_len, token_dim)}, got {pred.shape}"

    # Check no NaNs
    assert not torch.isnan(pred).any(), "Denoiser output contains NaN"

    print("✓ Denoiser forward pass test passed")


def test_forward_diffusion():
    """Test forward diffusion process."""
    token_dim = 16
    batch_size = 4
    seq_len = 8

    df = DiffusionForcing(token_dim)

    # Create clean tokens
    x_0 = torch.randn(batch_size, seq_len, token_dim)

    # At t=0, should add very little noise
    t_0 = torch.zeros(batch_size)
    x_t_0, alpha_0, beta_0 = df.forward_diffusion(x_0, t_0)
    assert torch.allclose(x_t_0, x_0, atol=0.1), "At t=0, x_t should be close to x_0"

    # At t=1, should add lot of noise
    t_1 = torch.ones(batch_size)
    x_t_1, alpha_1, beta_1 = df.forward_diffusion(x_0, t_1)
    # The noise-corrupted version should differ significantly
    diff = torch.abs(x_t_1 - x_0).mean()
    assert diff > 0.5, f"At t=1, difference from x_0 should be large, got {diff}"

    print("✓ Forward diffusion test passed")


def test_end_to_end():
    """End-to-end test: add noise and try to denoise."""
    token_dim = 8
    batch_size = 2
    seq_len = 4

    df = DiffusionForcing(token_dim)

    # Create clean tokens
    x_0 = torch.randn(batch_size, seq_len, token_dim)

    # Forward diffusion at mid-noise level
    t = torch.ones(batch_size) * 0.5
    x_t, alpha, beta = df.forward_diffusion(x_0, t)

    # Try to denoise
    x_pred = df.denoise(x_t, t)

    # Check output shape
    assert x_pred.shape == x_0.shape, f"Shape mismatch: {x_pred.shape} vs {x_0.shape}"

    # Untrained denoiser probably won't denoise well, but should produce valid output
    assert not torch.isnan(x_pred).any(), "Denoiser output contains NaN"
    assert not torch.isinf(x_pred).any(), "Denoiser output contains Inf"

    print("✓ End-to-end test passed")


def test_different_timesteps():
    """Test that different timesteps produce different noise levels."""
    token_dim = 8
    batch_size = 2
    seq_len = 4

    df = DiffusionForcing(token_dim)
    x_0 = torch.randn(batch_size, seq_len, token_dim)

    t_early = torch.ones(batch_size) * 0.2
    t_late = torch.ones(batch_size) * 0.8

    torch.manual_seed(42)
    x_t_early, _, _ = df.forward_diffusion(x_0, t_early)

    torch.manual_seed(42)
    x_t_late, _, _ = df.forward_diffusion(x_0, t_late)

    # Early should be closer to clean (alpha_early > alpha_late)
    early_diff = torch.abs(x_t_early - x_0).mean()
    late_diff = torch.abs(x_t_late - x_0).mean()

    assert late_diff > early_diff, \
        f"Later timestep should have more noise: early_diff={early_diff}, late_diff={late_diff}"

    print("✓ Different timesteps test passed")


if __name__ == "__main__":
    test_cosine_schedule()
    test_alpha_beta()
    test_denoiser_forward()
    test_forward_diffusion()
    test_end_to_end()
    test_different_timesteps()
    print("\n✅ All tests passed!")
