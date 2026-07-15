import numpy as np
import torch
from asyncpatch import HeterogeneousNoiseScheduler, AsyncPatchDiffusion


def test_noise_scheduler():
    """Test that noise scheduler produces valid α and σ values."""
    scheduler = HeterogeneousNoiseScheduler(T=1000)

    # At t=0, signal should be nearly 1, noise nearly 0
    alpha_0 = scheduler.get_alpha_t(0)
    sigma_0 = scheduler.get_sigma_t(0)
    assert 0.9 < alpha_0 <= 1.0, f"Expected alpha_0 ≈ 1, got {alpha_0}"
    assert 0.0 <= sigma_0 < 0.1, f"Expected sigma_0 ≈ 0, got {sigma_0}"

    # At t=T-1, signal should be small, noise should be large
    alpha_T = scheduler.get_alpha_t(999)
    sigma_T = scheduler.get_sigma_t(999)
    assert 0.0 <= alpha_T < 0.2, f"Expected alpha_T small, got {alpha_T}"
    assert 0.8 < sigma_T <= 1.0, f"Expected sigma_T ≈ 1, got {sigma_T}"

    # Variance should be preserved: alpha_t^2 + sigma_t^2 ≈ 1
    for t in [0, 100, 500, 999]:
        alpha_t = scheduler.get_alpha_t(t)
        sigma_t = scheduler.get_sigma_t(t)
        variance = alpha_t**2 + sigma_t**2
        assert abs(variance - 1.0) < 0.01, f"Variance at t={t}: {variance}, expected ≈ 1"

    print("✓ Noise scheduler test passed")


def test_heterogeneous_timesteps():
    """Test that timestep sampling works for different strategies."""
    scheduler = HeterogeneousNoiseScheduler(T=1000)

    # Uniform strategy
    timesteps = scheduler.sample_heterogeneous_timesteps(
        batch_size=2, num_patches=16, strategy="uniform"
    )
    assert timesteps.shape == (2, 16)
    assert np.all(timesteps >= 0) and np.all(timesteps < 1000)

    # Mixed strategy (some patches at t=0)
    timesteps_mixed = scheduler.sample_heterogeneous_timesteps(
        batch_size=2, num_patches=16, strategy="mixed"
    )
    assert timesteps_mixed.shape == (2, 16)
    assert np.any(timesteps_mixed == 0), "Expected some patches at t=0 in mixed strategy"

    print("✓ Heterogeneous timesteps test passed")


def test_forward_diffusion_shapes():
    """Test that forward diffusion produces correct output shapes."""
    scheduler = HeterogeneousNoiseScheduler(T=1000)
    diffusion = AsyncPatchDiffusion(scheduler)

    # Simple 16x16 image, 4 channels (RGBA), batch size 2
    x0 = torch.randn(2, 4, 16, 16)
    timesteps = scheduler.sample_heterogeneous_timesteps(batch_size=2, num_patches=16, strategy="uniform")

    xt, noise = diffusion.forward(x0, timesteps, patch_size=4)

    assert xt.shape == x0.shape, f"Expected xt shape {x0.shape}, got {xt.shape}"
    assert noise.shape == x0.shape, f"Expected noise shape {x0.shape}, got {noise.shape}"

    print("✓ Forward diffusion shapes test passed")


def test_variance_preservation():
    """Test that variance is approximately preserved across patches."""
    scheduler = HeterogeneousNoiseScheduler(T=1000)
    diffusion = AsyncPatchDiffusion(scheduler)

    timesteps = scheduler.sample_heterogeneous_timesteps(batch_size=3, num_patches=25, strategy="uniform")
    variances = diffusion.get_patch_variance(timesteps, patch_size=1)

    # All variances should be close to 1 (within numerical tolerance)
    assert variances.shape == timesteps.shape
    assert np.allclose(variances, 1.0, atol=0.01), \
        f"Variances not preserved. Min: {variances.min()}, Max: {variances.max()}"

    print("✓ Variance preservation test passed")


def test_forward_with_fixed_timesteps():
    """Test forward pass with fixed (deterministic) timestep assignments."""
    scheduler = HeterogeneousNoiseScheduler(T=1000)
    diffusion = AsyncPatchDiffusion(scheduler)

    # Create simple toy image
    x0 = torch.ones(1, 1, 8, 8)  # White image
    timesteps = np.array([[0, 100, 500, 999, 0, 100, 500, 999]])  # Different levels per patch

    # Should not crash
    xt, noise = diffusion.forward(x0, timesteps, patch_size=4)

    # At t=0 patches, should be nearly identical to original
    # At t=999 patches, should be almost pure noise
    assert xt.shape == x0.shape
    assert not torch.isnan(xt).any(), "Output contains NaN"
    assert not torch.isinf(xt).any(), "Output contains inf"

    print("✓ Forward with fixed timesteps test passed")


def test_mixed_corruption_levels():
    """Test that patches with different timesteps have different corruption levels."""
    torch.manual_seed(42)
    np.random.seed(42)

    scheduler = HeterogeneousNoiseScheduler(T=1000)
    diffusion = AsyncPatchDiffusion(scheduler)

    # Single patch at different timesteps
    x0 = torch.ones(2, 1, 8, 8)

    timesteps_low = np.array([[0, 0, 0, 0]])  # All patches clean
    timesteps_high = np.array([[999, 999, 999, 999]])  # All patches heavily corrupted

    xt_low, _ = diffusion.forward(x0[0:1], timesteps_low, patch_size=4)
    xt_high, _ = diffusion.forward(x0[1:2], timesteps_high, patch_size=4)

    # Low corruption should be closer to original
    diff_low = (xt_low - x0[0:1]).abs().mean()
    diff_high = (xt_high - x0[1:2]).abs().mean()

    assert diff_low < diff_high, \
        f"Expected diff_low < diff_high, got {diff_low} >= {diff_high}"

    print("✓ Mixed corruption levels test passed")


if __name__ == "__main__":
    test_noise_scheduler()
    test_heterogeneous_timesteps()
    test_forward_diffusion_shapes()
    test_variance_preservation()
    test_forward_with_fixed_timesteps()
    test_mixed_corruption_levels()
    print("\n✅ All tests passed!")
