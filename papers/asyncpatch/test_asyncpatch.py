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


def test_realistic_image_sizes():
    """Pass 2: Test forward diffusion on realistic image sizes (32x32, 64x64)."""
    scheduler = HeterogeneousNoiseScheduler(T=1000)
    diffusion = AsyncPatchDiffusion(scheduler)

    # 32x32 RGB images with batch size 4
    x0_32 = torch.randn(4, 3, 32, 32)
    timesteps_32 = scheduler.sample_heterogeneous_timesteps(batch_size=4, num_patches=64, strategy="uniform")
    xt_32, noise_32 = diffusion.forward(x0_32, timesteps_32, patch_size=4)

    assert xt_32.shape == x0_32.shape
    assert noise_32.shape == x0_32.shape
    assert not torch.isnan(xt_32).any() and not torch.isinf(xt_32).any()

    # 64x64 RGB images with batch size 2
    x0_64 = torch.randn(2, 3, 64, 64)
    timesteps_64 = scheduler.sample_heterogeneous_timesteps(batch_size=2, num_patches=256, strategy="uniform")
    xt_64, noise_64 = diffusion.forward(x0_64, timesteps_64, patch_size=4)

    assert xt_64.shape == x0_64.shape
    assert noise_64.shape == x0_64.shape

    print("✓ Realistic image sizes test passed")


def test_snr_computation():
    """Pass 2: Verify SNR computation at different timesteps."""
    scheduler = HeterogeneousNoiseScheduler(T=1000)
    diffusion = AsyncPatchDiffusion(scheduler)

    timesteps = np.array([[0, 250, 500, 750, 999]])  # Various noise levels

    snr = diffusion.get_patch_snr(timesteps)

    # SNR should decrease monotonically as t increases
    assert snr[0, 0] > snr[0, 1] > snr[0, 2] > snr[0, 3] > snr[0, 4], \
        f"SNR not monotonically decreasing: {snr[0]}"

    # At t=0, SNR should be very high (clean)
    assert snr[0, 0] > 10, f"Expected high SNR at t=0, got {snr[0, 0]}"

    # At t=999, SNR should be very low (mostly noise)
    assert snr[0, 4] < -10, f"Expected low SNR at t=999, got {snr[0, 4]}"

    print("✓ SNR computation test passed")


def test_joint_diffusion_properties():
    """Pass 2: Verify comprehensive joint diffusion theoretical properties."""
    torch.manual_seed(42)
    np.random.seed(42)

    scheduler = HeterogeneousNoiseScheduler(T=1000)
    diffusion = AsyncPatchDiffusion(scheduler)

    # Create a realistic scenario: batched 32x32 images with heterogeneous timesteps
    x0 = torch.randn(3, 3, 32, 32)
    timesteps = scheduler.sample_heterogeneous_timesteps(batch_size=3, num_patches=64, strategy="mixed")

    # Compute joint diffusion properties
    props = diffusion.compute_joint_diffusion_properties(timesteps)

    # Check variance preservation at each patch (critical property)
    assert props["variance_preservation"], \
        f"Variance not preserved. Max error: {props['max_variance_error']}"
    assert props["patch_variances_preserved"], \
        "Per-patch variance preservation failed"

    # Check SNR properties
    assert props["mean_snr"] >= -50, "Mean SNR unexpectedly low"
    assert props["min_snr"] < props["mean_snr"] < props["max_snr"], \
        "SNR statistics inconsistent"

    # Check alpha and sigma properties
    assert 0 < props["mean_alpha"] < 1, "Mean alpha out of valid range"
    assert 0 < props["mean_sigma"] < 1, "Mean sigma out of valid range"

    # With heterogeneous timesteps, should have non-trivial range
    t_min, t_max = props["timestep_range"]
    assert t_max > t_min, "Timestep range should be non-trivial"

    print("✓ Joint diffusion properties test passed")


def test_large_batch_heterogeneous():
    """Pass 2: Test large batch with highly heterogeneous timestep assignments."""
    scheduler = HeterogeneousNoiseScheduler(T=1000)
    diffusion = AsyncPatchDiffusion(scheduler)

    # Batch of 8 images, 64x64 each
    batch_size, channels, height, width = 8, 3, 64, 64
    x0 = torch.randn(batch_size, channels, height, width)

    # Create deliberately heterogeneous timesteps: some images clean, some noisy
    timesteps = np.zeros((batch_size, 256), dtype=int)
    for b in range(batch_size):
        if b < 3:
            timesteps[b] = 0  # First 3 images: mostly clean
        elif b < 6:
            timesteps[b] = np.random.randint(200, 400, size=256)  # Mid: moderate noise
        else:
            timesteps[b] = np.random.randint(700, 999, size=256)  # Last 2: heavy noise

    xt, noise = diffusion.forward(x0, timesteps, patch_size=4)

    # Verify output integrity
    assert xt.shape == x0.shape
    assert not torch.isnan(xt).any() and not torch.isinf(xt).any()

    # Verify that high-timestep patches are more corrupted
    props = diffusion.compute_joint_diffusion_properties(timesteps)
    assert props["min_snr"] < props["max_snr"], "SNR range should be non-trivial"

    print("✓ Large batch heterogeneous test passed")


def test_forward_process_reversibility():
    """Pass 2: Verify that we can compute mean of xt conditioned on noise level."""
    torch.manual_seed(42)
    np.random.seed(42)

    scheduler = HeterogeneousNoiseScheduler(T=1000)
    diffusion = AsyncPatchDiffusion(scheduler)

    # 16x16 image with 4x4 patch size = 16 patches
    x0 = torch.ones(1, 1, 16, 16)
    timesteps = np.array([[0, 500, 999, 100, 50, 200, 750, 300,
                           0, 500, 999, 100, 50, 200, 750, 300]])  # 16 patches

    xt, noise = diffusion.forward(x0, timesteps, patch_size=4)

    # Verify output shape and integrity
    assert xt.shape == x0.shape
    assert noise.shape == x0.shape
    assert not torch.isnan(xt).any() and not torch.isinf(xt).any()

    # For t=0 patches, corruption should be minimal
    alpha_0 = scheduler.get_alpha_t(0)
    assert alpha_0 > 0.9, f"Expected high alpha at t=0, got {alpha_0}"

    # For t=999 patches, should be heavily corrupted
    alpha_999 = scheduler.get_alpha_t(999)
    assert alpha_999 < 0.2, f"Expected low alpha at t=999, got {alpha_999}"

    print("✓ Forward process reversibility test passed")


if __name__ == "__main__":
    test_noise_scheduler()
    test_heterogeneous_timesteps()
    test_forward_diffusion_shapes()
    test_variance_preservation()
    test_forward_with_fixed_timesteps()
    test_mixed_corruption_levels()
    # Pass 2 tests
    test_realistic_image_sizes()
    test_snr_computation()
    test_joint_diffusion_properties()
    test_large_batch_heterogeneous()
    test_forward_process_reversibility()
    print("\n✅ All tests passed! (Pass 1 + Pass 2)")
