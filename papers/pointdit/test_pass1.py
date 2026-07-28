"""
Test Pass 1: Basic diffusion transformer backbone.
Verify that the model can process point patches and predict noise.
"""
import torch
from diffusion import DiffusionTransformer, GaussianDiffusion


def test_diffusion_forward_pass():
    """Test forward pass through diffusion transformer."""
    B, N, patch_size, point_dim = 2, 16, 16, 3
    hidden_dim = 64

    model = DiffusionTransformer(
        point_dim=point_dim,
        patch_size=patch_size,
        num_layers=4,
        hidden_dim=hidden_dim,
        num_heads=4,
    )
    model.eval()

    # Create random point patches
    x = torch.randn(B, N, patch_size * point_dim)

    # Create timestep indices
    t = torch.tensor([100, 500])

    # Forward pass
    noise_pred = model(x, t)

    # Verify output shape
    assert noise_pred.shape == x.shape, f"Expected shape {x.shape}, got {noise_pred.shape}"
    print(f"✓ Model forward pass: input shape {x.shape} -> output shape {noise_pred.shape}")


def test_diffusion_noise_schedule():
    """Test that diffusion schedule is well-formed."""
    diffusion = GaussianDiffusion(num_steps=1000, beta_start=0.0001, beta_end=0.02)

    # Check that sqrt_alphas_cumprod is monotonically decreasing
    alphas = diffusion.get_buffer("sqrt_alphas_cumprod")
    assert alphas[0] > alphas[-1], "sqrt_alphas_cumprod should decrease over time"
    assert torch.all(alphas >= 0) and torch.all(alphas <= 1), "alphas should be in [0, 1]"
    print(f"✓ Noise schedule: alpha(0)={alphas[0]:.4f}, alpha({999})={alphas[-1]:.4f}")


def test_forward_diffusion():
    """Test forward diffusion (adding noise)."""
    B, N, patch_size, point_dim = 2, 16, 16, 3
    diffusion = GaussianDiffusion(num_steps=1000)

    # Clean point maps (small values)
    x0 = torch.randn(B, N, patch_size * point_dim) * 0.1

    # At t=0, xt should be close to x0
    t_early = torch.tensor([0, 0])
    xt_early, noise_early = diffusion.forward_diffusion(x0, t_early)
    mse_early = torch.mean((xt_early - x0) ** 2).item()
    assert mse_early < 0.01, f"At t=0, xt should ≈ x0, MSE={mse_early:.4f}"

    # At t=999, xt should be mostly noise
    t_late = torch.tensor([999, 999])
    xt_late, noise_late = diffusion.forward_diffusion(x0, t_late)
    noise_fraction = torch.std(xt_late).item() / (torch.std(x0).item() + 1e-6)
    assert noise_fraction > 0.9, f"At t=999, xt should be ~noise, ratio={noise_fraction:.4f}"

    print(f"✓ Forward diffusion: t=0 (clean) MSE={mse_early:.4f}, t=999 (noisy) std_ratio={noise_fraction:.4f}")


def test_end_to_end_denoising():
    """Test a single reverse diffusion step."""
    B, N, patch_size, point_dim = 2, 16, 16, 3
    hidden_dim = 64

    model = DiffusionTransformer(point_dim=point_dim, patch_size=patch_size, hidden_dim=hidden_dim)
    model.eval()

    diffusion = GaussianDiffusion(num_steps=1000)

    # Start from pure noise at t=500
    xt = torch.randn(B, N, patch_size * point_dim)
    t = torch.tensor([500, 500])

    # Single reverse step
    x_prev = diffusion.reverse_diffusion(model, xt, t, device='cpu')

    # Check shape is preserved
    assert x_prev.shape == xt.shape, f"Shape mismatch in reverse step"
    print(f"✓ Reverse diffusion: single step maintains shape {x_prev.shape}")


def test_multi_step_reverse():
    """Test multiple reverse diffusion steps."""
    B, N, patch_size, point_dim = 1, 9, 16, 3
    hidden_dim = 64

    model = DiffusionTransformer(point_dim=point_dim, patch_size=patch_size, hidden_dim=hidden_dim)
    model.eval()

    diffusion = GaussianDiffusion(num_steps=1000)

    # Start from pure noise
    x_t = torch.randn(B, N, patch_size * point_dim)

    # Denoise over multiple steps
    num_steps = 10
    for step in range(num_steps):
        t_val = max(0, 999 - step * 100)
        t = torch.tensor([t_val])
        x_t = diffusion.reverse_diffusion(model, x_t, t, device='cpu')

    # Check final shape
    assert x_t.shape == (B, N, patch_size * point_dim)
    print(f"✓ Multi-step denoising: {num_steps} steps complete, final shape {x_t.shape}")


if __name__ == "__main__":
    print("\n=== Pass 1: Diffusion Backbone Tests ===\n")
    test_diffusion_forward_pass()
    test_diffusion_noise_schedule()
    test_forward_diffusion()
    test_end_to_end_denoising()
    test_multi_step_reverse()
    print("\n=== All tests passed ===\n")
