import torch
import torch.nn as nn
from diffusion import AudioDiffusionModel, GaussianDiffusion


def test_forward_diffusion():
    """Test forward diffusion process."""
    diffusion = GaussianDiffusion(timesteps=100)

    # Create clean audio (batch_size=4, audio_dim=128)
    x0 = torch.randn(4, 128)
    noise = torch.randn_like(x0)

    # Test diffusion at different timesteps
    for t in [0, 50, 99]:
        x_t = diffusion.q_sample(x0, t, noise)
        assert x_t.shape == x0.shape, f"Shape mismatch: {x_t.shape} vs {x0.shape}"

    # Test with deterministic inputs: x0=1, noise=0
    x0_ones = torch.ones(2, 128)
    noise_zeros = torch.zeros_like(x0_ones)

    # At t=0, x_t = alpha_sqrt * 1 + (1-alpha_sqrt) * 0 = alpha_sqrt
    x_t_early = diffusion.q_sample(x0_ones, 0, noise_zeros)
    alpha_sqrt_0 = diffusion.sqrt_alphas_cumprod[0]
    assert torch.allclose(x_t_early, alpha_sqrt_0 * x0_ones, atol=1e-5), "At t=0, formula check failed"

    # At t=99, alpha_bar is moderate (not tiny)
    alpha_bar_99 = diffusion.alphas_cumprod[99]
    assert 0.1 < alpha_bar_99 < 0.9, "Alpha_bar at t=99 should be in reasonable range"

    print("✓ Forward diffusion test passed")


def test_model_forward():
    """Test model forward pass."""
    model = AudioDiffusionModel(audio_dim=128, time_dim=64, hidden_dim=256)
    diffusion = GaussianDiffusion(timesteps=100)

    batch_size = 4
    x = torch.randn(batch_size, 128)
    t = torch.full((batch_size,), 50, dtype=torch.long)
    t_norm = t.float() / diffusion.timesteps

    # Forward pass
    noise_pred = model(x, t_norm)

    assert noise_pred.shape == x.shape, f"Output shape mismatch: {noise_pred.shape} vs {x.shape}"
    assert not torch.isnan(noise_pred).any(), "Model output contains NaN"
    assert not torch.isinf(noise_pred).any(), "Model output contains inf"

    print("✓ Model forward pass test passed")


def test_training_step():
    """Test a simple training step."""
    model = AudioDiffusionModel(audio_dim=128, time_dim=64, hidden_dim=256)
    diffusion = GaussianDiffusion(timesteps=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 4
    x0 = torch.randn(batch_size, 128)
    noise = torch.randn_like(x0)
    t = torch.randint(0, 100, (batch_size,))
    t_norm = t.float() / diffusion.timesteps

    # Forward diffusion
    x_t = diffusion.q_sample(x0, t, noise)

    # Model forward
    noise_pred = model(x_t, t_norm)

    # Simple MSE loss
    loss = nn.MSELoss()(noise_pred, noise)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("✓ Training step test passed")


def test_sampling():
    """Test sampling from the model."""
    model = AudioDiffusionModel(audio_dim=128, time_dim=64, hidden_dim=256)
    diffusion = GaussianDiffusion(timesteps=100)

    device = torch.device("cpu")
    batch_size = 2

    # Sample
    samples = diffusion.sample(model, (batch_size, 128), device=device, num_steps=10)

    assert samples.shape == (batch_size, 128), f"Sample shape mismatch: {samples.shape}"
    assert not torch.isnan(samples).any(), "Samples contain NaN"
    assert not torch.isinf(samples).any(), "Samples contain inf"

    print("✓ Sampling test passed")


def test_consistency():
    """Test that model produces consistent outputs for same input."""
    model = AudioDiffusionModel(audio_dim=128, time_dim=64, hidden_dim=256)
    model.eval()

    x = torch.randn(1, 128)
    t = torch.tensor([0.5])

    with torch.no_grad():
        out1 = model(x, t)
        out2 = model(x, t)

    assert torch.allclose(out1, out2), "Model should be deterministic in eval mode"

    print("✓ Consistency test passed")


if __name__ == "__main__":
    test_forward_diffusion()
    test_model_forward()
    test_training_step()
    test_sampling()
    test_consistency()
    print("\n✅ All tests passed!")
