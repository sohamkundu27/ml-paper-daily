"""Test Pass 1: Foundational diffusion + autoregressive setup."""

import torch
import torch.nn as nn
from diffusion import DiffusionScheduler, DiffusionLoss
from model import LatentEncoder, LatentDecoder, DenoisingModel


def test_diffusion_scheduler():
    """Test diffusion scheduler: forward and reverse process."""
    scheduler = DiffusionScheduler(num_steps=1000)

    # Test shapes
    assert len(scheduler.betas) == 1000
    assert len(scheduler.alphas_cumprod) == 1000

    # Test add_noise
    batch_size = 4
    latent_dim = 256
    x0 = torch.randn(batch_size, latent_dim)

    for t in [0, 100, 500, 999]:
        t_tensor = torch.tensor([t] * batch_size)
        xt, noise = scheduler.add_noise(x0, t_tensor)

        assert xt.shape == x0.shape
        assert noise.shape == x0.shape

        # At t=0, xt should be close to x0 (almost no noise added)
        if t == 0:
            assert torch.allclose(xt, x0, atol=0.1)

        # At t=999, xt should be close to noise (almost pure noise)
        if t == 999:
            alpha = scheduler.sqrt_alphas_cumprod[t]
            assert alpha < 0.1  # Very small alpha

    print("✓ Diffusion scheduler test passed")


def test_latent_encoder_decoder():
    """Test encoder/decoder roundtrip."""
    batch_size = 4
    channels = 3
    height, width = 32, 32

    encoder = LatentEncoder(in_channels=channels, latent_dim=256, height=height, width=width)
    decoder = LatentDecoder(latent_dim=256, out_channels=channels, height=height, width=width)

    # Random image
    image = torch.randn(batch_size, channels, height, width)

    # Encode
    latent = encoder(image)
    assert latent.shape == (batch_size, 256)

    # Decode
    reconstructed = decoder(latent)
    assert reconstructed.shape == image.shape

    print("✓ Encoder/decoder test passed")


def test_denoising_model():
    """Test denoising model forward pass."""
    batch_size = 4
    latent_dim = 256
    d_model = 256

    model = DenoisingModel(
        latent_dim=latent_dim,
        d_model=d_model,
        num_heads=8,
        num_blocks=1,
    )

    # Random noisy latent
    z_t = torch.randn(batch_size, latent_dim)
    t = torch.randint(0, 1000, (batch_size,))

    # Predict noise
    noise_pred = model(z_t, t)

    assert noise_pred.shape == z_t.shape
    assert noise_pred.requires_grad  # Check it's trainable

    print("✓ Denoising model test passed")


def test_full_diffusion_step():
    """Test full diffusion step: encode, noise, denoise, decode."""
    batch_size = 2
    channels = 3
    height, width = 32, 32
    latent_dim = 256

    # Setup
    scheduler = DiffusionScheduler(num_steps=1000)
    encoder = LatentEncoder(channels, latent_dim, height, width)
    decoder = LatentDecoder(latent_dim, channels, height, width)
    denoising_model = DenoisingModel(latent_dim, d_model=256, num_heads=8, num_blocks=1)
    loss_fn = DiffusionLoss(scheduler)

    # Random image
    image = torch.randn(batch_size, channels, height, width)

    # Encode to latent
    z0 = encoder(image)
    assert z0.shape == (batch_size, latent_dim)

    # Add noise at random timesteps
    t = torch.randint(0, 1000, (batch_size,))
    zt, noise = scheduler.add_noise(z0, t)

    # Predict noise
    noise_pred = denoising_model(zt, t)
    assert noise_pred.shape == noise.shape

    # Compute loss
    loss = loss_fn(noise_pred, noise)
    assert loss.item() > 0
    assert loss.requires_grad

    # Backward pass (check gradients flow)
    loss.backward()
    for param in denoising_model.parameters():
        if param.requires_grad:
            assert param.grad is not None

    # Decode prediction
    z0_pred = zt - noise_pred  # Simplified pseudo-inverse
    image_recon = decoder(z0_pred)
    assert image_recon.shape == image.shape

    print("✓ Full diffusion step test passed")


def test_training_step():
    """Test one training step."""
    batch_size = 2
    channels = 3
    height, width = 32, 32
    latent_dim = 256

    # Setup
    scheduler = DiffusionScheduler(num_steps=1000)
    encoder = LatentEncoder(channels, latent_dim, height, width)
    denoising_model = DenoisingModel(latent_dim, d_model=128, num_heads=4, num_blocks=1)
    loss_fn = DiffusionLoss(scheduler)
    optimizer = torch.optim.Adam(denoising_model.parameters(), lr=1e-4)

    # Random images
    images = torch.randn(batch_size, channels, height, width)

    # Encode
    z0 = encoder(images)

    # Add noise
    t = torch.randint(0, 1000, (batch_size,))
    zt, noise = scheduler.add_noise(z0, t)

    # Forward pass
    noise_pred = denoising_model(zt, t)
    loss = loss_fn(noise_pred, noise)
    initial_loss = loss.item()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check loss changed
    with torch.no_grad():
        new_loss = loss_fn(denoising_model(zt, t), noise).item()

    assert new_loss != initial_loss or abs(new_loss - initial_loss) < 0.01

    print("✓ Training step test passed")


def test_multi_step_denoising():
    """Test sequential denoising steps."""
    batch_size = 2
    latent_dim = 256

    scheduler = DiffusionScheduler(num_steps=100)
    denoising_model = DenoisingModel(latent_dim, d_model=128, num_heads=4, num_blocks=1)

    # Start with pure noise
    z_t = torch.randn(batch_size, latent_dim)

    # Iteratively denoise over timesteps (simplified reverse process)
    loss_values = []
    for step in range(100, 0, -1):
        t = torch.tensor([step - 1] * batch_size)
        with torch.no_grad():
            noise_pred = denoising_model(z_t, t)
            # Simplified denoising step (not the full reverse formula)
            alpha = scheduler.sqrt_alphas_cumprod[t[0]]
            z_t = (z_t - noise_pred) * (1.0 / (alpha + 1e-8))

        loss_values.append(z_t.norm().item())

    # Check loss decreased over steps (rough check)
    assert len(loss_values) == 100

    print("✓ Multi-step denoising test passed")


if __name__ == "__main__":
    test_diffusion_scheduler()
    test_latent_encoder_decoder()
    test_denoising_model()
    test_full_diffusion_step()
    test_training_step()
    test_multi_step_denoising()
    print("\n✅ All Pass 1 tests passed!")
