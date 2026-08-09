"""Test the binary diffusion head."""

import torch
import torch.optim as optim
from binary_diffusion import (
    BinaryDiffusionHead,
    BinaryDiffusionSampler,
    SinusoidalPositionalEncoding,
)


def test_positional_encoding():
    """Test sinusoidal positional encoding."""
    dim = 64
    encoder = SinusoidalPositionalEncoding(dim)

    # Test scalar input
    t = torch.tensor([0.0])
    emb = encoder(t)
    assert emb.shape == (1, dim), f"Expected shape (1, {dim}), got {emb.shape}"

    # Test batch input
    t_batch = torch.tensor([0.0, 50.0, 100.0])
    emb_batch = encoder(t_batch)
    assert emb_batch.shape == (3, dim), f"Expected shape (3, {dim}), got {emb_batch.shape}"

    # Test that different timesteps produce different embeddings
    t0 = encoder(torch.tensor([0.0]))
    t1 = encoder(torch.tensor([1.0]))
    assert not torch.allclose(t0, t1), "Different timesteps should produce different embeddings"

    print("✓ Positional encoding test passed")


def test_diffusion_head_shapes():
    """Test that diffusion head produces expected tensor shapes."""
    model = BinaryDiffusionHead(token_dim=32, hidden_dim=128, num_timesteps=1000)
    model.eval()

    batch_size = 2
    token_dim = 32
    h, w = 4, 4

    # Create dummy noisy tokens
    noisy_tokens = torch.randn(batch_size, token_dim, h, w)
    t = torch.randint(0, 1000, (batch_size,))

    with torch.no_grad():
        pred_tokens = model(noisy_tokens, t)

    # Check output shape
    assert pred_tokens.shape == (batch_size, token_dim, h, w), (
        f"Expected shape ({batch_size}, {token_dim}, {h}, {w}), got {pred_tokens.shape}"
    )

    print("✓ Diffusion head shape test passed")


def test_noise_schedule():
    """Test that noise schedule varies with timestep."""
    model = BinaryDiffusionHead(token_dim=16, hidden_dim=64, num_timesteps=1000)
    model.eval()

    tokens = torch.randn(2, 16, 4, 4)

    # At different timesteps, noise level should vary
    with torch.no_grad():
        noisy_early = model.add_noise(tokens, torch.tensor([10]), noise_scale=1.0)
        noisy_late = model.add_noise(tokens, torch.tensor([900]), noise_scale=1.0)

    # Late timesteps should have more accumulated noise (larger deviation from original)
    # Measure this by the norm of the difference
    diff_early = (noisy_early - tokens).norm()
    diff_late = (noisy_late - tokens).norm()

    assert diff_late > diff_early, (
        f"Later timesteps should have more noise. "
        f"Early diff: {diff_early:.4f}, Late diff: {diff_late:.4f}"
    )

    print(f"✓ Noise schedule test passed (early diff: {diff_early:.4f}, late diff: {diff_late:.4f})")


def test_diffusion_loss_computation():
    """Test that diffusion loss can be computed and backpropagated."""
    model = BinaryDiffusionHead(token_dim=32, hidden_dim=128, num_timesteps=1000)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 4
    clean_tokens = torch.randint(0, 2, (batch_size, 32, 4, 4)).float()
    t = torch.randint(0, 1000, (batch_size,))

    # Compute loss
    loss = model.diffusion_loss(clean_tokens, t, noise_scale=1.0)

    # Check that loss is a scalar
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✓ Diffusion loss test passed (loss: {loss.item():.6f})")


def test_diffusion_training():
    """Test that diffusion head can be trained to denoise tokens."""
    model = BinaryDiffusionHead(token_dim=16, hidden_dim=64, num_timesteps=1000)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create clean binary tokens (pattern: checkerboard)
    clean_tokens = torch.zeros(4, 16, 4, 4)
    clean_tokens[:, :, ::2, ::2] = 1.0
    clean_tokens[:, :, 1::2, 1::2] = 1.0

    initial_loss = None
    final_loss = None

    # Train for a few steps
    for step in range(30):
        optimizer.zero_grad()

        # Random timesteps
        t = torch.randint(100, 900, (4,))

        # Compute loss
        loss = model.diffusion_loss(clean_tokens, t, noise_scale=1.0)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 29:
            final_loss = loss.item()

    # Check that loss decreased
    assert final_loss < initial_loss, (
        f"Loss should decrease. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
    )

    print(f"✓ Diffusion training test passed (loss: {initial_loss:.6f} -> {final_loss:.6f})")


def test_sampler():
    """Test that the sampler can generate tokens via reverse diffusion."""
    model = BinaryDiffusionHead(token_dim=16, hidden_dim=64, num_timesteps=1000)
    model.eval()

    sampler = BinaryDiffusionSampler(model, num_timesteps=1000, noise_scale=1.0)

    # Generate tokens
    shape = (2, 16, 4, 4)
    with torch.no_grad():
        tokens = sampler.sample(shape, num_steps=20, device="cpu")

    # Check shape and range
    assert tokens.shape == shape, f"Expected shape {shape}, got {tokens.shape}"
    assert tokens.min() >= 0 and tokens.max() <= 1, (
        f"Generated tokens should be binary (0 or 1), got range [{tokens.min()}, {tokens.max()}]"
    )

    # Check that tokens are actually binary
    unique_vals = torch.unique(tokens)
    assert len(unique_vals) <= 2, f"Should be binary, got {len(unique_vals)} unique values"
    assert torch.all((unique_vals == 0) | (unique_vals == 1)), (
        f"Unique values should be 0 or 1, got {unique_vals}"
    )

    print("✓ Sampler test passed")


def test_multi_scale_tokens():
    """Test diffusion head with different spatial resolutions."""
    for h, w in [(2, 2), (4, 4), (8, 8)]:
        model = BinaryDiffusionHead(token_dim=32, hidden_dim=128, num_timesteps=1000)
        model.eval()

        noisy_tokens = torch.randn(1, 32, h, w)
        t = torch.tensor([500])

        with torch.no_grad():
            pred_tokens = model(noisy_tokens, t)

        assert pred_tokens.shape == (1, 32, h, w), (
            f"Expected shape (1, 32, {h}, {w}), got {pred_tokens.shape}"
        )

    print("✓ Multi-scale tokens test passed")


if __name__ == "__main__":
    test_positional_encoding()
    test_diffusion_head_shapes()
    test_noise_schedule()
    test_diffusion_loss_computation()
    test_diffusion_training()
    test_sampler()
    test_multi_scale_tokens()
    print("\n✅ All binary diffusion tests passed!")
