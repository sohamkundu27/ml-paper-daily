"""Test the autoregressive token predictor."""

import torch
import torch.optim as optim
from autoregressive_predictor import (
    AutoregressiveTokenPredictor,
    MultiScaleTokenGenerator,
    TokenEmbedding,
    AutoregressiveLoss,
)


def test_token_embedding():
    """Test token embedding layer."""
    token_dim = 32
    embed_dim = 128
    embed = TokenEmbedding(token_dim, embed_dim)

    # Test forward pass
    tokens = torch.randn(2, 10, token_dim)
    embeds = embed(tokens)

    assert embeds.shape == (2, 10, embed_dim), f"Expected shape (2, 10, {embed_dim}), got {embeds.shape}"

    print("✓ Token embedding test passed")


def test_autoregressive_predictor_shapes():
    """Test that autoregressive predictor produces expected shapes."""
    predictor = AutoregressiveTokenPredictor(
        token_dim=32, embed_dim=128, num_heads=4, depth=2, num_timesteps=1000
    )
    predictor.eval()

    batch_size = 2
    seq_len = 16
    token_dim = 32

    # Create dummy token sequence
    tokens = torch.randn(batch_size, seq_len, token_dim)

    with torch.no_grad():
        pred_tokens = predictor(tokens)

    assert pred_tokens.shape == (batch_size, seq_len, token_dim), (
        f"Expected shape ({batch_size}, {seq_len}, {token_dim}), got {pred_tokens.shape}"
    )

    print("✓ Autoregressive predictor shape test passed")


def test_autoregressive_with_diffusion():
    """Test autoregressive predictor with diffusion refinement."""
    predictor = AutoregressiveTokenPredictor(
        token_dim=32, embed_dim=128, num_heads=4, depth=2, num_timesteps=1000
    )
    predictor.eval()

    batch_size = 2
    seq_len = 16
    token_dim = 32

    tokens = torch.randn(batch_size, seq_len, token_dim)
    t = 500

    with torch.no_grad():
        pred_tokens = predictor.forward_with_diffusion(tokens, t=t)

    assert pred_tokens.shape == (batch_size, seq_len, token_dim), (
        f"Expected shape ({batch_size}, {seq_len}, {token_dim}), got {pred_tokens.shape}"
    )

    print("✓ Autoregressive with diffusion test passed")


def test_autoregressive_training():
    """Test that autoregressive predictor can be trained."""
    predictor = AutoregressiveTokenPredictor(
        token_dim=16, embed_dim=64, num_heads=2, depth=1, num_timesteps=1000
    )
    loss_fn = AutoregressiveLoss(token_pred_weight=1.0)
    optimizer = optim.Adam(predictor.parameters(), lr=0.001)

    # Create simple synthetic tokens: checkerboard pattern
    batch_size = 4
    seq_len = 8
    target_tokens = torch.zeros(batch_size, seq_len, 16)
    target_tokens[:, ::2, :] = 1.0

    initial_losses = []
    final_losses = []

    # Train for a few steps
    for step in range(20):
        optimizer.zero_grad()

        # Add noise to target for input
        noisy_tokens = target_tokens + 0.1 * torch.randn_like(target_tokens)

        # Forward pass
        pred_tokens = predictor(noisy_tokens)

        # Compute loss
        loss = loss_fn(pred_tokens, target_tokens)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_losses.append(loss.item())
        if step == 19:
            final_losses.append(loss.item())

    # Check that loss decreased
    assert final_losses[0] < initial_losses[0], (
        f"Loss should decrease. Initial: {initial_losses[0]:.6f}, Final: {final_losses[0]:.6f}"
    )

    print(f"✓ Autoregressive training test passed (loss: {initial_losses[0]:.6f} -> {final_losses[0]:.6f})")


def test_autoregressive_generation():
    """Test autoregressive token generation."""
    predictor = AutoregressiveTokenPredictor(
        token_dim=16, embed_dim=64, num_heads=2, depth=1, num_timesteps=1000
    )
    predictor.eval()

    generator = MultiScaleTokenGenerator(predictor, num_scales=1)

    batch_size = 2
    seq_len = 10
    token_dim = 16

    with torch.no_grad():
        tokens = generator.generate_autoregressive(batch_size, seq_len, token_dim, device="cpu")

    assert tokens.shape == (batch_size, seq_len, token_dim), (
        f"Expected shape ({batch_size}, {seq_len}, {token_dim}), got {tokens.shape}"
    )

    # Check that tokens are in reasonable range (can exceed [-2, 2] slightly due to random init)
    assert tokens.min() >= -3 and tokens.max() <= 3, (
        f"Generated tokens should be in reasonable range, got [{tokens.min()}, {tokens.max()}]"
    )

    print("✓ Autoregressive generation test passed")


def test_multiscale_generator():
    """Test multi-scale token generation."""
    predictor = AutoregressiveTokenPredictor(
        token_dim=32, embed_dim=128, num_heads=4, depth=2, num_timesteps=1000
    )
    predictor.eval()

    generator = MultiScaleTokenGenerator(predictor, num_scales=2)

    # Create initial token grid at coarsest scale (4x4)
    initial_tokens = torch.randn(2, 32, 4, 4)

    with torch.no_grad():
        refined_tokens = generator.generate(initial_tokens, scales=[4, 8], device="cpu")

    assert refined_tokens.shape == (2, 32, 4, 4), (
        f"Expected shape (2, 32, 4, 4), got {refined_tokens.shape}"
    )

    print("✓ Multi-scale generator test passed")


def test_caching():
    """Test that caching mechanism works."""
    predictor = AutoregressiveTokenPredictor(
        token_dim=16, embed_dim=64, num_heads=2, depth=1, num_timesteps=1000
    )
    predictor.eval()

    # Set cache
    dummy_cache = torch.randn(2, 8, 64)
    dummy_pos = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    predictor.set_cache(dummy_cache, dummy_pos)

    assert predictor.cache_embeds is not None, "Cache should be set"
    assert predictor.cache_pos is not None, "Position cache should be set"

    # Clear cache
    predictor.clear_cache()

    assert predictor.cache_embeds is None, "Cache should be cleared"
    assert predictor.cache_pos is None, "Position cache should be cleared"

    print("✓ Caching test passed")


def test_generation_with_caching():
    """Test token generation using caching."""
    predictor = AutoregressiveTokenPredictor(
        token_dim=16, embed_dim=64, num_heads=2, depth=1, num_timesteps=1000
    )
    predictor.eval()

    generator = MultiScaleTokenGenerator(predictor, num_scales=1)

    # Start with initial sequence
    initial_tokens = torch.randn(2, 5, 16)

    with torch.no_grad():
        extended_tokens = generator.generate_with_caching(initial_tokens, generate_steps=5, device="cpu")

    assert extended_tokens.shape == (2, 10, 16), (
        f"Expected shape (2, 10, 16), got {extended_tokens.shape}"
    )

    # First 5 tokens should be from initial
    assert torch.allclose(extended_tokens[:, :5, :], initial_tokens, atol=1e-5), (
        "Initial tokens should be preserved"
    )

    print("✓ Generation with caching test passed")


def test_autoregressive_loss():
    """Test autoregressive loss computation."""
    loss_fn = AutoregressiveLoss(token_pred_weight=1.0)

    batch_size = 4
    seq_len = 8
    token_dim = 16

    pred_tokens = torch.randn(batch_size, seq_len, token_dim)
    target_tokens = torch.randint(0, 2, (batch_size, seq_len, token_dim)).float()

    # Compute loss
    loss = loss_fn(pred_tokens, target_tokens)

    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"

    print(f"✓ Autoregressive loss test passed (loss: {loss.item():.6f})")


def test_loss_scaling():
    """Test that loss scales appropriately with predictions."""
    loss_fn = AutoregressiveLoss(token_pred_weight=1.0)

    batch_size = 4
    seq_len = 8
    token_dim = 16

    # Good predictions (close to target)
    target = torch.randint(0, 2, (batch_size, seq_len, token_dim)).float()
    pred_good = target + 0.1 * torch.randn_like(target)
    loss_good = loss_fn(pred_good, target)

    # Bad predictions (far from target)
    pred_bad = torch.randn_like(target)
    loss_bad = loss_fn(pred_bad, target)

    assert loss_bad > loss_good, (
        f"Bad predictions should have higher loss. "
        f"Good: {loss_good.item():.6f}, Bad: {loss_bad.item():.6f}"
    )

    print(f"✓ Loss scaling test passed (good: {loss_good.item():.6f}, bad: {loss_bad.item():.6f})")


if __name__ == "__main__":
    test_token_embedding()
    test_autoregressive_predictor_shapes()
    test_autoregressive_with_diffusion()
    test_autoregressive_training()
    test_autoregressive_generation()
    test_multiscale_generator()
    test_caching()
    test_generation_with_caching()
    test_autoregressive_loss()
    test_loss_scaling()
    print("\n✅ All autoregressive predictor tests passed!")
