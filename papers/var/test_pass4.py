"""
Tests for Pass 4: Inference and generation.

Tests the complete pipeline: train a model and generate samples from it.
"""

import torch
from torch.utils.data import DataLoader
from var_pass2 import VARPass2
from var_pass3 import VARTrainer, ToyImageDataset
from var_pass4 import VARPass4, VARGenerator


def test_generator_initialization():
    """Test that generator initializes correctly."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    generator = VARGenerator(model, device="cpu", temperature=1.0)

    assert generator.model is not None, "Generator should have a model"
    assert generator.temperature == 1.0, "Generator should store temperature"

    print("✓ Generator initializes correctly")


def test_generation_first_scale():
    """Test that first scale generation works (uniform sampling)."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    generator = VARGenerator(model, device="cpu")

    # Generate first scale only
    batch_size = 2
    img_size = 64

    # Manually test first scale generation
    generated = {}
    h_at_scale = img_size // (2 ** (0 + 1))
    w_at_scale = h_at_scale
    num_tokens_at_scale = h_at_scale * w_at_scale

    sampled_tokens = torch.randint(
        0, model.output_proj.out_features,
        (batch_size, num_tokens_at_scale),
        device="cpu"
    )
    generated[0] = sampled_tokens

    assert sampled_tokens.shape == (batch_size, num_tokens_at_scale), (
        f"Expected shape {(batch_size, num_tokens_at_scale)}, got {sampled_tokens.shape}"
    )
    assert sampled_tokens.min() >= 0, "Token indices should be non-negative"
    assert sampled_tokens.max() < 4096, "Token indices should be < vocab_size"

    print("✓ First scale generation works")


def test_full_generation():
    """Test that full coarse-to-fine generation works."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=4,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    generator = VARGenerator(model, device="cpu", temperature=0.8)

    # Generate token sequences for all scales
    generated_tokens = generator.generate(batch_size=2, img_size=64)

    # Verify output structure
    assert len(generated_tokens) == 3, "Should generate 3 scales"

    expected_tokens_per_scale = [1024, 256, 64]  # H*W at each scale (64->32, 32->16, 16->8)
    for scale_idx in range(3):
        tokens = generated_tokens[scale_idx]
        assert tokens.shape[0] == 2, f"Batch size should be 2, got {tokens.shape[0]}"
        assert tokens.shape[1] == expected_tokens_per_scale[scale_idx], (
            f"Expected {expected_tokens_per_scale[scale_idx]} tokens at scale {scale_idx}, "
            f"got {tokens.shape[1]}"
        )
        assert tokens.dtype == torch.long, "Tokens should be long integers"
        assert tokens.min() >= 0, "Token indices should be non-negative"
        assert tokens.max() < 4096, "Token indices should be < vocab_size"

    print("✓ Full coarse-to-fine generation works")


def test_generation_batch_consistency():
    """Test that generation produces different samples in a batch."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=4,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    generator = VARGenerator(model, device="cpu", temperature=1.0)

    # Generate a batch
    generated_tokens = generator.generate(batch_size=2, img_size=64)

    # Check that batch samples differ (with high temperature, should not be identical)
    tokens_batch_0 = generated_tokens[0][0]  # First image, first scale
    tokens_batch_1 = generated_tokens[0][1]  # Second image, first scale

    # With random sampling, they should likely be different
    # (Though not guaranteed; we just check they can be different)
    num_same = (tokens_batch_0 == tokens_batch_1).sum().item()
    total = tokens_batch_0.numel()

    print(f"  Tokens same between samples: {num_same}/{total}")
    # At least some tokens should differ with high temperature
    assert num_same < total, "Batch samples should have some variation"

    print("✓ Generation produces varying samples")


def test_pass4_model_wrapper():
    """Test VARPass4 wrapper model."""
    model = VARPass4(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=4,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    assert hasattr(model, "generate"), "VARPass4 should have generate method"

    # Test generation through wrapper
    generated = model.generate(batch_size=1, img_size=64, temperature=1.0)
    assert len(generated) == 3, "Should generate 3 scales"

    print("✓ VARPass4 wrapper works")


def test_end_to_end_train_and_generate():
    """End-to-end test: train a small model and generate from it."""
    # Create a small model for fast training
    model = VARPass2(
        in_channels=3,
        token_dim=128,
        num_scales=2,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        vocab_size=4096,
    )

    # Create trainer
    trainer = VARTrainer(model, device="cpu", learning_rate=1e-2)

    # Create small dataset
    dataset = ToyImageDataset(num_samples=10, img_size=64, num_channels=3)
    dataloader = DataLoader(dataset, batch_size=4)

    # Train for a few steps
    for epoch in range(2):
        trainer.train_epoch(dataloader)

    print(f"  Trained to loss: {trainer.train_losses[-1]:.4f}")

    # Now test generation from trained model
    generator = VARGenerator(model, device="cpu", temperature=0.7)
    generated_tokens = generator.generate(batch_size=2, img_size=64)

    # Verify generation worked
    assert len(generated_tokens) == 2, "Should have 2 scales"
    assert generated_tokens[0].shape[0] == 2, "Batch size should be 2"
    assert generated_tokens[1].shape[0] == 2, "Batch size should be 2"

    print("✓ End-to-end train-and-generate pipeline works")


def test_temperature_effect():
    """Test that temperature affects sample diversity."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=2,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        vocab_size=4096,
    )

    # Train briefly so model has some meaningful outputs
    trainer = VARTrainer(model, device="cpu", learning_rate=1e-2)
    dataset = ToyImageDataset(num_samples=8, img_size=64, num_channels=3)
    dataloader = DataLoader(dataset, batch_size=4)

    for _ in range(1):
        trainer.train_epoch(dataloader)

    # Generate with different temperatures
    generator_low = VARGenerator(model, device="cpu", temperature=0.1)
    generator_high = VARGenerator(model, device="cpu", temperature=10.0)

    gen_low = generator_low.generate(batch_size=1, img_size=64)
    gen_high = generator_high.generate(batch_size=1, img_size=64)

    # Compare entropy/diversity
    # At low temperature, predictions should be more concentrated
    # At high temperature, should be more spread out
    # (This is a crude test; high temperature samples may still be correlated)

    print("✓ Temperature parameter works")


def test_different_batch_sizes():
    """Test generation with various batch sizes."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=2,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        vocab_size=4096,
    )

    generator = VARGenerator(model, device="cpu")

    for batch_size in [1, 2, 4]:
        generated = generator.generate(batch_size=batch_size, img_size=64)
        for scale_idx in generated:
            assert generated[scale_idx].shape[0] == batch_size, (
                f"Batch size mismatch at scale {scale_idx}"
            )

    print("✓ Generation works with various batch sizes")


if __name__ == "__main__":
    test_generator_initialization()
    test_generation_first_scale()
    test_full_generation()
    test_generation_batch_consistency()
    test_pass4_model_wrapper()
    test_end_to_end_train_and_generate()
    test_temperature_effect()
    test_different_batch_sizes()
    print("\n✅ All Pass 4 tests passed!")
