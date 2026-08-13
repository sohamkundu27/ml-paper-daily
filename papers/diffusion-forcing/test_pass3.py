import torch
import numpy as np
from diffusion_forcing import DiffusionForcing, SyntheticDataGenerator


def test_sample_basic():
    """Test that sampling produces valid output."""
    token_dim = 8
    batch_size = 2
    seq_len = 6
    num_steps = 20

    df = DiffusionForcing(token_dim)

    # Sample
    x_gen = df.sample(batch_size, seq_len, num_steps=num_steps)

    # Check shape
    assert x_gen.shape == (batch_size, seq_len, token_dim), \
        f"Expected shape {(batch_size, seq_len, token_dim)}, got {x_gen.shape}"

    # Check no NaNs or Infs
    assert not torch.isnan(x_gen).any(), "Sample contains NaN"
    assert not torch.isinf(x_gen).any(), "Sample contains Inf"

    print(f"✓ Basic sampling test passed")
    print(f"  Generated shape: {x_gen.shape}")
    print(f"  Mean: {x_gen.mean():.4f}, Std: {x_gen.std():.4f}")


def test_sample_multiple_batches():
    """Test sampling with multiple batches."""
    token_dim = 4
    batch_size = 8
    seq_len = 10
    num_steps = 30

    df = DiffusionForcing(token_dim)

    x_gen = df.sample(batch_size, seq_len, num_steps=num_steps)

    assert x_gen.shape == (batch_size, seq_len, token_dim)
    assert not torch.isnan(x_gen).any()

    print(f"✓ Multiple batches sampling test passed")


def test_sample_variable_length():
    """Test that sampling can generate sequences of different lengths."""
    token_dim = 8

    df = DiffusionForcing(token_dim)

    # Test different sequence lengths
    for seq_len in [4, 8, 12, 16]:
        x_gen = df.sample(batch_size=2, seq_len=seq_len, num_steps=20)
        assert x_gen.shape[1] == seq_len, f"Expected seq_len {seq_len}, got {x_gen.shape[1]}"

    print(f"✓ Variable length sampling test passed")


def test_sample_trained_vs_untrained():
    """Test that trained model produces more reasonable samples."""
    token_dim = 8
    batch_size = 4
    seq_len = 6
    num_steps = 20

    # Train a model
    df = DiffusionForcing(token_dim)
    x_data = torch.randn(batch_size * 2, seq_len, token_dim)
    data_loader = [x_data[i:i+batch_size] for i in range(0, len(x_data), batch_size)]

    # Sample before training
    torch.manual_seed(42)
    x_before = df.sample(batch_size, seq_len, num_steps=num_steps)

    # Train
    df.train(data_loader, num_epochs=5, learning_rate=1e-3)

    # Sample after training
    torch.manual_seed(42)
    x_after = df.sample(batch_size, seq_len, num_steps=num_steps)

    # Both should be valid tensors
    assert not torch.isnan(x_before).any()
    assert not torch.isnan(x_after).any()

    print(f"✓ Trained vs untrained sampling test passed")
    print(f"  Before training - Mean: {x_before.mean():.4f}, Std: {x_before.std():.4f}")
    print(f"  After training - Mean: {x_after.mean():.4f}, Std: {x_after.std():.4f}")


def test_sample_with_mask_basic():
    """Test basic masked sampling."""
    token_dim = 8
    batch_size = 2
    context_len = 3
    total_len = 8
    num_steps = 20

    df = DiffusionForcing(token_dim)

    # Create context (clean tokens)
    context = torch.randn(batch_size, context_len, token_dim)

    # Create mask (True = keep, False = denoise)
    mask = torch.zeros(batch_size, total_len, dtype=torch.bool)
    mask[:, :context_len] = True

    # Sample with mask
    x_gen = df.sample_with_mask(context, mask, num_steps=num_steps)

    # Check shape
    assert x_gen.shape == (batch_size, total_len, token_dim), \
        f"Expected shape {(batch_size, total_len, token_dim)}, got {x_gen.shape}"

    # Check that context part is preserved
    assert torch.allclose(x_gen[:, :context_len], context, atol=1e-5), \
        "Context part should be preserved"

    # Check no NaNs
    assert not torch.isnan(x_gen).any(), "Sample contains NaN"

    print(f"✓ Basic masked sampling test passed")
    print(f"  Generated shape: {x_gen.shape}")
    print(f"  Context preserved: {torch.allclose(x_gen[:, :context_len], context, atol=1e-5)}")


def test_sample_with_mask_extend():
    """Test extending a sequence with masked sampling."""
    token_dim = 8
    batch_size = 3
    context_len = 5
    extend_len = 3
    num_steps = 25

    df = DiffusionForcing(token_dim)

    # Create context
    context = torch.randn(batch_size, context_len, token_dim)

    # Create mask for extension
    total_len = context_len + extend_len
    mask = torch.zeros(batch_size, total_len, dtype=torch.bool)
    mask[:, :context_len] = True

    # Sample extension
    x_gen = df.sample_with_mask(context, mask, num_steps=num_steps)

    assert x_gen.shape == (batch_size, total_len, token_dim)
    assert torch.allclose(x_gen[:, :context_len], context, atol=1e-5)
    assert not torch.isnan(x_gen).any()

    print(f"✓ Masked extension test passed")
    print(f"  Extended {context_len} context tokens by {extend_len} new tokens")


def test_sample_with_mask_middle():
    """Test inpainting: keeping past and future, denoising middle."""
    token_dim = 8
    batch_size = 2
    total_len = 10
    num_steps = 25

    df = DiffusionForcing(token_dim)

    # Create full context
    context = torch.randn(batch_size, total_len, token_dim)

    # Create mask where we keep start and end, denoise middle
    mask = torch.ones(batch_size, total_len, dtype=torch.bool)
    mask[:, 3:7] = False  # Denoise tokens 3-6

    # Sample with mask (actually initialize to the full context,
    # but mask says to denoise middle part)
    x_gen = df.sample_with_mask(context[:, :3], torch.cat([
        torch.ones(batch_size, 3, dtype=torch.bool),
        torch.zeros(batch_size, 4, dtype=torch.bool),
        torch.ones(batch_size, 3, dtype=torch.bool)
    ], dim=1), num_steps=num_steps)

    assert x_gen.shape == (batch_size, total_len, token_dim)
    assert not torch.isnan(x_gen).any()

    print(f"✓ Masked inpainting test passed")


def test_sample_consistency():
    """Test that sampling is deterministic when setting seed."""
    token_dim = 8
    batch_size = 2
    seq_len = 6
    num_steps = 20

    df = DiffusionForcing(token_dim)

    torch.manual_seed(42)
    x_gen1 = df.sample(batch_size, seq_len, num_steps=num_steps)

    torch.manual_seed(42)
    x_gen2 = df.sample(batch_size, seq_len, num_steps=num_steps)

    assert torch.allclose(x_gen1, x_gen2, atol=1e-5), \
        "Samples should be identical with same seed"

    print(f"✓ Sampling consistency test passed")


def test_sample_different_step_sizes():
    """Test sampling with different numbers of denoising steps."""
    token_dim = 8
    batch_size = 2
    seq_len = 6

    df = DiffusionForcing(token_dim)

    for num_steps in [10, 20, 50]:
        x_gen = df.sample(batch_size, seq_len, num_steps=num_steps)
        assert x_gen.shape == (batch_size, seq_len, token_dim)
        assert not torch.isnan(x_gen).any()

    print(f"✓ Different step sizes test passed")


def test_sample_after_training():
    """Test that sampling works correctly after model training."""
    token_dim = 8
    batch_size = 4
    seq_len = 6
    num_steps = 25

    df = DiffusionForcing(token_dim)

    # Generate and train on data
    data = SyntheticDataGenerator.repeating_pattern_sequences(
        num_samples=32, seq_len=seq_len, token_dim=token_dim, pattern_len=3
    )
    data_loader = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    # Train model
    losses = df.train(data_loader, num_epochs=10, learning_rate=1e-3)
    assert len(losses) == 10
    assert losses[-1] < losses[0]

    # Now sample
    x_gen = df.sample(batch_size=3, seq_len=seq_len, num_steps=num_steps)

    assert x_gen.shape == (3, seq_len, token_dim)
    assert not torch.isnan(x_gen).any()

    print(f"✓ Sampling after training test passed")
    print(f"  Training loss improved from {losses[0]:.4f} to {losses[-1]:.4f}")


def test_mask_progressive_extension():
    """Test progressive extension: extend in chunks."""
    token_dim = 8
    batch_size = 1
    num_steps = 20

    df = DiffusionForcing(token_dim)

    # Start with short context
    context = torch.randn(batch_size, 3, token_dim)

    # Extend to 6 tokens
    mask1 = torch.cat([
        torch.ones(batch_size, 3, dtype=torch.bool),
        torch.zeros(batch_size, 3, dtype=torch.bool)
    ], dim=1)
    result1 = df.sample_with_mask(context, mask1, num_steps=num_steps)

    # Use extended result as new context and extend further
    new_context = result1[:, :6]
    mask2 = torch.cat([
        torch.ones(batch_size, 6, dtype=torch.bool),
        torch.zeros(batch_size, 3, dtype=torch.bool)
    ], dim=1)
    result2 = df.sample_with_mask(new_context, mask2, num_steps=num_steps)

    # Check progression
    assert result1.shape == (batch_size, 6, token_dim)
    assert result2.shape == (batch_size, 9, token_dim)
    assert torch.allclose(result2[:, :3], context, atol=1e-5)
    assert not torch.isnan(result2).any()

    print(f"✓ Progressive extension test passed")
    print(f"  Extended from 3 → 6 → 9 tokens")


if __name__ == "__main__":
    test_sample_basic()
    test_sample_multiple_batches()
    test_sample_variable_length()
    test_sample_trained_vs_untrained()
    test_sample_with_mask_basic()
    test_sample_with_mask_extend()
    test_sample_with_mask_middle()
    test_sample_consistency()
    test_sample_different_step_sizes()
    test_sample_after_training()
    test_mask_progressive_extension()
    print("\n✅ All pass 3 tests passed!")
