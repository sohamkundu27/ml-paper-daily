"""
Tests for Pass 1: Multi-scale tokenizer and basic transformer core.
"""

import torch
from var_pass1 import (
    HierarchicalTokenizer,
    tokenize_to_sequence,
    VARTransformerLayer,
    VARPass1,
)


def test_hierarchical_tokenizer():
    """Test that tokenizer produces correct hierarchical scales."""
    tokenizer = HierarchicalTokenizer(in_channels=3, token_dim=256, num_scales=3)

    # Create a random image: batch_size=2, channels=3, height=64, width=64
    x = torch.randn(2, 3, 64, 64)

    token_maps = tokenizer(x)

    # Check number of scales
    assert len(token_maps) == 3, f"Expected 3 scales, got {len(token_maps)}"

    # Check shapes: each scale should be half the resolution of the previous
    expected_shapes = [
        (2, 256, 32, 32),  # 64 -> 32
        (2, 256, 16, 16),  # 32 -> 16
        (2, 256, 8, 8),    # 16 -> 8
    ]

    for i, (token_map, expected_shape) in enumerate(zip(token_maps, expected_shapes)):
        assert token_map.shape == expected_shape, (
            f"Scale {i}: expected shape {expected_shape}, got {token_map.shape}"
        )

    print("✓ Hierarchical tokenizer produces correct shapes")


def test_tokenize_to_sequence():
    """Test that tokenization to sequence preserves information."""
    token_maps = [
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 256, 8, 8),
    ]

    token_sequence, scale_indices = tokenize_to_sequence(token_maps)

    # Check sequence shape
    batch_size = 2
    num_tokens = 32*32 + 16*16 + 8*8  # 1024 + 256 + 64 = 1344
    assert token_sequence.shape == (batch_size, num_tokens, 256), (
        f"Expected shape (2, {num_tokens}, 256), got {token_sequence.shape}"
    )

    # Check scale indices
    expected_indices_len = num_tokens
    assert len(scale_indices) == expected_indices_len, (
        f"Expected {expected_indices_len} scale indices, got {len(scale_indices)}"
    )

    # Verify scale indices are correct
    assert (scale_indices[:1024] == 0).all(), "First 1024 tokens should belong to scale 0"
    assert (scale_indices[1024:1280] == 1).all(), "Next 256 tokens should belong to scale 1"
    assert (scale_indices[1280:] == 2).all(), "Last 64 tokens should belong to scale 2"

    print("✓ Tokenization to sequence works correctly")


def test_var_transformer_layer():
    """Test that transformer layer produces correct output shape."""
    transformer = VARTransformerLayer(
        token_dim=256, num_heads=8, ff_dim=1024, vocab_size=4096
    )

    # Random token sequence: (batch_size=2, num_tokens=1344, token_dim=256)
    x = torch.randn(2, 1344, 256)

    logits, hidden = transformer(x)

    # Check logits shape
    assert logits.shape == (2, 1344, 4096), (
        f"Expected logits shape (2, 1344, 4096), got {logits.shape}"
    )

    # Check hidden state shape
    assert hidden.shape == (2, 1344, 256), (
        f"Expected hidden shape (2, 1344, 256), got {hidden.shape}"
    )

    print("✓ Transformer layer produces correct shapes")


def test_var_pass1_end_to_end():
    """Test end-to-end forward pass of Pass 1 model."""
    model = VARPass1(
        in_channels=3, token_dim=256, num_scales=3,
        num_heads=8, ff_dim=1024, vocab_size=4096
    )

    # Random image
    x = torch.randn(2, 3, 64, 64)

    logits, token_maps = model(x)

    # Check logits
    assert logits.shape[0] == 2, "Batch size should be 2"
    assert logits.shape[2] == 4096, "Vocab size should be 4096"

    # Check token maps
    assert len(token_maps) == 3, "Should have 3 scales"

    # Verify we can compute a loss (e.g., dummy target)
    target = torch.randint(0, 4096, (2, logits.shape[1]))
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.reshape(-1, 4096), target.reshape(-1))
    assert loss.item() > 0, "Loss should be positive"

    print("✓ End-to-end forward pass works and loss computes")


def test_var_pass1_gradient_flow():
    """Test that gradients flow through the model."""
    model = VARPass1(
        in_channels=3, token_dim=256, num_scales=3,
        num_heads=8, ff_dim=1024, vocab_size=4096
    )

    # Random image and target
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randint(0, 4096, (2, 1344))

    logits, _ = model(x)

    # Compute loss and backprop
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.reshape(-1, 4096), target.reshape(-1))
    loss.backward()

    # Check that gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"

    print("✓ Gradients flow through the model")


if __name__ == "__main__":
    test_hierarchical_tokenizer()
    test_tokenize_to_sequence()
    test_var_transformer_layer()
    test_var_pass1_end_to_end()
    test_var_pass1_gradient_flow()
    print("\n✅ All tests passed!")
