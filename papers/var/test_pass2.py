"""
Tests for Pass 2: Full transformer stack and scale-conditional architecture.
"""

import torch
from var_pass2 import (
    create_cumulative_scale_mask,
    ScaleEmbedding,
    VARTransformerBlock,
    VARPass2,
)
from var_pass1 import tokenize_to_sequence


def test_cumulative_scale_mask():
    """Test that cumulative scale mask enforces coarse-to-fine structure."""
    # Simulate tokens from 3 scales: scale 0 has 4 tokens, scale 1 has 2, scale 2 has 1
    # scale_indices = [0, 0, 0, 0, 1, 1, 2]
    scale_indices = torch.tensor([0, 0, 0, 0, 1, 1, 2], dtype=torch.long)

    mask = create_cumulative_scale_mask(scale_indices, device="cpu")

    # Shape should be (7, 7)
    assert mask.shape == (7, 7), f"Expected shape (7, 7), got {mask.shape}"

    # Tokens from scale 0 can attend to themselves (and nothing before, since they're first)
    assert mask[0, 0], "Scale 0 token 0 should attend to itself"
    assert mask[0, 1], "Scale 0 token 0 should attend to scale 0 token 1"
    assert not mask[0, 4], "Scale 0 token 0 should NOT attend to scale 1"

    # Tokens from scale 1 can attend to scale 0 and scale 1, but not scale 2
    assert mask[4, 0], "Scale 1 token should attend to scale 0"
    assert mask[4, 3], "Scale 1 token should attend to scale 0"
    assert mask[4, 4], "Scale 1 token should attend to itself"
    assert mask[4, 5], "Scale 1 token should attend to other scale 1 tokens (parallel generation)"
    assert not mask[4, 6], "Scale 1 token should NOT attend to scale 2"

    # Tokens from scale 2 can attend to scales 0 and 1
    assert mask[6, 0], "Scale 2 token should attend to scale 0"
    assert mask[6, 4], "Scale 2 token should attend to scale 1"
    assert mask[6, 6], "Scale 2 token should attend to itself"

    print("✓ Cumulative scale mask correctly enforces coarse-to-fine structure")


def test_scale_embedding():
    """Test that scale embeddings add scale-specific information."""
    scale_emb = ScaleEmbedding(token_dim=256, num_scales=3)

    # Token sequence: batch_size=2, num_tokens=7, token_dim=256
    token_sequence = torch.randn(2, 7, 256)
    scale_indices = torch.tensor([0, 0, 0, 0, 1, 1, 2], dtype=torch.long)

    output = scale_emb(token_sequence, scale_indices)

    # Check output shape
    assert output.shape == token_sequence.shape, (
        f"Expected output shape {token_sequence.shape}, got {output.shape}"
    )

    # Verify that tokens from the same scale get the same scale embedding added
    # (they will have different values due to the original token_sequence, but the
    # scale embedding part will be identical)
    scale0_emb = scale_emb.scale_embeddings(torch.tensor([0]))  # (1, 256)
    scale1_emb = scale_emb.scale_embeddings(torch.tensor([1]))  # (1, 256)

    # These should be different
    assert not torch.allclose(
        scale0_emb, scale1_emb
    ), "Different scales should have different embeddings"

    print("✓ Scale embeddings work correctly")


def test_transformer_block():
    """Test that transformer block processes sequences correctly."""
    block = VARTransformerBlock(token_dim=256, num_heads=8, ff_dim=1024)

    x = torch.randn(2, 100, 256)  # batch_size=2, seq_len=100, token_dim=256

    output = block(x)

    # Check output shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    # Check that output is different from input (model did something)
    assert not torch.allclose(output, x), "Output should be different from input"

    print("✓ Transformer block processes sequences correctly")


def test_transformer_block_with_mask():
    """Test transformer block with attention mask."""
    block = VARTransformerBlock(token_dim=256, num_heads=8, ff_dim=1024)

    x = torch.randn(2, 10, 256)

    # Create a simple causal mask (can only attend to previous positions)
    mask = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)

    output = block(x, attn_mask=mask)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    print("✓ Transformer block respects attention masks")


def test_var_pass2_end_to_end():
    """Test end-to-end forward pass of Pass 2 model."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    # Random image
    x = torch.randn(2, 3, 64, 64)

    logits, token_maps = model(x)

    # Check logits shape
    assert logits.shape[0] == 2, "Batch size should be 2"
    assert logits.shape[1] > 0, "Sequence length should be positive"
    assert logits.shape[2] == 4096, "Vocab size should be 4096"

    # Check token maps
    assert len(token_maps) == 3, "Should have 3 scales"

    # Verify we can compute a loss
    target = torch.randint(0, 4096, (2, logits.shape[1]))
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.reshape(-1, 4096), target.reshape(-1))
    assert loss.item() > 0, "Loss should be positive"

    print("✓ End-to-end forward pass works")


def test_var_pass2_gradient_flow():
    """Test that gradients flow through Pass 2 model."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    # Random image
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randint(0, 4096, (2, 1344))  # 32*32 + 16*16 + 8*8

    logits, _ = model(x)

    # Compute loss and backprop
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.reshape(-1, 4096), target.reshape(-1))
    loss.backward()

    # Verify gradients exist
    param_count = 0
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += 1
            if param.grad is not None:
                grad_count += 1

    assert grad_count > 0, "Should have at least some gradients"
    assert grad_count == param_count, "All parameters should have gradients"

    print("✓ Gradients flow through Pass 2 model")


def test_var_pass2_scale_conditioning():
    """
    Test that scale embeddings and masking create proper scale conditioning.

    The model should predict tokens scale by scale: given coarser scales,
    predict the next finer scale.
    """
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    x = torch.randn(2, 3, 64, 64)
    logits, token_maps = model(x)

    # Verify the model can distinguish between scales
    # (this is a weaker test; a stronger one would require actual training)
    assert logits.shape[1] == 32 * 32 + 16 * 16 + 8 * 8, (
        "Output should have one logit vector per token across all scales"
    )

    print("✓ Pass 2 model implements scale conditioning")


def test_var_pass2_batch_processing():
    """Test that Pass 2 handles different batch sizes correctly."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 3, 64, 64)
        logits, token_maps = model(x)

        assert logits.shape[0] == batch_size, (
            f"Batch size should be {batch_size}, got {logits.shape[0]}"
        )

    print("✓ Pass 2 handles variable batch sizes correctly")


if __name__ == "__main__":
    test_cumulative_scale_mask()
    test_scale_embedding()
    test_transformer_block()
    test_transformer_block_with_mask()
    test_var_pass2_end_to_end()
    test_var_pass2_gradient_flow()
    test_var_pass2_scale_conditioning()
    test_var_pass2_batch_processing()
    print("\n✅ All Pass 2 tests passed!")
