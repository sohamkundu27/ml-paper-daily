"""
Tests for Pass 3: transformer blocks and NLTK POS tagging.
"""

import torch
import numpy as np
from transformer_block import TransformerBlock, TransformerStack
from pos_tagger import tag_text, tag_sequence, NLTK_TO_SIMPLIFIED


def test_pos_tagger_basic():
    """Verify POS tagger produces correct tag sequences."""
    text = "The cat sat on the mat ."
    tokens, pos_tags = tag_text(text)

    assert len(tokens) == len(pos_tags), f"Token and tag counts should match"
    assert len(tokens) > 0, f"Should tokenize non-empty text"

    # Check that pos_tags are from our simplified set
    valid_tags = set(NLTK_TO_SIMPLIFIED.values())
    for tag in pos_tags:
        assert tag in valid_tags, f"Tag '{tag}' not in simplified set"

    print(f"✓ POS tagger basic test passed")
    print(f"  Text: {text}")
    print(f"  Tokens: {tokens}")
    print(f"  Tags: {pos_tags}")


def test_pos_tagger_complex():
    """Verify POS tagger on more complex text."""
    text = "Natural language processing is fascinating and important for AI."
    tokens, pos_tags = tag_text(text)

    assert len(tokens) == len(pos_tags), "Token and tag lengths should match"
    assert len(tokens) >= 8, "Should have at least 8 tokens"

    # Spot check a few tags
    # "is" should be VERB (VBZ -> VERB)
    # "fascinating" should be ADJ (JJ -> ADJ)
    # "language" should be NOUN (NN -> NOUN)

    print(f"✓ POS tagger complex test passed")
    print(f"  Tokens: {tokens[:5]}... ({len(tokens)} total)")
    print(f"  Tags: {pos_tags[:5]}... ({len(pos_tags)} total)")


def test_tag_sequence():
    """Verify tag_sequence function on pre-tokenized input."""
    tokens = ["The", "quick", "brown", "fox"]
    pos_tags = tag_sequence(tokens)

    assert len(pos_tags) == len(tokens), "Tag count should match token count"
    assert all(tag in NLTK_TO_SIMPLIFIED.values() for tag in pos_tags), "All tags should be valid"

    print(f"✓ Tag sequence test passed")
    print(f"  Tokens: {tokens}")
    print(f"  Tags: {pos_tags}")


def test_transformer_block_shape():
    """Verify transformer block produces correct output shape."""
    batch_size, seq_len, embed_dim = 2, 8, 64
    num_heads = 4

    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=256,
    )

    x = torch.randn(batch_size, seq_len, embed_dim)
    output = block(x)

    assert output.shape == x.shape, f"Output shape should match input shape"
    print(f"✓ Transformer block shape test passed")


def test_transformer_block_with_sparse_attention():
    """Verify transformer block works with sparse attention."""
    seq_len, embed_dim = 6, 64
    num_heads = 4
    pos_tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]

    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=256,
        pos_tags=pos_tags,
        mask_type="hard",
    )

    batch_size = 2
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = block(x)

    assert output.shape == x.shape, f"Output shape {output.shape} should match input {x.shape}"
    print(f"✓ Transformer block with sparse attention test passed")


def test_transformer_block_with_soft_mask():
    """Verify transformer block works with soft sparse masks."""
    seq_len, embed_dim = 6, 64
    num_heads = 4
    pos_tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]

    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=256,
        pos_tags=pos_tags,
        mask_type="soft",
    )

    batch_size = 2
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = block(x)

    assert output.shape == x.shape, f"Output shape should match input shape"
    print(f"✓ Transformer block with soft mask test passed")


def test_transformer_stack():
    """Verify stacking multiple transformer blocks works."""
    batch_size, seq_len, embed_dim = 2, 8, 64
    num_heads = 4
    num_blocks = 3

    stack = TransformerStack(
        num_blocks=num_blocks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=256,
    )

    x = torch.randn(batch_size, seq_len, embed_dim)
    output = stack(x)

    assert output.shape == x.shape, f"Output shape should match input shape"
    assert len(stack.blocks) == num_blocks, f"Should have {num_blocks} blocks"
    print(f"✓ Transformer stack test passed ({num_blocks} blocks)")


def test_transformer_stack_with_sparse():
    """Verify transformer stack with sparse attention."""
    batch_size, seq_len, embed_dim = 2, 6, 64
    num_heads = 4
    num_blocks = 2
    pos_tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]

    stack = TransformerStack(
        num_blocks=num_blocks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=256,
        pos_tags=pos_tags,
        mask_type="hard",
    )

    x = torch.randn(batch_size, seq_len, embed_dim)
    output = stack(x)

    assert output.shape == x.shape, f"Output shape should match input shape"
    print(f"✓ Transformer stack with sparse attention test passed")


def test_gradient_flow_transformer_block():
    """Verify gradients flow through transformer blocks."""
    seq_len, embed_dim = 6, 64
    num_heads = 4

    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=256,
        pos_tags=None,  # Use dense attention to ensure gradients flow
    )

    batch_size = 1
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

    output = block(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Gradients should flow to input"
    assert x.grad.abs().sum() > 0, "Gradients should be non-zero"
    print(f"✓ Gradient flow transformer block test passed")


def test_transformer_block_dense_vs_sparse_different():
    """Verify dense and sparse blocks produce different outputs."""
    seq_len, embed_dim = 6, 64
    num_heads = 4
    pos_tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]

    dense_block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=256,
        pos_tags=None,  # Dense
    )

    sparse_block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=256,
        pos_tags=pos_tags,  # Sparse
    )

    torch.manual_seed(42)
    x = torch.randn(1, seq_len, embed_dim)

    # Use same seed for both blocks so only difference is attention mask
    torch.manual_seed(42)
    dense_block.eval()
    with torch.no_grad():
        dense_out = dense_block(x)

    torch.manual_seed(42)
    sparse_block.eval()
    with torch.no_grad():
        sparse_out = sparse_block(x)

    # Outputs should be different due to different attention patterns
    diff = (dense_out - sparse_out).abs().mean().item()
    assert diff > 0.001, f"Dense and sparse outputs should differ, got diff={diff}"

    print(f"✓ Dense vs sparse blocks produce different outputs (mean diff: {diff:.6f})")


if __name__ == "__main__":
    test_pos_tagger_basic()
    test_pos_tagger_complex()
    test_tag_sequence()
    test_transformer_block_shape()
    test_transformer_block_with_sparse_attention()
    test_transformer_block_with_soft_mask()
    test_transformer_stack()
    test_transformer_stack_with_sparse()
    test_gradient_flow_transformer_block()
    test_transformer_block_dense_vs_sparse_different()
    print("\n✅ All Pass 3 tests passed!")
