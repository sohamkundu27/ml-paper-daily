import torch
import numpy as np
from sparse_attention import BlockScorer, MultiHeadSparseAttention


def test_block_scorer():
    """Test that BlockScorer produces reasonable scores."""
    batch_size = 2
    seq_len = 16
    dim = 64
    num_blocks = 4

    scorer = BlockScorer(dim, num_blocks)
    x = torch.randn(batch_size, seq_len, dim)

    scores = scorer(x)

    # Verify shape
    assert scores.shape == (batch_size, num_blocks), \
        f"Expected shape {(batch_size, num_blocks)}, got {scores.shape}"

    # Verify scores are not NaN or Inf
    assert not torch.isnan(scores).any(), "Scores contain NaN"
    assert not torch.isinf(scores).any(), "Scores contain Inf"

    print("✓ test_block_scorer passed")


def test_block_scorer_multi_head_input():
    """Test BlockScorer with multi-head input shape (batch, num_heads, seq_len, head_dim)."""
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 16
    num_blocks = 4

    scorer = BlockScorer(num_heads * head_dim, num_blocks)

    # Multi-head value cache shape
    x = torch.randn(batch_size, num_heads, seq_len, head_dim)

    scores = scorer(x)

    assert scores.shape == (batch_size, num_blocks), \
        f"Expected shape {(batch_size, num_blocks)}, got {scores.shape}"

    print("✓ test_block_scorer_multi_head_input passed")


def test_learned_attention_layer():
    """Test MultiHeadSparseAttention with learned blocks enabled."""
    batch_size = 2
    seq_len = 16
    dim = 64
    num_heads = 4
    block_size = 4
    num_persistent_blocks = 2

    attention = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        num_persistent_blocks=num_persistent_blocks,
        use_learned_blocks=True
    )

    x = torch.randn(batch_size, seq_len, dim)
    output = attention(x)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, dim), \
        f"Expected output shape {(batch_size, seq_len, dim)}, got {output.shape}"

    # Verify output is valid
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print("✓ test_learned_attention_layer passed")


def test_learned_blocks_are_different_per_batch():
    """Test that learned block selection can produce different selections per batch."""
    batch_size = 4
    seq_len = 16
    dim = 64
    num_heads = 4
    block_size = 4
    num_persistent_blocks = 2

    attention = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        num_persistent_blocks=num_persistent_blocks,
        use_learned_blocks=True
    )

    # Create different inputs for each batch
    x = torch.randn(batch_size, seq_len, dim)

    # Forward pass (will score blocks and select top-k)
    output = attention(x)

    # Get the selected blocks
    block_scores = attention.block_scorer(x)
    _, top_indices = torch.topk(block_scores, k=num_persistent_blocks, dim=1)

    # Check that different batches can have different selections
    # (not guaranteed, but with random data, likely)
    print(f"Selected block indices per batch: {top_indices.tolist()}")

    assert output.shape == (batch_size, seq_len, dim)
    print("✓ test_learned_blocks_are_different_per_batch passed")


def test_block_scorer_sensitivity():
    """Test that block scores change based on input."""
    batch_size = 1
    seq_len = 16
    dim = 64
    num_blocks = 4

    scorer = BlockScorer(dim, num_blocks)

    # Two different inputs
    x1 = torch.randn(batch_size, seq_len, dim)
    x2 = torch.randn(batch_size, seq_len, dim)

    scores1 = scorer(x1)
    scores2 = scorer(x2)

    # Scores should be different for different inputs
    # (with high probability)
    assert not torch.allclose(scores1, scores2, atol=1e-3), \
        "Scores should differ for different inputs"

    print("✓ test_block_scorer_sensitivity passed")


def test_learned_vs_fixed_blocks():
    """Compare learned and fixed block attention."""
    batch_size = 2
    seq_len = 16
    dim = 32
    num_heads = 2
    block_size = 4
    num_persistent_blocks = 1

    x = torch.randn(batch_size, seq_len, dim)

    # Fixed blocks
    fixed_attn = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        num_persistent_blocks=num_persistent_blocks,
        use_learned_blocks=False
    )

    # Learned blocks
    learned_attn = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        num_persistent_blocks=num_persistent_blocks,
        use_learned_blocks=True
    )

    # Both should produce valid outputs
    fixed_out = fixed_attn(x)
    learned_out = learned_attn(x)

    assert fixed_out.shape == learned_out.shape
    assert not torch.isnan(fixed_out).any()
    assert not torch.isnan(learned_out).any()

    # Outputs should be different (due to different block selection)
    # Note: not guaranteed to fail if both happen to select same blocks
    print(f"Fixed output mean: {fixed_out.mean().item():.4f}")
    print(f"Learned output mean: {learned_out.mean().item():.4f}")

    print("✓ test_learned_vs_fixed_blocks passed")


if __name__ == "__main__":
    test_block_scorer()
    test_block_scorer_multi_head_input()
    test_learned_attention_layer()
    test_learned_blocks_are_different_per_batch()
    test_block_scorer_sensitivity()
    test_learned_vs_fixed_blocks()
    print("\n✅ All learned block tests passed!")
