"""
Tests for sparse multi-head attention layer.
"""

import torch
import numpy as np
from transformer_attention import SparseMultiHeadAttention, count_sparse_flops
from sparse_attention import compute_sparsity, create_hard_mask


def test_attention_output_shape():
    """Verify attention output has correct shape."""
    batch_size, seq_len, embed_dim = 2, 8, 64
    num_heads = 4

    attn = SparseMultiHeadAttention(embed_dim, num_heads)

    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)

    output, _ = attn(q, k, v)
    assert output.shape == (batch_size, seq_len, embed_dim), f"Expected {(batch_size, seq_len, embed_dim)}, got {output.shape}"
    print("✓ Attention output shape test passed")


def test_attention_with_sparse_hard_mask():
    """Verify attention works with hard sparse mask."""
    seq_len, embed_dim = 6, 64
    num_heads = 4
    pos_tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]

    attn = SparseMultiHeadAttention(
        embed_dim,
        num_heads,
        pos_tags=pos_tags,
        mask_type="hard"
    )

    batch_size = 2
    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)

    output, attn_weights = attn(q, k, v, need_weights=True)
    assert output.shape == (batch_size, seq_len, embed_dim), f"Unexpected output shape: {output.shape}"
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), f"Unexpected weight shape: {attn_weights.shape}"
    print("✓ Attention with hard mask test passed")


def test_attention_with_sparse_soft_mask():
    """Verify attention works with soft sparse mask."""
    seq_len, embed_dim = 6, 64
    num_heads = 4
    pos_tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]

    attn = SparseMultiHeadAttention(
        embed_dim,
        num_heads,
        pos_tags=pos_tags,
        mask_type="soft"
    )

    batch_size = 2
    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)

    output, _ = attn(q, k, v)
    assert output.shape == (batch_size, seq_len, embed_dim)
    print("✓ Attention with soft mask test passed")


def test_attention_without_sparse_mask():
    """Verify attention works without sparse mask (dense)."""
    batch_size, seq_len, embed_dim = 2, 8, 64
    num_heads = 4

    attn = SparseMultiHeadAttention(embed_dim, num_heads, pos_tags=None)

    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)

    output, _ = attn(q, k, v)
    assert output.shape == (batch_size, seq_len, embed_dim)
    print("✓ Attention without mask (dense) test passed")


def test_sparse_mask_is_applied():
    """Verify that sparse mask actually reduces attention to masked positions."""
    seq_len, embed_dim = 4, 64
    num_heads = 2
    # NOUN should attend to DET, but not to VERB (simplified rules)
    pos_tags = ["NOUN", "DET", "VERB", "NOUN"]

    attn = SparseMultiHeadAttention(
        embed_dim,
        num_heads,
        pos_tags=pos_tags,
        mask_type="hard"
    )

    # Use fixed input for reproducibility
    torch.manual_seed(42)
    batch_size = 1
    q = torch.ones(batch_size, seq_len, embed_dim)
    k = torch.ones(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)

    _, attn_weights = attn(q, k, v, need_weights=True)

    # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
    # Check that masked positions have very low attention (close to 0)
    # NOUN (0) should not attend to VERB (2), so attn_weights[0, :, 0, 2] should be ~0
    masked_attn = attn_weights[0, :, 0, 2].mean().item()
    assert masked_attn < 0.1, f"Masked attention weight should be ~0, got {masked_attn}"
    print("✓ Sparse mask is applied test passed")


def test_compute_sparse_flops():
    """Verify FLOP counting function."""
    seq_len, embed_dim, num_heads = 128, 512, 8
    sparsity = 0.5

    dense_flops, sparse_flops = count_sparse_flops(seq_len, embed_dim, num_heads, sparsity)

    assert sparse_flops < dense_flops, "Sparse FLOPs should be less than dense"
    # With 50% sparsity, sparse should be roughly half of dense
    assert sparse_flops == int(dense_flops * (1 - sparsity)), "FLOP ratio mismatch"
    print(f"✓ FLOP computation test passed (dense: {dense_flops}, sparse: {sparse_flops})")


def test_realistic_sequence_savings():
    """Test on realistic POS sequence and verify computational savings."""
    seq_len, embed_dim = 12, 256
    num_heads = 8
    pos_tags = ["DET", "NOUN", "VERB", "DET", "NOUN", "PUNCT",
                "NOUN", "ADP", "NOUN", "VERB", "ADV", "PUNCT"]

    # Create sparse attention
    attn_sparse = SparseMultiHeadAttention(
        embed_dim,
        num_heads,
        pos_tags=pos_tags,
        mask_type="hard"
    )

    # Get sparsity from the mask
    mask_np = create_hard_mask(pos_tags)
    sparsity = compute_sparsity(mask_np)

    # Compute FLOP savings
    dense_flops, sparse_flops = count_sparse_flops(seq_len, embed_dim, num_heads, sparsity)
    savings_percent = (1 - sparse_flops / dense_flops) * 100

    print(f"✓ Realistic sequence savings: {savings_percent:.1f}% reduction (sparsity: {sparsity:.1%})")
    assert savings_percent > 0, "Should have some FLOP savings"


def test_gradient_flow():
    """Verify that gradients flow through sparse attention."""
    seq_len, embed_dim = 6, 64
    num_heads = 4
    pos_tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]

    attn = SparseMultiHeadAttention(
        embed_dim,
        num_heads,
        pos_tags=pos_tags,
        mask_type="hard"
    )

    batch_size = 1
    q = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    k = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    v = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

    output, _ = attn(q, k, v)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist and are non-zero
    assert q.grad is not None, "Query gradients should exist"
    assert k.grad is not None, "Key gradients should exist"
    assert v.grad is not None, "Value gradients should exist"
    assert q.grad.abs().sum() > 0, "Query gradients should be non-zero"
    print("✓ Gradient flow test passed")


def test_hard_vs_soft_masks():
    """Verify hard and soft masks produce different attention distributions."""
    seq_len, embed_dim = 6, 64
    num_heads = 4
    pos_tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]

    attn_hard = SparseMultiHeadAttention(
        embed_dim,
        num_heads,
        pos_tags=pos_tags,
        mask_type="hard"
    )

    attn_soft = SparseMultiHeadAttention(
        embed_dim,
        num_heads,
        pos_tags=pos_tags,
        mask_type="soft"
    )

    torch.manual_seed(42)
    batch_size = 1
    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)

    _, weights_hard = attn_hard(q, k, v, need_weights=True)
    _, weights_soft = attn_soft(q, k, v, need_weights=True)

    # Hard and soft should produce different distributions
    diff = (weights_hard - weights_soft).abs().mean().item()
    assert diff > 0.01, "Hard and soft masks should produce different distributions"
    print(f"✓ Hard vs soft masks test passed (mean difference: {diff:.4f})")


if __name__ == "__main__":
    test_attention_output_shape()
    test_attention_with_sparse_hard_mask()
    test_attention_with_sparse_soft_mask()
    test_attention_without_sparse_mask()
    test_sparse_mask_is_applied()
    test_compute_sparse_flops()
    test_realistic_sequence_savings()
    test_gradient_flow()
    test_hard_vs_soft_masks()
    print("\n✅ All transformer attention tests passed!")
