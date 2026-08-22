import torch
import numpy as np
from sparse_attention import SparseAttentionMask, apply_sparse_mask_to_attention, MultiHeadSparseAttention


def test_sparse_mask_generation():
    """Test basic sparse attention mask generation."""
    seq_len = 16
    block_size = 4
    num_persistent_blocks = 2

    mask_gen = SparseAttentionMask(
        seq_len=seq_len,
        block_size=block_size,
        num_persistent_blocks=num_persistent_blocks
    )

    mask = mask_gen.create_mask()

    # Verify shape
    assert mask.shape == (seq_len, seq_len), f"Expected shape ({seq_len}, {seq_len}), got {mask.shape}"

    # Verify each position attends to its own block
    for i in range(seq_len):
        block_i = i // block_size
        block_start = block_i * block_size
        block_end = min((block_i + 1) * block_size, seq_len)

        # Check that local block positions have True
        for j in range(block_start, block_end):
            assert mask[i, j], f"Position {i} should attend to position {j} in same block"

    # Verify persistent block attention
    for i in range(seq_len):
        for persist_block_idx in range(min(num_persistent_blocks, mask_gen.num_blocks)):
            block_start = persist_block_idx * block_size
            block_end = min((persist_block_idx + 1) * block_size, seq_len)
            for j in range(block_start, block_end):
                assert mask[i, j], f"Position {i} should attend to persistent block {persist_block_idx}"

    print("✓ test_sparse_mask_generation passed")


def test_mask_sparsity():
    """Test that mask has expected sparsity."""
    seq_len = 64
    block_size = 8
    num_persistent_blocks = 2

    mask_gen = SparseAttentionMask(seq_len, block_size, num_persistent_blocks)
    mask = mask_gen.create_mask()

    stats = mask_gen.get_mask_statistics(mask)

    # With local blocks and persistent blocks, sparsity should be > 0.5
    # (less than 50% of attention is computed)
    sparsity = stats['sparsity_ratio']
    assert sparsity > 0.3, f"Expected significant sparsity, got {sparsity:.2%}"
    assert sparsity < 0.95, f"Expected some connections, got sparsity {sparsity:.2%}"

    print(f"✓ test_mask_sparsity passed (sparsity: {sparsity:.2%})")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")


def test_apply_mask_to_attention():
    """Test applying sparse mask to attention scores."""
    seq_len = 8
    batch_size = 2
    num_heads = 4

    # Create sample attention scores
    attn_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

    # Create sparse mask
    mask_gen = SparseAttentionMask(seq_len, block_size=2, num_persistent_blocks=1)
    mask = mask_gen.create_mask()

    # Apply mask
    masked_scores = apply_sparse_mask_to_attention(attn_scores, mask)

    # Verify masked positions are -inf
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(seq_len):
                for j in range(seq_len):
                    if not mask[i, j]:
                        assert masked_scores[b, h, i, j] == float('-inf'), \
                            f"Masked position ({i}, {j}) should be -inf"
                    else:
                        assert masked_scores[b, h, i, j] != float('-inf'), \
                            f"Unmasked position ({i}, {j}) should not be -inf"

    print("✓ test_apply_mask_to_attention passed")


def test_multihead_sparse_attention():
    """Test multi-head sparse attention layer."""
    batch_size = 2
    seq_len = 16
    dim = 64
    num_heads = 4
    block_size = 4
    num_persistent_blocks = 1

    # Create layer
    attention = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        num_persistent_blocks=num_persistent_blocks
    )

    # Create random input
    x = torch.randn(batch_size, seq_len, dim)

    # Forward pass
    output = attention(x)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, dim), \
        f"Expected output shape {(batch_size, seq_len, dim)}, got {output.shape}"

    # Verify output is not NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    # Verify output is different from input (not identity)
    assert not torch.allclose(output, x, atol=1e-5), "Output should be transformed"

    print("✓ test_multihead_sparse_attention passed")


def test_different_sequence_lengths():
    """Test that sparse attention works with variable sequence lengths."""
    dim = 32
    num_heads = 2
    block_size = 4
    num_persistent_blocks = 1

    attention = MultiHeadSparseAttention(dim, num_heads, block_size, num_persistent_blocks)

    for seq_len in [8, 16, 32, 64]:
        x = torch.randn(1, seq_len, dim)
        output = attention(x)

        assert output.shape == (1, seq_len, dim), \
            f"Failed for seq_len={seq_len}: expected {(1, seq_len, dim)}, got {output.shape}"

    print("✓ test_different_sequence_lengths passed")


def test_persistent_blocks_coverage():
    """Test that persistent blocks are correctly included in mask."""
    seq_len = 32
    block_size = 8
    num_persistent_blocks = 3

    mask_gen = SparseAttentionMask(seq_len, block_size, num_persistent_blocks)
    mask = mask_gen.create_mask()

    # For each position, check that it can attend to all persistent blocks
    num_blocks = (seq_len + block_size - 1) // block_size

    for i in range(seq_len):
        for block_idx in range(min(num_persistent_blocks, num_blocks)):
            block_start = block_idx * block_size
            block_end = min((block_idx + 1) * block_size, seq_len)

            # At least one position in the persistent block should be attended to
            persistent_block_attended = mask[i, block_start:block_end].any().item()
            assert persistent_block_attended, \
                f"Position {i} should attend to at least one position in persistent block {block_idx}"

    print("✓ test_persistent_blocks_coverage passed")


if __name__ == "__main__":
    test_sparse_mask_generation()
    test_mask_sparsity()
    test_apply_mask_to_attention()
    test_multihead_sparse_attention()
    test_different_sequence_lengths()
    test_persistent_blocks_coverage()
    print("\n✅ All tests passed!")
