"""
Demo of sparse attention mask patterns.
Visualizes how sparse forcing concentrates attention on blocks.
"""

import torch
from sparse_attention import SparseAttentionMask, MultiHeadSparseAttention


def visualize_mask(mask, title="Sparse Attention Mask"):
    """
    Print a text visualization of the attention mask.
    '█' = allowed attention, '·' = masked out
    """
    print(f"\n{title}")
    print(f"Sequence length: {mask.shape[0]}")

    # Show a subset for readability if too large
    display_size = min(32, mask.shape[0])
    display_mask = mask[:display_size, :display_size]

    print("  ", end="")
    for j in range(display_size):
        print(f"{j % 10}", end="")
    print()

    for i in range(display_size):
        print(f"{i:2d}", end="")
        for j in range(display_size):
            print("█" if display_mask[i, j] else "·", end="")
        print()


def demo_basic_sparse_mask():
    """Demo 1: Show basic sparse attention mask pattern."""
    print("=" * 60)
    print("DEMO 1: Basic Sparse Attention Mask")
    print("=" * 60)

    seq_len = 32
    block_size = 8
    num_persistent_blocks = 2

    mask_gen = SparseAttentionMask(seq_len, block_size, num_persistent_blocks)
    mask = mask_gen.create_mask()

    visualize_mask(mask, "Sparse Attention Pattern (block_size=8, persistent_blocks=2)")

    stats = mask_gen.get_mask_statistics(mask)
    print(f"\nStatistics:")
    print(f"  Total positions: {stats['total_positions']}")
    print(f"  Allowed positions: {stats['allowed_positions']}")
    print(f"  Sparsity: {stats['sparsity_ratio']:.1%}")
    print(f"  Compression: {stats['compression_ratio']:.2f}x")


def demo_effect_of_block_size():
    """Demo 2: Show effect of different block sizes."""
    print("\n" + "=" * 60)
    print("DEMO 2: Effect of Block Size on Sparsity")
    print("=" * 60)

    seq_len = 64
    num_persistent_blocks = 2

    for block_size in [4, 8, 16]:
        mask_gen = SparseAttentionMask(seq_len, block_size, num_persistent_blocks)
        mask = mask_gen.create_mask()
        stats = mask_gen.get_mask_statistics(mask)

        print(f"\nBlock size = {block_size}:")
        print(f"  Sparsity: {stats['sparsity_ratio']:.1%}")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")


def demo_effect_of_persistent_blocks():
    """Demo 3: Show effect of number of persistent blocks."""
    print("\n" + "=" * 60)
    print("DEMO 3: Effect of Persistent Blocks on Sparsity")
    print("=" * 60)

    seq_len = 64
    block_size = 8

    for num_persistent in [1, 2, 4, 8]:
        mask_gen = SparseAttentionMask(seq_len, block_size, num_persistent)
        mask = mask_gen.create_mask()
        stats = mask_gen.get_mask_statistics(mask)

        print(f"\nPersistent blocks = {num_persistent}:")
        print(f"  Sparsity: {stats['sparsity_ratio']:.1%}")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")


def demo_multihead_attention():
    """Demo 4: Show multi-head sparse attention layer."""
    print("\n" + "=" * 60)
    print("DEMO 4: Multi-Head Sparse Attention Layer")
    print("=" * 60)

    batch_size = 2
    seq_len = 16
    dim = 64
    num_heads = 4

    attention = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=4,
        num_persistent_blocks=1
    )

    # Create sample video frames (simulated as sequence)
    x = torch.randn(batch_size, seq_len, dim)

    print(f"\nInput shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Feature dimension: {dim}")

    # Forward pass
    output = attention(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")

    # Show sparse mask being used
    mask = attention.sparse_mask_gen.create_mask()
    visualize_mask(mask, "Sparse Attention Mask (seq_len=16, block_size=4)")


def demo_computation_savings():
    """Demo 5: Estimate computational savings from sparsity."""
    print("\n" + "=" * 60)
    print("DEMO 5: Computational Savings")
    print("=" * 60)

    seq_len = 256
    block_size = 16
    num_persistent_blocks = 4

    mask_gen = SparseAttentionMask(seq_len, block_size, num_persistent_blocks)
    mask = mask_gen.create_mask()
    stats = mask_gen.get_mask_statistics(mask)

    full_attention_ops = seq_len ** 2
    sparse_attention_ops = stats['allowed_positions']
    savings = full_attention_ops - sparse_attention_ops

    print(f"\nFor sequence length {seq_len}:")
    print(f"  Full attention operations: {full_attention_ops:,}")
    print(f"  Sparse attention operations: {sparse_attention_ops:,}")
    print(f"  Operations saved: {savings:,}")
    print(f"  Speedup factor: {full_attention_ops / sparse_attention_ops:.2f}x")
    print(f"  Memory savings: {stats['sparsity_ratio']:.1%}")


if __name__ == "__main__":
    demo_basic_sparse_mask()
    demo_effect_of_block_size()
    demo_effect_of_persistent_blocks()
    demo_multihead_attention()
    demo_computation_savings()
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
