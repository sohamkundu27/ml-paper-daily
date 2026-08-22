"""
Demo of learnable block selection for sparse attention.
Compares fixed vs learned persistent block selection on synthetic video data.
"""

import torch
import torch.nn.functional as F
from sparse_attention import MultiHeadSparseAttention, BlockScorer


def demo_block_scorer():
    """Demo 1: Show how block scorer works."""
    print("=" * 70)
    print("DEMO 1: Block Scorer - Learning which blocks are important")
    print("=" * 70)

    batch_size = 1
    seq_len = 32
    dim = 64
    num_blocks = 4

    scorer = BlockScorer(dim, num_blocks, hidden_dim=64)

    # Create synthetic "frames" - each block represents one frame region
    x = torch.randn(batch_size, seq_len, dim)

    # Score blocks
    scores = scorer(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Number of blocks: {num_blocks}")
    print(f"Block size: {seq_len // num_blocks} tokens per block")

    print(f"\nBlock importance scores:")
    for block_idx in range(num_blocks):
        score = scores[0, block_idx].item()
        print(f"  Block {block_idx}: {score:8.4f} {'█' * int(max(0, score + 5) * 2)}")

    # Top-k selection
    top_scores, top_indices = torch.topk(scores, k=2, dim=1)
    print(f"\nTop 2 blocks selected: {top_indices[0].tolist()}")


def demo_learned_block_selection():
    """Demo 2: Show learned block selection in a sparse attention layer."""
    print("\n" + "=" * 70)
    print("DEMO 2: Learned Block Selection in Attention Layer")
    print("=" * 70)

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

    # Forward pass
    output = attention(x)

    # Get block scores
    block_scores = attention.block_scorer(x)
    _, selected_blocks = torch.topk(block_scores, k=num_persistent_blocks, dim=1)

    print(f"\nInput: {x.shape}")
    print(f"Block size: {block_size} tokens")
    print(f"Total blocks: {(seq_len + block_size - 1) // block_size}")

    for batch_idx in range(batch_size):
        scores = block_scores[batch_idx].tolist()
        selected = selected_blocks[batch_idx].tolist()
        print(f"\nBatch {batch_idx}:")
        print(f"  Block scores: {[f'{s:.3f}' for s in scores]}")
        print(f"  Selected persistent blocks: {selected}")

    print(f"\nOutput shape: {output.shape}")
    print(f"Output is valid (no NaN/Inf): {not (torch.isnan(output).any() or torch.isinf(output).any())}")


def demo_synthetic_video_comparison():
    """Demo 3: Compare learned vs fixed blocks on synthetic "video" data."""
    print("\n" + "=" * 70)
    print("DEMO 3: Learned vs Fixed Blocks on Synthetic Video")
    print("=" * 70)

    # Simulate a mini video: seq_len=64 as 4 frames * 16 spatial tokens
    batch_size = 4
    seq_len = 64
    dim = 128
    num_heads = 8
    block_size = 16
    num_persistent_blocks = 2

    print(f"\nSynthetic video setup:")
    print(f"  Sequence length: {seq_len} (4 frames × 16 tokens)")
    print(f"  Block size: {block_size} tokens per block")
    print(f"  Number of blocks: {(seq_len + block_size - 1) // block_size}")
    print(f"  Persistent blocks: {num_persistent_blocks}")

    # Create fixed and learned attention layers
    fixed_attn = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        num_persistent_blocks=num_persistent_blocks,
        use_learned_blocks=False
    )

    learned_attn = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        num_persistent_blocks=num_persistent_blocks,
        use_learned_blocks=True
    )

    # Create synthetic video frames
    x = torch.randn(batch_size, seq_len, dim)

    # Forward pass with both
    fixed_out = fixed_attn(x)
    learned_out = learned_attn(x)

    print(f"\nFixed block selection:")
    print(f"  Always uses blocks: {fixed_attn.sparse_mask_gen.persistent_block_indices}")

    print(f"\nLearned block selection per batch:")
    block_scores = learned_attn.block_scorer(x)
    _, selected_blocks = torch.topk(block_scores, k=num_persistent_blocks, dim=1)
    for batch_idx in range(batch_size):
        selected = selected_blocks[batch_idx].tolist()
        print(f"  Batch {batch_idx}: blocks {selected}")

    # Compute attention patterns
    fixed_mask = fixed_attn.sparse_mask_gen.create_mask()
    learned_mask = learned_attn.sparse_mask_gen.create_mask()

    fixed_sparsity = 1.0 - (fixed_mask.sum().float() / fixed_mask.numel())
    learned_sparsity = 1.0 - (learned_mask.sum().float() / learned_mask.numel())

    print(f"\nMask statistics:")
    print(f"  Fixed mask sparsity: {fixed_sparsity:.1%}")
    print(f"  Learned mask sparsity: {learned_sparsity:.1%}")
    print(f"  Compression (fixed): {fixed_mask.numel() / fixed_mask.sum().float():.2f}x")
    print(f"  Compression (learned): {learned_mask.numel() / learned_mask.sum().float():.2f}x")

    print(f"\nOutput statistics:")
    print(f"  Fixed output mean: {fixed_out.mean():.4f}, std: {fixed_out.std():.4f}")
    print(f"  Learned output mean: {learned_out.mean():.4f}, std: {learned_out.std():.4f}")


def demo_block_scores_over_iterations():
    """Demo 4: Show that block scores change as model trains."""
    print("\n" + "=" * 70)
    print("DEMO 4: Block Scores Change During Training")
    print("=" * 70)

    batch_size = 1
    seq_len = 16
    dim = 64
    num_blocks = 4

    scorer = BlockScorer(dim, num_blocks, hidden_dim=32)

    # Optimize scorer to maximize first block's score (toy learning task)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=0.1)

    x = torch.randn(batch_size, seq_len, dim)

    print(f"\nTraining scorer to prefer block 0...")
    print(f"Iteration | Block Scores")

    for iteration in range(5):
        scores = scorer(x)

        # Loss: minimize score of block 0 (just for demo)
        loss = -scores[:, 0].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 1 == 0:
            with torch.no_grad():
                scores = scorer(x)
            score_str = " ".join([f"{s.item():7.3f}" for s in scores[0]])
            print(f"    {iteration:4d}   | {score_str}")

    print(f"\nBlock scores changed through optimization!")


if __name__ == "__main__":
    demo_block_scorer()
    demo_learned_block_selection()
    demo_synthetic_video_comparison()
    demo_block_scores_over_iterations()

    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)
    print("\nSummary:")
    print("- BlockScorer learns to score block importance from input")
    print("- MultiHeadSparseAttention can select blocks dynamically")
    print("- Learned selection adapts per batch, fixed selection is static")
    print("- Block scores can be optimized during training")
