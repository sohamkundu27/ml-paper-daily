"""Demo of persistent block state management across decoding steps (Pass 3)."""

import torch
from sparse_attention import PersistentBlockCache, PersistentBlockState, MultiHeadSparseAttention


def demo_block_state_across_frames():
    """Demo 1: Show block state accumulation across frames."""
    print("=" * 70)
    print("DEMO 1: Persistent Block State Accumulation Across Frames")
    print("=" * 70)

    num_blocks = 4
    feature_dim = 32

    state = PersistentBlockState(num_blocks, feature_dim, max_history=10)

    print(f"\nSimulating {5} autoregressive decoding steps...")
    print(f"Number of blocks: {num_blocks}, Feature dimension: {feature_dim}")
    print("\nFrame | History Size | Current State Mean | Memory (MB)")
    print("-" * 70)

    for frame_idx in range(5):
        # Create synthetic "decoded" block features for this frame
        block_features = torch.randn(num_blocks, feature_dim)
        state.update(block_features)

        # Get current compressed state
        current_state = state.get_current_state()
        mean_val = current_state.mean().item()

        # Get memory info
        mem_info = state.memory_info()
        memory_mb = mem_info['est_memory_mb']

        print(f"  {frame_idx} | {len(state.history):12d} | {mean_val:17.4f} | {memory_mb:9.3f}")

    print(f"\n✓ Frame history preserved efficiently across {frame_idx + 1} steps")


def demo_compression_reduces_memory():
    """Demo 2: Show that compression reduces memory."""
    print("\n" + "=" * 70)
    print("DEMO 2: State Compression Reduces Memory Usage")
    print("=" * 70)

    num_blocks = 8
    feature_dim = 128
    num_frames = 10

    state = PersistentBlockState(num_blocks, feature_dim, max_history=20)

    # Add many frames
    print(f"\nAdding {num_frames} frames...")
    for i in range(num_frames):
        features = torch.randn(num_blocks, feature_dim)
        state.update(features)

    before_mem = state.memory_info()
    before_frames = len(state.history)

    print(f"Before compression:")
    print(f"  Frames stored: {before_frames}")
    print(f"  Memory usage: {before_mem['est_memory_mb']:.3f} MB")

    # Compress
    state.compress(compression_ratio=0.5)

    after_mem = state.memory_info()
    after_frames = len(state.history)

    print(f"\nAfter compression (50%):")
    print(f"  Frames stored: {after_frames}")
    print(f"  Memory usage: {after_mem['est_memory_mb']:.3f} MB")
    print(f"  Reduction: {(1.0 - after_mem['est_memory_mb'] / before_mem['est_memory_mb']) * 100:.1f}%")

    # State still usable
    current = state.get_current_state()
    assert current.shape == (num_blocks, feature_dim)
    print(f"\n✓ Compressed state is still valid shape: {current.shape}")


def demo_stale_clearing():
    """Demo 3: Show clearing of stale information."""
    print("\n" + "=" * 70)
    print("DEMO 3: Clearing Stale Information (Sliding Window)")
    print("=" * 70)

    num_blocks = 4
    feature_dim = 32

    state = PersistentBlockState(num_blocks, feature_dim, max_history=20)

    # Add 10 frames
    print(f"\nAdding 10 frames to state...")
    for i in range(10):
        features = torch.randn(num_blocks, feature_dim)
        state.update(features)

    print(f"History size before clear: {len(state.history)} frames")

    # Keep only 3 recent frames
    state.clear_stale(window_size=3)

    print(f"History size after clear (window=3): {len(state.history)} frames")

    # Get importance of remaining blocks
    importance = state.get_block_importance()
    print(f"\nBlock importance (variance-based):")
    for block_idx in range(num_blocks):
        score = importance[block_idx].item()
        bar = "█" * int(max(0, (score + 1)) * 5)
        print(f"  Block {block_idx}: {score:7.4f} {bar}")

    print(f"\n✓ Stale clearing keeps only recent important information")


def demo_cache_through_decoding():
    """Demo 4: Simulate full autoregressive decoding with cache."""
    print("\n" + "=" * 70)
    print("DEMO 4: Persistent Cache Through Autoregressive Decoding")
    print("=" * 70)

    batch_size = 1
    seq_len = 16
    dim = 64
    num_heads = 4
    block_size = 4
    num_frames = 6

    cache = PersistentBlockCache(
        num_blocks=(seq_len + block_size - 1) // block_size,
        feature_dim=dim,
        max_history=10
    )

    print(f"\nAutoregressive decoding simulation:")
    print(f"  Frames to generate: {num_frames}")
    print(f"  Sequence length: {seq_len} tokens")
    print(f"  Block size: {block_size}")
    print(f"  Blocks: {(seq_len + block_size - 1) // block_size}")

    print(f"\nFrame | Cache Size | Memory (MB) | Current State Norm")
    print("-" * 70)

    for frame_idx in range(num_frames):
        # Simulate generating a new frame
        attn_output = torch.randn(batch_size, seq_len, dim)

        # Update cache with this frame's attention output
        cache.update_from_attention_output(attn_output, block_size)

        # Get memory info
        mem_info = cache.get_memory_info()
        current_state = cache.get_state()
        state_norm = current_state.norm().item()

        print(f"  {frame_idx} | {mem_info['num_frames']:10d} | {mem_info['est_memory_mb']:10.3f} | {state_norm:17.4f}")

        # Periodically compress to prevent unbounded growth
        if frame_idx % 2 == 1:
            cache.compress(ratio=0.6)
            print(f"    [compressed]")

    print(f"\n✓ Cache maintains state efficiently over {num_frames} decoding steps")


def demo_with_sparse_attention_integration():
    """Demo 5: Integration with MultiHeadSparseAttention."""
    print("\n" + "=" * 70)
    print("DEMO 5: Persistent Cache Integrated with Sparse Attention")
    print("=" * 70)

    batch_size = 2
    seq_len = 16
    dim = 64
    num_heads = 4
    block_size = 4
    num_persistent_blocks = 2
    num_frames = 4

    attention = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        num_persistent_blocks=num_persistent_blocks,
        use_persistent_cache=True
    )

    print(f"\nDecoding with sparse attention + persistent cache:")
    print(f"  Input dim: {dim}, Heads: {num_heads}, Block size: {block_size}")
    print(f"  Persistent blocks: {num_persistent_blocks}")

    print(f"\nFrame | Output Shape | Cache Frames | Output Norm")
    print("-" * 70)

    for frame_idx in range(num_frames):
        # Generate random input (simulating new tokens)
        x = torch.randn(batch_size, seq_len, dim)

        # Forward pass - this updates cache internally
        output = attention(x)

        # Check cache state
        cache_frames = len(attention.block_cache.state.history)
        output_norm = output.norm().item()

        print(f"  {frame_idx} | {str(output.shape):12s} | {cache_frames:12d} | {output_norm:11.4f}")

    print(f"\n✓ Sparse attention layer successfully maintains persistent block state")
    print(f"  Total cache entries: {len(attention.block_cache.state.history)}")


def demo_memory_efficiency_comparison():
    """Demo 6: Show memory efficiency of compression."""
    print("\n" + "=" * 70)
    print("DEMO 6: Memory Efficiency Comparison")
    print("=" * 70)

    num_blocks = 16
    feature_dim = 256
    num_frames_total = 20

    print(f"\nScenario: Generate {num_frames_total} frames")
    print(f"Blocks: {num_blocks}, Feature dim: {feature_dim}")

    # Without compression
    state_no_compress = PersistentBlockState(num_blocks, feature_dim, max_history=100)
    for i in range(num_frames_total):
        features = torch.randn(num_blocks, feature_dim)
        state_no_compress.update(features)

    mem_no_compress = state_no_compress.memory_info()

    # With periodic compression
    state_with_compress = PersistentBlockState(num_blocks, feature_dim, max_history=100)
    for i in range(num_frames_total):
        features = torch.randn(num_blocks, feature_dim)
        state_with_compress.update(features)

        if i % 4 == 0:
            state_with_compress.compress(compression_ratio=0.5)

    mem_with_compress = state_with_compress.memory_info()

    print(f"\nWithout compression:")
    print(f"  Frames: {mem_no_compress['num_frames']}")
    print(f"  Memory: {mem_no_compress['est_memory_mb']:.3f} MB")

    print(f"\nWith periodic compression (every 4 steps):")
    print(f"  Frames: {mem_with_compress['num_frames']}")
    print(f"  Memory: {mem_with_compress['est_memory_mb']:.3f} MB")

    reduction = (1.0 - mem_with_compress['est_memory_mb'] / mem_no_compress['est_memory_mb']) * 100
    print(f"\nMemory reduction: {reduction:.1f}%")
    print(f"✓ Compression allows long-horizon decoding without unbounded memory growth")


if __name__ == "__main__":
    demo_block_state_across_frames()
    demo_compression_reduces_memory()
    demo_stale_clearing()
    demo_cache_through_decoding()
    demo_with_sparse_attention_integration()
    demo_memory_efficiency_comparison()

    print("\n" + "=" * 70)
    print("✅ All demos complete!")
    print("=" * 70)
    print("\nSummary:")
    print("- PersistentBlockState tracks block features across frames")
    print("- Compression reduces memory while preserving important info")
    print("- Stale clearing maintains recent information only")
    print("- Cache integrates seamlessly with sparse attention")
    print("- Long-horizon decoding is now memory-efficient")
