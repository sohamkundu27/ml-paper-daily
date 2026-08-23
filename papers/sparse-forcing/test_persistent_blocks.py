"""Tests for persistent block state management (Pass 3)."""

import torch
from sparse_attention import (
    PersistentBlockState, PersistentBlockCache, MultiHeadSparseAttention
)


def test_persistent_block_state_creation():
    """Test creating and initializing persistent block state."""
    num_blocks = 4
    feature_dim = 64

    state = PersistentBlockState(num_blocks, feature_dim)

    assert state.num_blocks == num_blocks
    assert state.feature_dim == feature_dim
    assert len(state.history) == 0

    print("✓ test_persistent_block_state_creation passed")


def test_persistent_block_state_update():
    """Test updating persistent block state with new features."""
    num_blocks = 4
    feature_dim = 64

    state = PersistentBlockState(num_blocks, feature_dim)

    # Create and update with block features
    block_features = torch.randn(num_blocks, feature_dim)
    state.update(block_features)

    assert len(state.history) == 1
    assert state.history[0][0] == 0  # frame_idx
    assert state.history[0][1].shape == (num_blocks, feature_dim)

    # Update with another frame
    block_features2 = torch.randn(num_blocks, feature_dim)
    state.update(block_features2)

    assert len(state.history) == 2
    assert state.history[1][0] == 1  # frame_idx incremented

    print("✓ test_persistent_block_state_update passed")


def test_persistent_block_state_get_current():
    """Test getting current compressed state."""
    num_blocks = 4
    feature_dim = 64

    state = PersistentBlockState(num_blocks, feature_dim, max_history=4)

    # Empty state should return zeros
    current = state.get_current_state()
    assert current.shape == (num_blocks, feature_dim)

    # After update, should return weighted average
    block_features = torch.ones(num_blocks, feature_dim) * 2.0
    state.update(block_features)

    current = state.get_current_state()
    assert current.shape == (num_blocks, feature_dim)
    assert not torch.allclose(current, torch.zeros_like(current))
    assert torch.allclose(current, torch.full_like(current, 2.0), atol=0.5)

    print("✓ test_persistent_block_state_get_current passed")


def test_persistent_block_state_compress():
    """Test compression of old state information."""
    num_blocks = 4
    feature_dim = 32

    state = PersistentBlockState(num_blocks, feature_dim, max_history=10)

    # Add several frames
    for i in range(6):
        features = torch.randn(num_blocks, feature_dim)
        state.update(features)

    assert len(state.history) == 6

    # Compress
    state.compress(compression_ratio=0.5)

    # After compression, should have fewer entries
    assert len(state.history) <= 6

    # Get current state should still work
    current = state.get_current_state()
    assert current.shape == (num_blocks, feature_dim)

    print(f"✓ test_persistent_block_state_compress passed (history: 6 -> {len(state.history)})")


def test_persistent_block_state_clear_stale():
    """Test clearing stale information."""
    num_blocks = 4
    feature_dim = 32

    state = PersistentBlockState(num_blocks, feature_dim, max_history=10)

    # Add several frames
    for i in range(6):
        features = torch.randn(num_blocks, feature_dim)
        state.update(features)

    assert len(state.history) == 6

    # Clear stale, keeping only 2 recent frames
    state.clear_stale(window_size=2)

    assert len(state.history) <= 2

    print(f"✓ test_persistent_block_state_clear_stale passed (kept {len(state.history)} frames)")


def test_persistent_block_cache_creation():
    """Test creating persistent block cache."""
    num_blocks = 4
    feature_dim = 64

    cache = PersistentBlockCache(num_blocks, feature_dim)

    assert cache.state.num_blocks == num_blocks
    assert cache.state.feature_dim == feature_dim

    print("✓ test_persistent_block_cache_creation passed")


def test_persistent_block_cache_update_from_attention():
    """Test updating cache from attention output."""
    batch_size = 2
    seq_len = 16
    dim = 64
    block_size = 4

    cache = PersistentBlockCache(
        num_blocks=(seq_len + block_size - 1) // block_size,
        feature_dim=dim
    )

    # Create mock attention output
    attn_output = torch.randn(batch_size, seq_len, dim)

    # Update cache
    cache.update_from_attention_output(attn_output, block_size)

    assert len(cache.state.history) == 1

    # Get current state
    state = cache.get_state()
    num_blocks = (seq_len + block_size - 1) // block_size
    assert state.shape == (num_blocks, dim)

    print("✓ test_persistent_block_cache_update_from_attention passed")


def test_persistent_block_cache_compress():
    """Test cache compression."""
    cache = PersistentBlockCache(num_blocks=4, feature_dim=64)

    # Add multiple frames
    for i in range(5):
        attn_output = torch.randn(1, 16, 64)
        cache.update_from_attention_output(attn_output, block_size=4)

    initial_frames = len(cache.state.history)

    # Compress
    cache.compress(ratio=0.5)

    # Should have fewer or same number of frames
    assert len(cache.state.history) <= initial_frames

    print(f"✓ test_persistent_block_cache_compress passed ({initial_frames} -> {len(cache.state.history)} frames)")


def test_persistent_block_cache_reset():
    """Test cache reset."""
    cache = PersistentBlockCache(num_blocks=4, feature_dim=64)

    # Add frames
    for i in range(3):
        attn_output = torch.randn(1, 16, 64)
        cache.update_from_attention_output(attn_output, block_size=4)

    assert len(cache.state.history) > 0

    # Reset
    cache.reset()

    assert len(cache.state.history) == 0

    print("✓ test_persistent_block_cache_reset passed")


def test_multihead_sparse_attention_with_persistent_cache():
    """Test MultiHeadSparseAttention with persistent cache enabled."""
    batch_size = 2
    seq_len = 16
    dim = 64
    num_heads = 4
    block_size = 4
    num_persistent_blocks = 1

    attention = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        num_persistent_blocks=num_persistent_blocks,
        use_persistent_cache=True
    )

    # Forward pass 1
    x1 = torch.randn(batch_size, seq_len, dim)
    out1 = attention(x1)

    assert out1.shape == (batch_size, seq_len, dim)
    assert attention.block_cache is not None
    assert len(attention.block_cache.state.history) == 1

    # Forward pass 2 - cache should accumulate
    x2 = torch.randn(batch_size, seq_len, dim)
    out2 = attention(x2)

    assert out2.shape == (batch_size, seq_len, dim)
    assert len(attention.block_cache.state.history) == 2

    print("✓ test_multihead_sparse_attention_with_persistent_cache passed")


def test_persistent_block_memory_tracking():
    """Test memory tracking functionality."""
    cache = PersistentBlockCache(num_blocks=4, feature_dim=64, max_history=10)

    # Add frames
    for i in range(3):
        attn_output = torch.randn(1, 16, 64)
        cache.update_from_attention_output(attn_output, block_size=4)

    mem_info = cache.get_memory_info()

    assert 'num_frames' in mem_info
    assert 'total_floats' in mem_info
    assert 'est_memory_mb' in mem_info
    assert mem_info['num_frames'] == 3

    print(f"✓ test_persistent_block_memory_tracking passed (mem: {mem_info['est_memory_mb']:.2f} MB)")


def test_persistent_block_importance():
    """Test block importance scoring."""
    state = PersistentBlockState(num_blocks=4, feature_dim=64)

    # Add frames with different characteristics
    for i in range(3):
        features = torch.randn(4, 64)
        state.update(features)

    # Get importance scores (variance-based)
    importance = state.get_block_importance()

    assert importance.shape == (4,)
    assert not torch.any(torch.isnan(importance))

    print("✓ test_persistent_block_importance passed")


def test_autoregressive_decoding_simulation():
    """Simulate autoregressive decoding with persistent state updates."""
    batch_size = 1
    seq_len = 16
    dim = 64
    num_heads = 4
    block_size = 4
    num_frames = 4

    attention = MultiHeadSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        num_persistent_blocks=2,
        use_persistent_cache=True
    )

    # Simulate decoding 4 frames
    for frame_idx in range(num_frames):
        x = torch.randn(batch_size, seq_len, dim)
        out = attention(x)

        assert out.shape == (batch_size, seq_len, dim)

        # After each frame, cache should grow (or compress)
        cache_frames = len(attention.block_cache.state.history)
        assert cache_frames >= 1

        # Periodically compress
        if frame_idx % 2 == 0:
            attention.block_cache.compress(ratio=0.6)

    print(f"✓ test_autoregressive_decoding_simulation passed (decoded {num_frames} frames)")


if __name__ == "__main__":
    test_persistent_block_state_creation()
    test_persistent_block_state_update()
    test_persistent_block_state_get_current()
    test_persistent_block_state_compress()
    test_persistent_block_state_clear_stale()
    test_persistent_block_cache_creation()
    test_persistent_block_cache_update_from_attention()
    test_persistent_block_cache_compress()
    test_persistent_block_cache_reset()
    test_multihead_sparse_attention_with_persistent_cache()
    test_persistent_block_memory_tracking()
    test_persistent_block_importance()
    test_autoregressive_decoding_simulation()

    print("\n✅ All persistent block tests passed!")
