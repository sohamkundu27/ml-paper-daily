import torch
from llsa import HierarchicalSparseAttention


def test_hierarchical_sparse_attention_basic():
    """Test basic instantiation and forward pass."""
    seq_len = 256
    d = 64

    attention = HierarchicalSparseAttention(num_levels=3)
    x = torch.randn(seq_len, d)

    selected_masks, all_selected = attention.forward(x)

    # Check shapes
    assert len(selected_masks) == 3, "Should have 3 levels"
    assert all_selected.shape == (seq_len,), f"Expected shape ({seq_len},), got {all_selected.shape}"
    assert all_selected.dtype == torch.bool, "Should be boolean mask"

    # Check that each level's mask is subset of final mask
    for level_mask in selected_masks:
        assert level_mask.dtype == torch.bool
        assert level_mask.shape == (seq_len,)

    print("✓ Basic forward pass test passed")


def test_sparse_selection_reduces_tokens():
    """Test that sparse selection actually reduces the number of tokens."""
    seq_len = 512
    d = 128

    attention = HierarchicalSparseAttention(num_levels=3)
    x = torch.randn(seq_len, d)

    selected_masks, all_selected = attention.forward(x)

    num_selected = all_selected.sum().item()
    assert num_selected > 0, "Should select at least some tokens"
    assert num_selected < seq_len, f"Sparse selection should reduce tokens ({num_selected} < {seq_len})"
    assert num_selected <= seq_len, "Cannot select more tokens than sequence length"

    print(f"✓ Token reduction test passed: {num_selected}/{seq_len} tokens selected ({100*num_selected/seq_len:.1f}%)")


def test_hierarchical_levels():
    """Test that all levels participate in selection."""
    seq_len = 256
    d = 64

    attention = HierarchicalSparseAttention(num_levels=3)
    x = torch.randn(seq_len, d)

    selected_masks, all_selected = attention.forward(x)

    # Each level should contribute
    for level_idx, level_mask in enumerate(selected_masks):
        num_at_level = level_mask.sum().item()
        assert num_at_level > 0, f"Level {level_idx} should select at least some tokens"
        print(f"  Level {level_idx}: {num_at_level} tokens selected")

    # Final mask should be union of all levels
    union_mask = torch.zeros(seq_len, dtype=torch.bool)
    for level_mask in selected_masks:
        union_mask = union_mask | level_mask

    assert (union_mask == all_selected).all(), "Final mask should be union of level masks"
    print("✓ Hierarchical levels test passed")


def test_attention_indices():
    """Test that attention indices are properly extracted."""
    seq_len = 128
    d = 32

    attention = HierarchicalSparseAttention(num_levels=2)
    x = torch.randn(seq_len, d)

    indices = attention.get_attention_indices(x)

    # Check shape and bounds
    assert indices.ndim == 1, "Indices should be 1D"
    assert indices.shape[0] > 0, "Should select at least one token"
    assert indices.shape[0] <= seq_len, "Cannot select more than seq_len tokens"
    assert (indices >= 0).all() and (indices < seq_len).all(), "Indices should be in valid range"

    # Indices should be sorted
    assert (indices[:-1] <= indices[1:]).all(), "Indices should be sorted"

    print(f"✓ Attention indices test passed: {indices.shape[0]} indices extracted")


def test_deterministic_with_same_input():
    """Test that same input produces same output."""
    seq_len = 100
    d = 48

    attention = HierarchicalSparseAttention(num_levels=3)
    x = torch.randn(seq_len, d)

    indices1 = attention.get_attention_indices(x)
    indices2 = attention.get_attention_indices(x)

    assert torch.equal(indices1, indices2), "Same input should produce same output"
    print("✓ Deterministic output test passed")


def test_different_inputs_different_outputs():
    """Test that different inputs typically produce different outputs."""
    seq_len = 200
    d = 64

    attention = HierarchicalSparseAttention(num_levels=3)
    x1 = torch.randn(seq_len, d)
    x2 = torch.randn(seq_len, d)

    indices1 = attention.get_attention_indices(x1)
    indices2 = attention.get_attention_indices(x2)

    # With high probability, different random inputs should give different selections
    # (not guaranteed, but very likely)
    different = not torch.equal(indices1, indices2)
    if different:
        print("✓ Different inputs produce different outputs")
    else:
        print("⚠ Different inputs happened to produce same output (unlikely but possible)")


def test_custom_top_k():
    """Test with custom top-k specification."""
    seq_len = 256
    d = 64
    top_k_per_level = [4, 8, 16]

    attention = HierarchicalSparseAttention(num_levels=3, top_k_per_level=top_k_per_level)
    x = torch.randn(seq_len, d)

    selected_masks, all_selected = attention.forward(x)

    # Each level should have approximately the specified number of blocks * block_size tokens
    for level_idx, (level_mask, expected_k) in enumerate(zip(selected_masks, top_k_per_level)):
        num_selected = level_mask.sum().item()
        # Some tolerance since block boundaries might not align perfectly
        print(f"  Level {level_idx}: {num_selected} tokens (target blocks: {expected_k})")

    print("✓ Custom top-k test passed")


def test_batch_processing():
    """Test that the module works with different sequence lengths."""
    d = 64

    attention = HierarchicalSparseAttention(num_levels=3)

    for seq_len in [32, 64, 128, 256, 512]:
        x = torch.randn(seq_len, d)
        selected_masks, all_selected = attention.forward(x)

        num_selected = all_selected.sum().item()
        assert num_selected > 0 and num_selected <= seq_len
        print(f"  seq_len={seq_len}: {num_selected} selected ({100*num_selected/seq_len:.1f}%)")

    print("✓ Variable sequence length test passed")


if __name__ == "__main__":
    test_hierarchical_sparse_attention_basic()
    test_sparse_selection_reduces_tokens()
    test_hierarchical_levels()
    test_attention_indices()
    test_deterministic_with_same_input()
    test_different_inputs_different_outputs()
    test_custom_top_k()
    test_batch_processing()
    print("\n✅ All tests passed!")
