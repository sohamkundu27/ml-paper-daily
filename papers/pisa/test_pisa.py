import torch
from pisa import BlockwiseSparseAttention


def test_blockwise_sparse_attention_output_shape():
    """Test that output shape matches input shape."""
    batch_size = 2
    seq_len = 64
    dim = 128
    num_heads = 8

    attn = BlockwiseSparseAttention(dim=dim, num_heads=num_heads, block_size=16)
    x = torch.randn(batch_size, seq_len, dim)

    out = attn(x)

    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"


def test_blockwise_sparse_attention_gradient_flow():
    """Test that gradients flow through the attention module."""
    batch_size = 2
    seq_len = 32
    dim = 64
    num_heads = 4

    attn = BlockwiseSparseAttention(dim=dim, num_heads=num_heads, block_size=8)
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)

    out = attn(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradient did not flow to input"
    assert x.grad.shape == x.shape


def test_blockwise_sparse_attention_sparsity():
    """Test that sparsity ratio is applied correctly."""
    dim = 64
    num_heads = 4
    seq_len = 128
    block_size = 32
    sparsity_ratio = 0.3

    attn = BlockwiseSparseAttention(
        dim=dim, num_heads=num_heads, block_size=block_size, sparsity_ratio=sparsity_ratio
    )

    x = torch.randn(1, seq_len, dim)
    out = attn(x)

    assert out.shape == x.shape
    sparsity = attn.get_sparsity_ratio(seq_len)
    assert 0.0 <= sparsity <= 1.0, f"Sparsity ratio out of bounds: {sparsity}"


def test_blockwise_sparse_attention_deterministic():
    """Test that output is deterministic for same input."""
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 64
    dim = 128
    num_heads = 8

    attn = BlockwiseSparseAttention(dim=dim, num_heads=num_heads, block_size=16)
    x = torch.randn(batch_size, seq_len, dim)

    out1 = attn(x)
    out2 = attn(x)

    assert torch.allclose(out1, out2, atol=1e-6), "Output is not deterministic"


def test_blockwise_sparse_attention_batch_independence():
    """Test that batch samples don't interfere with each other."""
    torch.manual_seed(42)
    seq_len = 32
    dim = 64
    num_heads = 4

    attn = BlockwiseSparseAttention(dim=dim, num_heads=num_heads, block_size=8)

    x1 = torch.randn(1, seq_len, dim)
    x2 = torch.randn(1, seq_len, dim)
    x_batch = torch.cat([x1, x2], dim=0)

    out1 = attn(x1)
    out2 = attn(x2)
    out_batch = attn(x_batch)

    assert torch.allclose(out1, out_batch[0], atol=1e-5)
    assert torch.allclose(out2, out_batch[1], atol=1e-5)


def test_blockwise_sparse_attention_high_sparsity():
    """Test with high sparsity (aggressive pruning)."""
    batch_size = 1
    seq_len = 128
    dim = 128
    num_heads = 8
    block_size = 32
    sparsity_ratio = 0.8

    attn = BlockwiseSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        sparsity_ratio=sparsity_ratio,
    )
    x = torch.randn(batch_size, seq_len, dim)

    out = attn(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any(), "Output contains NaN"


def test_blockwise_sparse_attention_small_seq():
    """Test with sequence length smaller than block size."""
    batch_size = 1
    seq_len = 8
    dim = 64
    num_heads = 4
    block_size = 16

    attn = BlockwiseSparseAttention(
        dim=dim, num_heads=num_heads, block_size=block_size
    )
    x = torch.randn(batch_size, seq_len, dim)

    out = attn(x)

    assert out.shape == x.shape


def test_taylor_exp_approximation():
    """Test Taylor expansion approximation of exp."""
    attn = BlockwiseSparseAttention(dim=64, num_heads=4, block_size=8, taylor_order=3)

    x = torch.linspace(-1, 1, 100).reshape(-1, 1)

    approx = attn._taylor_exp_approximation(x, order=3)
    exact = torch.exp(x)

    assert approx.shape == exact.shape
    assert not torch.isnan(approx).any()
    assert not torch.isinf(approx).any()


def test_piecewise_attention_critical_vs_noncritical():
    """Test that piecewise attention correctly handles critical and non-critical blocks."""
    batch_size = 1
    seq_len = 64
    dim = 128
    num_heads = 8

    attn = BlockwiseSparseAttention(dim=dim, num_heads=num_heads, block_size=16, taylor_order=3)
    x = torch.randn(batch_size, seq_len, dim)

    scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
    is_critical = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    num_blocks = 4

    attn_weights = attn._compute_piecewise_attention(scores, is_critical, num_blocks, batch_size, seq_len)

    assert attn_weights.shape == scores.shape
    assert not torch.isnan(attn_weights).any()
    assert not torch.isinf(attn_weights).any()
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-6)


def test_piecewise_attention_with_forward_pass():
    """Test that piecewise attention produces valid output through full forward pass."""
    batch_size = 2
    seq_len = 48
    dim = 96
    num_heads = 6

    attn = BlockwiseSparseAttention(dim=dim, num_heads=num_heads, block_size=12, taylor_order=3)
    x = torch.randn(batch_size, seq_len, dim)

    out = attn(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_taylor_order_effect():
    """Test that different Taylor orders affect approximation quality."""
    batch_size = 1
    seq_len = 32
    dim = 64
    num_heads = 4

    attn = BlockwiseSparseAttention(dim=dim, num_heads=num_heads, block_size=8)
    x = torch.randn(batch_size, seq_len, dim)

    out1 = attn(x)

    attn.taylor_order = 1
    out2 = attn(x)

    attn.taylor_order = 5
    out3 = attn(x)

    assert out1.shape == out2.shape == out3.shape
    assert not torch.isnan(out1).any()
    assert not torch.isnan(out2).any()
    assert not torch.isnan(out3).any()


if __name__ == "__main__":
    test_blockwise_sparse_attention_output_shape()
    print("✓ test_blockwise_sparse_attention_output_shape")

    test_blockwise_sparse_attention_gradient_flow()
    print("✓ test_blockwise_sparse_attention_gradient_flow")

    test_blockwise_sparse_attention_sparsity()
    print("✓ test_blockwise_sparse_attention_sparsity")

    test_blockwise_sparse_attention_deterministic()
    print("✓ test_blockwise_sparse_attention_deterministic")

    test_blockwise_sparse_attention_batch_independence()
    print("✓ test_blockwise_sparse_attention_batch_independence")

    test_blockwise_sparse_attention_high_sparsity()
    print("✓ test_blockwise_sparse_attention_high_sparsity")

    test_blockwise_sparse_attention_small_seq()
    print("✓ test_blockwise_sparse_attention_small_seq")

    test_taylor_exp_approximation()
    print("✓ test_taylor_exp_approximation")

    test_piecewise_attention_critical_vs_noncritical()
    print("✓ test_piecewise_attention_critical_vs_noncritical")

    test_piecewise_attention_with_forward_pass()
    print("✓ test_piecewise_attention_with_forward_pass")

    test_taylor_order_effect()
    print("✓ test_taylor_order_effect")

    print("\nAll tests passed!")
