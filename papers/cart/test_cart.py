"""
Tests for CART Pass 1: Core MLA block with LTI gate
"""

import torch
from cart import CartMLABlock, LearnedLTIGate, CartRecurrentCore, create_dummy_context


def test_mla_block_forward_shape():
    """Test that MLA block produces correct output shape."""
    batch, seq_len, dim, ctx_len, head_dim = 2, 10, 64, 20, 32

    mla = CartMLABlock(dim, head_dim)
    x = torch.randn(batch, seq_len, dim)
    context_k, context_v = create_dummy_context(batch, seq_len, ctx_len, head_dim)

    output = mla(x, context_k, context_v)

    assert output.shape == (batch, seq_len, dim), f"Expected shape {(batch, seq_len, dim)}, got {output.shape}"


def test_mla_block_gradient_flow():
    """Test that gradients flow through MLA block."""
    batch, seq_len, dim, ctx_len, head_dim = 2, 10, 64, 20, 32

    mla = CartMLABlock(dim, head_dim)
    x = torch.randn(batch, seq_len, dim, requires_grad=True)
    context_k, context_v = create_dummy_context(batch, seq_len, ctx_len, head_dim)

    output = mla(x, context_k, context_v)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Gradient not computed for input"
    assert mla.q_proj.weight.grad is not None, "Gradient not computed for q_proj"
    assert mla.out_proj.weight.grad is not None, "Gradient not computed for out_proj"


def test_lti_gate_spectral_radius():
    """Test that LTI gate maintains spectral radius in stable range."""
    gate = LearnedLTIGate(init_alpha=0.8)

    # Check initial spectral radius
    rho_init = gate.get_spectral_radius()
    assert 0.0 <= rho_init < 1.0, f"Spectral radius {rho_init} out of stable range [0, 1)"

    # Test recurrent update
    batch, seq_len, dim = 2, 10, 64
    x = torch.randn(batch, seq_len, dim)
    y = torch.randn(batch, seq_len, dim)

    x_new = gate(x, y)

    # Verify x_new = alpha * x + y
    expected = rho_init * x + y
    torch.testing.assert_close(x_new, expected, rtol=1e-5, atol=1e-6)


def test_lti_gate_clipping():
    """Test that LTI gate clips alpha to valid range."""
    gate = LearnedLTIGate(init_alpha=1.5)  # Invalid: > 1

    rho = gate.get_spectral_radius()
    assert 0.0 <= rho < 1.0, f"Spectral radius {rho} not clipped to [0, 1)"


def test_lti_gate_trainable():
    """Test that LTI gate alpha is trainable."""
    gate = LearnedLTIGate(init_alpha=0.8)
    alpha_init = gate.get_spectral_radius().item()

    batch, seq_len, dim = 2, 10, 64
    x = torch.randn(batch, seq_len, dim)
    y = torch.randn(batch, seq_len, dim)

    x_new = gate(x, y)
    loss = x_new.sum()
    loss.backward()

    assert gate.alpha.grad is not None, "Gradient not computed for alpha"

    # Verify alpha changed after update
    with torch.no_grad():
        gate.alpha -= 0.01 * gate.alpha.grad

    alpha_new = gate.get_spectral_radius().item()
    assert alpha_init != alpha_new, "Alpha not updated"


def test_recurrent_core_single_iteration():
    """Test recurrent core with single iteration."""
    batch, seq_len, dim, ctx_len, head_dim = 2, 10, 64, 20, 32

    core = CartRecurrentCore(dim, head_dim, num_iterations=1)
    x_init = torch.randn(batch, seq_len, dim)
    context_k, context_v = create_dummy_context(batch, seq_len, ctx_len, head_dim)

    output = core(x_init, context_k, context_v)

    assert output.shape == (batch, seq_len, dim), f"Expected shape {(batch, seq_len, dim)}, got {output.shape}"


def test_recurrent_core_multiple_iterations():
    """Test recurrent core with multiple iterations."""
    batch, seq_len, dim, ctx_len, head_dim = 2, 10, 64, 20, 32

    core = CartRecurrentCore(dim, head_dim, num_iterations=3)
    x_init = torch.randn(batch, seq_len, dim)
    context_k, context_v = create_dummy_context(batch, seq_len, ctx_len, head_dim)

    output = core(x_init, context_k, context_v)

    assert output.shape == (batch, seq_len, dim), f"Expected shape {(batch, seq_len, dim)}, got {output.shape}"


def test_recurrent_core_stability():
    """Test that recurrent core converges due to LTI gate stability."""
    batch, seq_len, dim, ctx_len, head_dim = 1, 5, 32, 10, 16

    # Use many iterations to check stability
    core = CartRecurrentCore(dim, head_dim, num_iterations=10)

    x_init = torch.ones(batch, seq_len, dim) * 0.1
    context_k = torch.ones(batch, ctx_len, head_dim) * 0.05
    context_v = torch.ones(batch, ctx_len, head_dim) * 0.05

    output = core(x_init, context_k, context_v)

    # Check that output is not NaN or Inf (stability check)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    # Spectral radius should be < 1
    rho = core.lti_gate.get_spectral_radius()
    assert rho < 1.0, f"Spectral radius {rho} >= 1, recurrence is unstable"


if __name__ == "__main__":
    # Run tests
    test_mla_block_forward_shape()
    print("✓ test_mla_block_forward_shape")

    test_mla_block_gradient_flow()
    print("✓ test_mla_block_gradient_flow")

    test_lti_gate_spectral_radius()
    print("✓ test_lti_gate_spectral_radius")

    test_lti_gate_clipping()
    print("✓ test_lti_gate_clipping")

    test_lti_gate_trainable()
    print("✓ test_lti_gate_trainable")

    test_recurrent_core_single_iteration()
    print("✓ test_recurrent_core_single_iteration")

    test_recurrent_core_multiple_iterations()
    print("✓ test_recurrent_core_multiple_iterations")

    test_recurrent_core_stability()
    print("✓ test_recurrent_core_stability")

    print("\nAll tests passed!")
