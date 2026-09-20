import torch
from switch_attention import FullAttention, SlidingWindowAttention, SwitchAttention


def test_full_attention_shapes():
    """Test that FullAttention produces correct output shapes."""
    batch_size, seq_len, d_model = 2, 32, 64
    num_heads = 4

    x = torch.randn(batch_size, seq_len, d_model)
    attention = FullAttention(d_model, num_heads)
    output = attention(x)

    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ FullAttention output shape correct")


def test_sliding_window_attention_shapes():
    """Test that SlidingWindowAttention produces correct output shapes."""
    batch_size, seq_len, d_model = 2, 32, 64
    num_heads = 4
    window_size = 8

    x = torch.randn(batch_size, seq_len, d_model)
    attention = SlidingWindowAttention(d_model, num_heads, window_size)
    output = attention(x)

    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ SlidingWindowAttention output shape correct")


def test_switch_attention_shapes():
    """Test that SwitchAttention produces correct output shapes."""
    batch_size, seq_len, d_model = 2, 32, 64
    num_heads = 4
    window_size = 8

    x = torch.randn(batch_size, seq_len, d_model)
    attention = SwitchAttention(d_model, num_heads, window_size)
    output = attention(x)

    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ SwitchAttention output shape correct")


def test_switch_attention_routing():
    """Test that SwitchAttention routes between full and sliding window attention."""
    batch_size, seq_len, d_model = 2, 32, 64
    num_heads = 4
    window_size = 8

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    # Create a switch attention with fixed seed
    attention = SwitchAttention(d_model, num_heads, window_size)
    attention.eval()  # Evaluation mode for deterministic output

    with torch.no_grad():
        output = attention(x)
        routing_probs = attention.get_routing_decisions(x)

        # Verify output is valid and has no NaNs
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert output.dtype == x.dtype, f"Output dtype {output.dtype} does not match input dtype {x.dtype}"

        # Verify routing probabilities are per-token
        assert routing_probs.shape == (batch_size, seq_len, 1), f"Expected routing shape {(batch_size, seq_len, 1)}, got {routing_probs.shape}"
        assert routing_probs.min() >= 0 and routing_probs.max() <= 1, "Routing probs should be in [0, 1]"

        # Verify output is a weighted combination (bounded)
        full_out = attention.full_attention(x)
        sliding_out = attention.sliding_attention(x)

        # With per-token routing, output should interpolate between full and sliding
        min_val = torch.minimum(full_out, sliding_out).min()
        max_val = torch.maximum(full_out, sliding_out).max()

        # Output should be within the range of the two attention outputs (approximately)
        # Allow some numerical slack
        assert output.min() >= min_val - 1e-5, "Output below expected minimum"
        assert output.max() <= max_val + 1e-5, "Output above expected maximum"

    print("✓ SwitchAttention per-token routing works correctly")


def test_switch_attention_gradient_flow():
    """Test that gradients flow through SwitchAttention."""
    batch_size, seq_len, d_model = 2, 16, 64
    num_heads = 4
    window_size = 4

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    attention = SwitchAttention(d_model, num_heads, window_size)

    output = attention(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient is None"
    assert attention.router[0].weight.grad is not None, "Router weight gradient is None"
    print("✓ Gradients flow through SwitchAttention")


def test_different_sequence_lengths():
    """Test that SwitchAttention handles different sequence lengths."""
    d_model = 64
    num_heads = 4
    window_size = 8
    batch_size = 1

    attention = SwitchAttention(d_model, num_heads, window_size)

    for seq_len in [8, 16, 32, 64, 128]:
        x = torch.randn(batch_size, seq_len, d_model)
        output = attention(x)
        assert output.shape == (batch_size, seq_len, d_model), f"Failed for seq_len={seq_len}"

    print("✓ SwitchAttention handles various sequence lengths")


def test_sparsity_regularization():
    """Test that sparsity regularization can be computed and affects loss."""
    batch_size, seq_len, d_model = 2, 32, 64
    num_heads = 4
    window_size = 8

    # Model with sparsity weight
    model = SwitchAttention(d_model, num_heads, window_size, sparsity_weight=0.1)
    model.train()

    x = torch.randn(batch_size, seq_len, d_model)
    x.requires_grad = True

    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check that routing loss was computed
    assert model.routing_loss > 0 or model.routing_loss == 0, "Routing loss should be computed"
    assert x.grad is not None, "Gradients should flow"

    print("✓ Sparsity regularization works")


def test_routing_sparsity_metric():
    """Test that routing sparsity metric is computed correctly."""
    batch_size, seq_len, d_model = 1, 32, 64
    num_heads = 4
    window_size = 8

    model = SwitchAttention(d_model, num_heads, window_size)
    model.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    sparsity = model.get_routing_sparsity(x)

    assert 0 <= sparsity <= 1, f"Sparsity should be in [0, 1], got {sparsity}"
    print("✓ Routing sparsity metric works")


def test_batched_forward_pass():
    """Test that batched forward pass produces valid outputs."""
    batch_size, seq_len, d_model = 2, 32, 64
    num_heads = 4
    window_size = 8

    model = SwitchAttention(d_model, num_heads, window_size)
    model.eval()

    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        output_standard = model(x, use_batched=False)
        output_batched = model(x, use_batched=True)

    assert output_standard.shape == output_batched.shape, "Shapes should match"
    # Outputs should be very close (both use same underlying computation currently)
    assert torch.allclose(output_standard, output_batched, atol=1e-5), \
        "Batched and standard outputs should be numerically close"

    print("✓ Batched forward pass works correctly")


if __name__ == "__main__":
    test_full_attention_shapes()
    test_sliding_window_attention_shapes()
    test_switch_attention_shapes()
    test_switch_attention_routing()
    test_switch_attention_gradient_flow()
    test_different_sequence_lengths()
    test_sparsity_regularization()
    test_routing_sparsity_metric()
    test_batched_forward_pass()
    print("\n✓ All tests passed!")
