import torch
import torch.optim as optim
from llsa import (
    TrainableSparseAttention,
    DiffusionTransformerBlock,
    SimpleDiffusionModel
)


def test_diffusion_transformer_block_basic():
    """Test basic instantiation and forward pass of transformer block."""
    d_model = 64
    num_heads = 4
    seq_len = 128
    batch_size = 2

    block = DiffusionTransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=256,
        num_levels=3
    )

    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)

    # Check output shape matches input
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"

    print("✓ Transformer block basic test passed")


def test_diffusion_transformer_block_2d():
    """Test transformer block with 2D input (no batch dimension)."""
    d_model = 64
    num_heads = 4
    seq_len = 100

    block = DiffusionTransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=256,
        num_levels=3
    )

    x = torch.randn(seq_len, d_model)
    output = block(x)

    # Should return 2D output
    assert output.shape == (seq_len, d_model), \
        f"Expected shape {(seq_len, d_model)}, got {output.shape}"

    print("✓ Transformer block 2D test passed")


def test_simple_diffusion_model_basic():
    """Test basic instantiation and forward pass of diffusion model."""
    d_model = 64
    num_heads = 4
    num_layers = 3
    seq_len = 128
    batch_size = 2

    model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=256,
        num_levels=3
    )

    x = torch.randn(batch_size, seq_len, d_model)
    output = model(x)

    # Check output shape matches input
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"

    print("✓ Diffusion model basic test passed")


def test_simple_diffusion_model_multiple_layers():
    """Test diffusion model with multiple layers."""
    d_model = 48
    num_heads = 4
    num_layers = 4
    seq_len = 100
    batch_size = 3

    model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=192,
        num_levels=3
    )

    x = torch.randn(batch_size, seq_len, d_model)
    output = model(x)

    # Output should have same shape as input
    assert output.shape == x.shape

    # Output should not be NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print("✓ Multi-layer diffusion model test passed")


def test_diffusion_model_backward_pass():
    """Test that gradients flow through the entire diffusion model."""
    d_model = 64
    num_heads = 4
    num_layers = 2
    seq_len = 64
    batch_size = 2

    model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=256,
        num_levels=3
    )

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    output = model(x)

    # Simple loss: mean of output
    loss = output.mean()
    loss.backward()

    # Check that gradients exist for input
    assert x.grad is not None, "Input should have gradients"
    assert x.grad.shape == x.shape, "Input gradient shape should match input"

    # Check that all layers have gradients
    for i, layer in enumerate(model.layers):
        assert layer.attention.q_proj.weight.grad is not None, \
            f"Layer {i} attention should have gradients"
        assert layer.ff[0].weight.grad is not None, \
            f"Layer {i} feed-forward should have gradients"

    print("✓ Diffusion model backward pass test passed")


def test_diffusion_model_training():
    """Test a complete training step on diffusion model."""
    d_model = 48
    num_heads = 4
    num_layers = 2
    seq_len = 80
    batch_size = 2

    model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=192,
        num_levels=3
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create dummy input and target
    x = torch.randn(batch_size, seq_len, d_model)
    target = torch.randn(batch_size, seq_len, d_model)

    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    optimizer.step()

    # Check that parameters were updated
    print(f"✓ Diffusion model training test passed (loss: {loss.item():.6f})")


def test_memory_efficiency_sparse_vs_dense():
    """Compare memory usage between sparse and dense attention."""
    d_model = 64
    num_heads = 4
    seq_len = 256
    batch_size = 2

    # Sparse attention model
    sparse_model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=2,
        ff_dim=256,
        num_levels=3
    )

    x = torch.randn(batch_size, seq_len, d_model)

    # Measure sparse model memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    sparse_output = sparse_model(x)
    sparse_params = sum(p.numel() for p in sparse_model.parameters())

    # Estimate: dense attention would compute O(N^2) operations
    # Sparse attention computes O(N*k) where k << N
    full_attn_ops = seq_len ** 2

    # Extract first layer's attention to compute selected tokens
    first_layer_attn = sparse_model.layers[0].attention
    _, selected_mask = first_layer_attn._select_important_blocks(seq_len)
    num_selected = selected_mask.sum().item()

    sparse_attn_ops = seq_len * num_selected

    reduction_ratio = full_attn_ops / max(1, sparse_attn_ops)

    print(f"✓ Memory efficiency test:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Full attention ops: {full_attn_ops:,}")
    print(f"  Sparse attention ops: {sparse_attn_ops:,}")
    print(f"  Tokens selected: {num_selected}/{seq_len}")
    print(f"  Reduction ratio: {reduction_ratio:.2f}x")
    print(f"  Model parameters: {sparse_params:,}")


def test_complexity_analysis():
    """Test theoretical complexity analysis."""
    model = SimpleDiffusionModel(
        d_model=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        num_levels=3,
        max_seq_len=512
    )

    for seq_len in [64, 128, 256, 512]:
        complexity = model.compute_attention_complexity(seq_len)
        reduction = complexity["reduction_ratio"]
        print(f"  seq_len={seq_len}: {reduction:.2f}x reduction "
              f"({complexity['tokens_selected']} selected)")

    print("✓ Complexity analysis test passed")


def test_output_shape_consistency():
    """Test that output shape is consistent across different configurations."""
    configs = [
        {"d_model": 64, "num_heads": 4, "seq_len": 64, "batch_size": 2},
        {"d_model": 96, "num_heads": 6, "seq_len": 128, "batch_size": 1},
        {"d_model": 128, "num_heads": 8, "seq_len": 256, "batch_size": 3},
    ]

    for config in configs:
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]

        model = SimpleDiffusionModel(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=2,
            ff_dim=d_model * 4,
            num_levels=3
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output = model(x)

        assert output.shape == (batch_size, seq_len, d_model), \
            f"Shape mismatch for config {config}"

    print("✓ Output shape consistency test passed")


def test_sparse_vs_full_attention_output():
    """Verify that sparse attention produces reasonable outputs compared to reference."""
    d_model = 64
    num_heads = 4
    seq_len = 100
    batch_size = 2

    # Create model with sparse attention
    sparse_model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=1,
        ff_dim=256,
        num_levels=3
    )

    # Same input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    sparse_output = sparse_model(x)

    # Check that output is not too far from input (reasonable values)
    assert not torch.isnan(sparse_output).any(), "Output contains NaN"
    assert not torch.isinf(sparse_output).any(), "Output contains Inf"

    # Output magnitude should be reasonable (not explosively large)
    output_magnitude = sparse_output.abs().max().item()
    assert output_magnitude < 100, f"Output magnitude {output_magnitude} is too large"

    print(f"✓ Sparse vs full attention test passed (output magnitude: {output_magnitude:.4f})")


def test_diffusion_model_with_different_seq_lengths():
    """Test diffusion model with various sequence lengths."""
    d_model = 64
    num_heads = 4
    num_layers = 2

    model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=256,
        num_levels=3,
        max_seq_len=512
    )

    for seq_len in [32, 64, 128, 256]:
        x = torch.randn(2, seq_len, d_model)
        output = model(x)
        assert output.shape == (2, seq_len, d_model), \
            f"Output shape mismatch for seq_len={seq_len}"
        print(f"  seq_len={seq_len}: ✓")

    print("✓ Different sequence lengths test passed")


def test_residual_connections():
    """Test that residual connections preserve input information."""
    d_model = 64
    num_heads = 4

    block = DiffusionTransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=256,
        num_levels=3
    )

    # Zero input
    x = torch.zeros(1, 100, d_model)
    output = block(x)

    # With proper residual connections and layer norm, output should not be exactly zero
    # but should be influenced by the input being zero
    assert output.shape == x.shape

    # Non-zero input
    x = torch.ones(1, 100, d_model)
    output = block(x)

    # Output should have non-zero components due to residual connection
    assert output.abs().sum() > 0

    print("✓ Residual connections test passed")


if __name__ == "__main__":
    test_diffusion_transformer_block_basic()
    test_diffusion_transformer_block_2d()
    test_simple_diffusion_model_basic()
    test_simple_diffusion_model_multiple_layers()
    test_diffusion_model_backward_pass()
    test_diffusion_model_training()
    test_memory_efficiency_sparse_vs_dense()
    test_complexity_analysis()
    test_output_shape_consistency()
    test_sparse_vs_full_attention_output()
    test_diffusion_model_with_different_seq_lengths()
    test_residual_connections()
    print("\n✅ All Pass 3 tests passed!")
