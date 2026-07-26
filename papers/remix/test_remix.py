import torch
import torch.nn as nn
from remix import (LoRALayer, LoRALinear, SimpleRouter, MixtureOfLoRAs,
                   LearnedRouter, MixtureOfLoRAsRL)


def test_lora_layer_forward():
    """Test LoRA layer forward pass and output shape."""
    in_features, out_features, rank = 64, 32, 8
    lora = LoRALayer(in_features, out_features, rank, alpha=1.0)

    x = torch.randn(4, in_features)  # (batch_size=4, in_features)
    output = lora(x)

    assert output.shape == (4, out_features), f"Expected shape (4, {out_features}), got {output.shape}"
    print("✓ LoRA layer forward pass shape test passed")


def test_lora_layer_gradients():
    """Test that LoRA parameters receive gradients."""
    in_features, out_features, rank = 64, 32, 8
    lora = LoRALayer(in_features, out_features, rank, alpha=1.0)

    x = torch.randn(4, in_features)
    y = lora(x)
    loss = y.sum()
    loss.backward()

    assert lora.lora_a.grad is not None, "lora_a should have gradients"
    assert lora.lora_b.grad is not None, "lora_b should have gradients"
    assert lora.lora_a.grad.shape == lora.lora_a.shape
    assert lora.lora_b.grad.shape == lora.lora_b.shape
    print("✓ LoRA layer gradient test passed")


def test_lora_linear():
    """Test LoRA-injected linear layer."""
    in_features, out_features, rank = 64, 32, 8
    lora_linear = LoRALinear(in_features, out_features, rank, alpha=0.5)

    x = torch.randn(4, in_features)
    output = lora_linear(x)

    assert output.shape == (4, out_features)
    # Base weights should not have gradients
    assert lora_linear.base_linear.weight.requires_grad == False
    # LoRA weights should have gradients
    assert lora_linear.lora.lora_a.requires_grad == True
    assert lora_linear.lora.lora_b.requires_grad == True
    print("✓ LoRA linear layer test passed")


def test_simple_router():
    """Test the simple uniform router."""
    num_loras = 5
    num_active = 2
    router = SimpleRouter(num_loras, num_active)

    batch_size = 4
    routing = router(batch_size)

    assert routing.shape == (batch_size, num_loras), f"Expected shape ({batch_size}, {num_loras})"
    # Check that each row has exactly num_active ones
    row_sums = routing.sum(dim=1)
    assert torch.allclose(row_sums, torch.full_like(row_sums, num_active)), \
        f"Each row should sum to {num_active}, got {row_sums}"
    # Check binary values
    assert torch.all((routing == 0) | (routing == 1)), "Router should output binary values"
    print("✓ Simple router test passed")


def test_mixture_of_loras_forward():
    """Test mixture of LoRAs forward pass."""
    in_features, out_features = 64, 32
    num_loras, num_active, rank = 4, 2, 8

    mixture = MixtureOfLoRAs(in_features, out_features, num_loras, num_active, rank, alpha=1.0)

    x = torch.randn(4, in_features)
    output, routing = mixture(x)

    assert output.shape == (4, out_features), f"Expected output shape (4, {out_features})"
    assert routing.shape == (4, num_loras), f"Expected routing shape (4, {num_loras})"
    print("✓ Mixture of LoRAs forward pass test passed")


def test_mixture_of_loras_gradients():
    """Test that LoRA parameters in mixture receive gradients."""
    in_features, out_features = 64, 32
    num_loras, num_active, rank = 4, 2, 8

    mixture = MixtureOfLoRAs(in_features, out_features, num_loras, num_active, rank, alpha=1.0)

    x = torch.randn(4, in_features)
    output, routing = mixture(x)
    loss = output.sum()
    loss.backward()

    # Check that at least one LoRA has received gradients
    has_grad = False
    for lora in mixture.loras:
        if lora.lora_a.grad is not None:
            has_grad = True
            break
    assert has_grad, "At least one LoRA should receive gradients"
    print("✓ Mixture of LoRAs gradient test passed")


def test_mixture_routing_statistics():
    """Test that routing produces expected statistics."""
    in_features, out_features = 64, 32
    num_loras, num_active, rank = 10, 3, 8

    mixture = MixtureOfLoRAs(in_features, out_features, num_loras, num_active, rank, alpha=1.0)

    # Generate multiple batches to get routing statistics
    all_routing = []
    for _ in range(100):
        x = torch.randn(8, in_features)
        _, routing = mixture(x)
        all_routing.append(routing)

    all_routing = torch.cat(all_routing, dim=0)

    # Each column (LoRA) should be selected roughly equally often
    column_counts = all_routing.sum(dim=0)
    expected_count = all_routing.shape[0] * num_active / num_loras
    # Allow some variance (each should be within 30% of expected)
    assert (column_counts > expected_count * 0.7).all() and (column_counts < expected_count * 1.3).all(), \
        f"Routing should be roughly uniform. Counts: {column_counts}, expected ~{expected_count}"
    print("✓ Mixture routing statistics test passed")


def test_end_to_end():
    """Test a complete forward-backward pass."""
    in_features, out_features = 128, 64
    num_loras, num_active, rank = 6, 3, 16

    mixture = MixtureOfLoRAs(in_features, out_features, num_loras, num_active, rank, alpha=0.5)

    # Simulate a simple task: predict one value from input
    x = torch.randn(8, in_features)
    y_target = torch.randn(8, out_features)

    # Forward pass
    output, _ = mixture(x)
    loss = nn.MSELoss()(output, y_target)

    # Backward pass
    loss.backward()

    # Verify gradients exist
    for lora in mixture.loras:
        assert lora.lora_a.grad is not None
        assert lora.lora_b.grad is not None

    # Verify base weights don't have gradients
    assert mixture.base_linear.weight.grad is None
    assert mixture.base_linear.bias.grad is None

    print("✓ End-to-end test passed")
    print(f"  Loss value: {loss.item():.4f}")


def test_learned_router_forward():
    """Test learned router forward pass and probability output."""
    in_features, num_loras, num_active = 64, 5, 2
    router = LearnedRouter(in_features, num_loras, num_active)

    x = torch.randn(4, in_features)
    routing, probs = router.get_routing(x)

    assert routing.shape == (4, num_loras), f"Expected routing shape (4, {num_loras})"
    assert probs.shape == (4, num_loras), f"Expected probs shape (4, {num_loras})"
    # Check that routing is binary with exactly num_active ones per sample
    row_sums = routing.sum(dim=1)
    assert torch.allclose(row_sums, torch.full_like(row_sums, num_active)), \
        f"Each routing row should have {num_active} ones, got {row_sums}"
    # Check probabilities sum to 1
    assert torch.allclose(probs.sum(dim=1), torch.ones(4)), "Probabilities should sum to 1"
    print("✓ Learned router forward pass test passed")


def test_learned_router_policy_loss():
    """Test that policy loss is computable and flows gradients."""
    in_features, num_loras, num_active = 64, 5, 2
    router = LearnedRouter(in_features, num_loras, num_active)

    x = torch.randn(4, in_features)
    routing, probs = router.get_routing(x)

    # Simulate task loss (per-sample)
    task_loss = torch.tensor([0.5, 0.6, 0.4, 0.7])

    policy_loss = router.compute_policy_loss(probs, routing, task_loss)

    assert policy_loss.requires_grad, "Policy loss should be differentiable"
    assert policy_loss.shape == torch.Size([]), f"Policy loss should be scalar, got {policy_loss.shape}"

    # Test backward
    policy_loss.backward()
    has_grad = False
    for param in router.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "Router parameters should receive gradients"
    print("✓ Learned router policy loss test passed")


def test_learned_router_load_statistics():
    """Test that load balancing statistics are tracked."""
    in_features, num_loras, num_active = 64, 5, 2
    router = LearnedRouter(in_features, num_loras, num_active)

    # Generate multiple batches and compute policy loss to update statistics
    for _ in range(10):
        x = torch.randn(4, in_features)
        routing, probs = router.get_routing(x)
        task_loss = torch.tensor([0.5, 0.6, 0.4, 0.7])
        # This call updates the load statistics
        _ = router.compute_policy_loss(probs, routing, task_loss)

    load_stats = router.get_load_statistics()
    assert load_stats.shape == (num_loras,), f"Expected shape ({num_loras},)"
    # Load should be roughly balanced
    assert load_stats.sum() > 0, "At least some load should be recorded"
    print(f"✓ Learned router load statistics test passed (load per LoRA: {load_stats})")


def test_mixture_of_loras_rl_forward():
    """Test MixtureOfLoRAsRL forward pass."""
    in_features, out_features = 64, 32
    num_loras, num_active, rank = 4, 2, 8

    mixture = MixtureOfLoRAsRL(in_features, out_features, num_loras, num_active, rank, alpha=1.0)

    x = torch.randn(4, in_features)
    output, routing, probs = mixture(x)

    assert output.shape == (4, out_features), f"Expected output shape (4, {out_features})"
    assert routing.shape == (4, num_loras), f"Expected routing shape (4, {num_loras})"
    assert probs.shape == (4, num_loras), f"Expected probs shape (4, {num_loras})"
    print("✓ Mixture of LoRAs RL forward pass test passed")


def test_mixture_of_loras_rl_training():
    """Test MixtureOfLoRAsRL training with loss computation."""
    in_features, out_features = 64, 32
    num_loras, num_active, rank = 4, 2, 8

    mixture = MixtureOfLoRAsRL(in_features, out_features, num_loras, num_active, rank, alpha=1.0)
    optimizer = torch.optim.Adam(mixture.parameters(), lr=0.001)

    x = torch.randn(8, in_features)
    y_target = torch.randn(8, out_features)

    # Training step
    for _ in range(3):
        optimizer.zero_grad()
        total_loss, task_loss = mixture.compute_loss(x, y_target)
        total_loss.backward()
        optimizer.step()

    # Verify losses are finite and computable
    assert task_loss.item() > 0, "Task loss should be positive"
    assert not torch.isnan(total_loss), "Total loss should not be NaN"
    assert not torch.isinf(total_loss), "Total loss should not be infinite"

    # Verify gradients flowed
    for param in mixture.loras[0].parameters():
        assert param.grad is not None, "LoRA parameters should have gradients"
    print("✓ Mixture of LoRAs RL training test passed")


def test_mixture_of_loras_rl_load_balancing():
    """Test that RL training encourages load balancing."""
    in_features, out_features = 64, 32
    num_loras, num_active, rank = 6, 2, 8

    mixture = MixtureOfLoRAsRL(in_features, out_features, num_loras, num_active, rank,
                              alpha=1.0, load_balance_weight=0.5)
    optimizer = torch.optim.Adam(mixture.parameters(), lr=0.01)

    # Train for a few steps
    for _ in range(50):
        x = torch.randn(8, in_features)
        y_target = torch.randn(8, out_features)

        optimizer.zero_grad()
        total_loss, _ = mixture.compute_loss(x, y_target)
        total_loss.backward()
        optimizer.step()

    # Check load statistics
    load_stats = mixture.router.get_load_statistics()
    # With load balancing, load should be more uniform
    # All should be reasonably loaded (no complete collapse)
    assert (load_stats > 0.1).all(), "All LoRAs should have some load with balancing"
    print(f"✓ Mixture of LoRAs RL load balancing test passed (load: {load_stats})")


def test_mixture_of_loras_rl_compared_to_simple():
    """Compare learned vs simple router behavior."""
    in_features, out_features = 64, 32
    num_loras, num_active, rank = 4, 2, 8

    # Simple router
    simple_mixture = MixtureOfLoRAs(in_features, out_features, num_loras, num_active, rank)

    # Learned router
    learned_mixture = MixtureOfLoRAsRL(in_features, out_features, num_loras, num_active, rank)

    x = torch.randn(4, in_features)

    simple_out, simple_routing = simple_mixture(x)
    learned_out, learned_routing, learned_probs = learned_mixture(x)

    # Outputs should have same shape
    assert simple_out.shape == learned_out.shape, "Output shapes should match"

    # Both should have binary routing
    assert (simple_routing == 0).sum() + (simple_routing == 1).sum() == simple_routing.numel()
    assert (learned_routing == 0).sum() + (learned_routing == 1).sum() == learned_routing.numel()

    print("✓ Learned vs simple router comparison test passed")


if __name__ == "__main__":
    test_lora_layer_forward()
    test_lora_layer_gradients()
    test_lora_linear()
    test_simple_router()
    test_mixture_of_loras_forward()
    test_mixture_of_loras_gradients()
    test_mixture_routing_statistics()
    test_end_to_end()

    # Pass 2 tests
    test_learned_router_forward()
    test_learned_router_policy_loss()
    test_learned_router_load_statistics()
    test_mixture_of_loras_rl_forward()
    test_mixture_of_loras_rl_training()
    test_mixture_of_loras_rl_load_balancing()
    test_mixture_of_loras_rl_compared_to_simple()

    print("\n✅ All tests passed!")
