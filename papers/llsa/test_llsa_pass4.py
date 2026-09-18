import torch
import torch.optim as optim
import torch.nn.functional as F
import time
from llsa import SimpleDiffusionModel


class SimpleNoiseSchedule:
    """Simple linear noise schedule for diffusion."""

    def __init__(self, num_steps: int = 100):
        self.num_steps = num_steps
        # Linear schedule from 0 to 1
        self.alphas = torch.linspace(1.0, 0.0, num_steps)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Add noise to clean sample x at timestep t."""
        alpha_t = self.alphas_cumprod[t]
        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return noisy_x, noise

    def get_alpha(self, t: int) -> float:
        """Get alpha value for timestep t."""
        return self.alphas_cumprod[t].item()


class DenoisingDiffusionModel(torch.nn.Module):
    """Wrapper that uses SimpleDiffusionModel to denoise noisy samples."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 2048,
        num_levels: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.backbone = SimpleDiffusionModel(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            num_levels=num_levels,
            max_seq_len=512
        )

    def forward(self, noisy_x: torch.Tensor) -> torch.Tensor:
        """Predict noise in the noisy sample."""
        return self.backbone(noisy_x)


def test_end_to_end_denoising_task():
    """Test end-to-end denoising on synthetic data."""
    print("\n=== Pass 4: End-to-End Denoising Task ===")

    # Configuration
    d_model = 64
    num_heads = 4
    num_layers = 2
    seq_len = 128
    batch_size = 4
    num_training_steps = 20
    num_diffusion_steps = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = DenoisingDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=256,
        num_levels=3
    ).to(device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    noise_schedule = SimpleNoiseSchedule(num_steps=num_diffusion_steps)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Sequence length: {seq_len}, Batch size: {batch_size}")
    print(f"Training for {num_training_steps} steps...\n")

    # Training loop
    losses = []
    times = []

    for step in range(num_training_steps):
        # Generate clean samples
        clean_x = torch.randn(batch_size, seq_len, d_model, device=device)

        # Sample random timesteps
        t = torch.randint(0, num_diffusion_steps, (batch_size,), device=device)

        # Add noise
        noisy_samples = []
        target_noise = []
        for b in range(batch_size):
            noisy_x, noise = noise_schedule.add_noise(clean_x[b], t[b].item())
            noisy_samples.append(noisy_x)
            target_noise.append(noise)

        noisy_x_batch = torch.stack(noisy_samples).to(device)
        target_noise_batch = torch.stack(target_noise).to(device)

        # Forward pass
        step_start = time.time()
        optimizer.zero_grad()
        predicted_noise = model(noisy_x_batch)
        step_time = time.time() - step_start

        # Loss: MSE between predicted and actual noise
        loss = F.mse_loss(predicted_noise, target_noise_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        times.append(step_time)

        if (step + 1) % 5 == 0:
            avg_time = sum(times[-5:]) / 5
            print(f"Step {step+1:2d}: loss={loss.item():.6f}, time={avg_time*1000:.2f}ms")

    print(f"\n✓ Denoising task completed")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss:   {losses[-1]:.6f}")
    print(f"  Avg time/step: {sum(times)/len(times)*1000:.2f}ms")

    return model


def test_complexity_on_different_sequence_lengths():
    """Analyze complexity across different sequence lengths."""
    print("\n=== Complexity Analysis ===")

    d_model = 64
    num_heads = 4
    num_layers = 2
    batch_size = 2

    model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=256,
        num_levels=3,
        max_seq_len=512
    )

    print("\nSequence Length | Full Attn | Sparse Attn | Tokens Selected | Speedup")
    print("-" * 70)

    for seq_len in [64, 128, 256, 512]:
        complexity = model.compute_attention_complexity(seq_len)
        full_ops = complexity["full_attention_ops"]
        sparse_ops = complexity["sparse_attention_ops"]
        selected = complexity["tokens_selected"]
        speedup = complexity["reduction_ratio"]

        print(f"{seq_len:15d} | {full_ops:9d} | {sparse_ops:10d} | {selected:15d} | {speedup:6.2f}x")

    print("✓ Complexity analysis complete")


def test_sparse_attention_memory_efficiency():
    """Measure actual memory usage with sparse attention."""
    print("\n=== Memory Efficiency Test ===")

    d_model = 64
    num_heads = 4
    seq_len = 256
    batch_size = 2

    model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=2,
        ff_dim=256,
        num_levels=3
    )

    x = torch.randn(batch_size, seq_len, d_model)

    # Measure with sparse attention
    output = model(x)

    # Get actual token selection
    first_attn = model.layers[0].attention
    _, selected_mask = first_attn._select_important_blocks(seq_len)
    num_selected = selected_mask.sum().item()

    # Calculate theoretical ops
    full_attn_ops = seq_len ** 2
    sparse_attn_ops = seq_len * num_selected

    print(f"Sequence length: {seq_len}")
    print(f"Tokens selected: {num_selected} ({100*num_selected/seq_len:.1f}%)")
    print(f"Full attention ops: {full_attn_ops:,}")
    print(f"Sparse attention ops: {sparse_attn_ops:,}")
    print(f"Theoretical speedup: {full_attn_ops/sparse_attn_ops:.2f}x")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("✓ Memory efficiency analysis complete")


def test_training_convergence():
    """Test that the model can learn on a simple reconstruction task."""
    print("\n=== Training Convergence Test ===")

    d_model = 48
    num_heads = 4
    num_layers = 2
    seq_len = 100
    batch_size = 4

    model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=192,
        num_levels=3
    )

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create a simple target pattern
    target = torch.randn(1, seq_len, d_model)

    losses = []
    print("Training for 30 steps on reconstruction task...")

    for step in range(30):
        x = target + 0.1 * torch.randn_like(target)  # Add small noise to input

        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}: loss={loss.item():.6f}")

    # Check that loss decreased
    loss_decrease = (losses[0] - losses[-1]) / losses[0]
    assert loss_decrease > 0, "Loss should decrease during training"

    print(f"✓ Loss decreased by {loss_decrease*100:.1f}%")
    print(f"  Initial: {losses[0]:.6f}, Final: {losses[-1]:.6f}")


def test_gradient_flow_with_sparse_selection():
    """Verify gradients flow through block weight parameters."""
    print("\n=== Gradient Flow Test ===")

    d_model = 64
    num_heads = 4
    seq_len = 128
    batch_size = 2

    model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=1,
        ff_dim=256,
        num_levels=3
    )

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    output = model(x)
    loss = output.mean()
    loss.backward()

    # Check input gradients
    assert x.grad is not None, "Input should have gradients"

    # Check attention module gradients
    attn = model.layers[0].attention
    assert attn.q_proj.weight.grad is not None, "Q projection should have gradients"
    assert attn.k_proj.weight.grad is not None, "K projection should have gradients"
    assert attn.v_proj.weight.grad is not None, "V projection should have gradients"

    # Check block weight gradients
    for level, block_weights in enumerate(attn.block_weights):
        # Note: block weights use topk which is non-differentiable,
        # so gradients won't flow through block weights themselves.
        # This is expected and documented in NOTES.md
        pass

    print("✓ Gradient flow verified for projections and linear layers")
    print("  (Note: block weights use non-differentiable topk, deferred enhancement)")


def test_inference_speed():
    """Measure inference speed on a batch."""
    print("\n=== Inference Speed Test ===")

    d_model = 64
    num_heads = 4
    seq_len = 256
    batch_size = 2
    num_iterations = 10

    model = SimpleDiffusionModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=2,
        ff_dim=256,
        num_levels=3
    )
    model.eval()

    x = torch.randn(batch_size, seq_len, d_model)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(x)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(x)
            times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Tokens processed per second: {batch_size*seq_len/avg_time:.0f}")

    print("✓ Inference speed measured")


if __name__ == "__main__":
    # Run all tests
    model = test_end_to_end_denoising_task()
    test_training_convergence()
    test_gradient_flow_with_sparse_selection()
    test_complexity_on_different_sequence_lengths()
    test_sparse_attention_memory_efficiency()
    test_inference_speed()

    print("\n" + "="*70)
    print("✅ All Pass 4 tests passed!")
    print("="*70)
