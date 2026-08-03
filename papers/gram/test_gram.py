import torch
import torch.optim as optim
from gram import GRAMPass1


def test_gram_basic():
    """Test basic forward pass and shapes"""
    model = GRAMPass1(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    x = torch.randn(4, 5)  # batch_size=4, input_dim=5

    output, latent_trajectory = model(x)

    # Check output shape
    assert output.shape == (4, 5), f"Expected output shape (4, 5), got {output.shape}"

    # Check latent trajectory
    assert len(latent_trajectory) == 4, f"Expected 4 latent states (initial + 3 steps), got {len(latent_trajectory)}"
    assert latent_trajectory[0].shape == (4, 16), f"Expected latent shape (4, 16), got {latent_trajectory[0].shape}"

    print("✓ Basic forward pass test passed")


def test_gram_reconstruction():
    """Test that model can learn a simple reconstruction task"""
    model = GRAMPass1(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Simple target: reconstruct identity (output should match input)
    x = torch.randn(8, 5)

    # Train for a few steps
    losses = []
    for step in range(50):
        output, _ = model(x)
        loss = model.compute_loss(output, x)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    # Loss should decrease
    initial_loss = losses[0]
    final_loss = losses[-1]
    assert (
        final_loss < initial_loss
    ), f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    print(f"✓ Reconstruction test passed (loss decreased from {initial_loss:.4f} to {final_loss:.4f})")


def test_gram_constraint_satisfaction():
    """
    Test on a simple constraint satisfaction task.

    Task: Given a noisy vector, refine it via recursive reasoning so that:
    - Elements sum to approximately 5.0
    - All elements are in [0, 2]
    """
    model = GRAMPass1(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    def create_target(x):
        """
        Create a target that satisfies constraints.
        Start with x, then clamp and renormalize.
        """
        x_clamped = torch.clamp(x, 0, 2)
        # Normalize so sum is approximately 5
        sums = x_clamped.sum(dim=1, keepdim=True)
        x_normalized = x_clamped * (5.0 / (sums + 1e-6))
        return torch.clamp(x_normalized, 0, 2)

    # Create synthetic dataset
    num_samples = 16
    x_noisy = torch.randn(num_samples, 5) * 2  # Noisy input
    x_target = create_target(x_noisy)

    # Train
    losses = []
    for step in range(100):
        output, _ = model(x_noisy)
        loss = model.compute_loss(output, x_target)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Verify loss decreases
    initial_loss = losses[0]
    final_loss = losses[-1]
    assert (
        final_loss < initial_loss
    ), f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    # Verify model output satisfies constraints reasonably well
    with torch.no_grad():
        output, _ = model(x_noisy)
        # Check that outputs are mostly in valid range
        in_range = (output >= -0.1) & (output <= 2.1)
        in_range_ratio = in_range.float().mean().item()
        assert (
            in_range_ratio > 0.8
        ), f"Expected >80% of values in range, got {in_range_ratio*100:.1f}%"

    print(
        f"✓ Constraint satisfaction test passed "
        f"(loss: {initial_loss:.4f} -> {final_loss:.4f}, "
        f"{in_range_ratio*100:.1f}% in valid range)"
    )


if __name__ == "__main__":
    print("Testing GRAM Pass 1...")
    test_gram_basic()
    test_gram_reconstruction()
    test_gram_constraint_satisfaction()
    print("\n✅ All tests passed!")
