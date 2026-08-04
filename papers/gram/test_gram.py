import torch
import torch.optim as optim
from gram import GRAMPass1, GRAMPass2


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


def test_gram_pass2_forward():
    """Test Pass 2 forward pass with stochastic transitions"""
    model = GRAMPass2(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    x = torch.randn(4, 5)  # batch_size=4, input_dim=5

    # Test with sampling
    output, latent_trajectory, kl_loss = model(x, sample=True)

    assert output.shape == (4, 5), f"Expected output shape (4, 5), got {output.shape}"
    assert len(latent_trajectory) == 4, f"Expected 4 latent states, got {len(latent_trajectory)}"
    assert latent_trajectory[0].shape == (4, 16), f"Expected latent shape (4, 16)"
    assert kl_loss.item() >= 0, "KL loss should be non-negative"

    # Test deterministic mode (use mean)
    output_det, latent_trajectory_det, kl_loss_det = model(x, sample=False)
    assert output_det.shape == (4, 5), "Deterministic output shape mismatch"
    assert kl_loss_det.item() >= 0, "KL loss should be non-negative in deterministic mode"

    print("✓ Pass 2 forward pass test passed")


def test_gram_pass2_stochasticity():
    """Test that stochastic mode produces different trajectories"""
    model = GRAMPass2(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    x = torch.randn(1, 5)

    # Sample multiple trajectories
    outputs = []
    for _ in range(5):
        output, _, _ = model(x, sample=True)
        outputs.append(output)

    outputs = torch.stack(outputs)

    # Check that not all outputs are identical (stochasticity is working)
    # Compute pairwise distances
    distances = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            dist = torch.norm(outputs[i] - outputs[j]).item()
            distances.append(dist)

    mean_distance = sum(distances) / len(distances)
    assert mean_distance > 0.01, f"Stochastic trajectories should differ, but mean distance is {mean_distance:.6f}"

    print(f"✓ Stochasticity test passed (mean trajectory distance: {mean_distance:.6f})")


def test_gram_pass2_kl_loss():
    """Test that KL loss is computed and has expected properties"""
    model = GRAMPass2(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    x = torch.randn(4, 5)

    output, _, kl_loss = model(x, sample=True)

    # KL loss should be positive (measuring divergence from standard normal)
    assert kl_loss.item() > 0, f"KL loss should be positive, got {kl_loss.item()}"

    # KL loss should be accumulated over num_steps
    # (one KL per transition, so 3 steps = sum of 3 KL terms)
    assert kl_loss.item() < 100, f"KL loss seems too large: {kl_loss.item()}"

    print(f"✓ KL loss test passed (KL loss: {kl_loss.item():.4f})")


def test_gram_pass2_training():
    """Test that Pass 2 can learn with both reconstruction and KL terms"""
    model = GRAMPass2(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Simple reconstruction task: output should match input
    x = torch.randn(8, 5)

    losses = []
    for step in range(100):
        output, _, kl_loss = model(x, sample=True)
        total_loss, recon_loss, _ = model.compute_loss(output, x, kl_loss, kl_weight=0.01)
        losses.append(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 25 == 0:
            print(f"  Step {step}: loss = {total_loss.item():.4f} (recon: {recon_loss.item():.4f})")

    # Loss should decrease
    initial_loss = losses[0]
    final_loss = losses[-1]
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    print(f"✓ Pass 2 training test passed (loss: {initial_loss:.4f} -> {final_loss:.4f})")


def test_gram_pass2_constraint_satisfaction():
    """
    Test Pass 2 on constraint satisfaction task with stochastic reasoning.
    """
    model = GRAMPass2(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    def create_target(x):
        """Create target that satisfies constraints"""
        x_clamped = torch.clamp(x, 0, 2)
        sums = x_clamped.sum(dim=1, keepdim=True)
        x_normalized = x_clamped * (5.0 / (sums + 1e-6))
        return torch.clamp(x_normalized, 0, 2)

    # Create synthetic dataset
    num_samples = 16
    x_noisy = torch.randn(num_samples, 5) * 2
    x_target = create_target(x_noisy)

    # Train with stochastic transitions
    losses = []
    for step in range(150):
        output, _, kl_loss = model(x_noisy, sample=True)
        total_loss, recon_loss, _ = model.compute_loss(output, x_target, kl_loss, kl_weight=0.01)
        losses.append(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Verify loss decreases
    initial_loss = losses[0]
    final_loss = losses[-1]
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    # Verify model output satisfies constraints
    with torch.no_grad():
        output, _, _ = model(x_noisy, sample=False)
        in_range = (output >= -0.1) & (output <= 2.1)
        in_range_ratio = in_range.float().mean().item()
        assert in_range_ratio > 0.7, f"Expected >70% in range, got {in_range_ratio*100:.1f}%"

    print(
        f"✓ Pass 2 constraint satisfaction test passed "
        f"(loss: {initial_loss:.4f} -> {final_loss:.4f}, {in_range_ratio*100:.1f}% in valid range)"
    )


if __name__ == "__main__":
    print("Testing GRAM Pass 1...")
    test_gram_basic()
    test_gram_reconstruction()
    test_gram_constraint_satisfaction()

    print("\nTesting GRAM Pass 2 (Stochastic Core)...")
    test_gram_pass2_forward()
    test_gram_pass2_stochasticity()
    test_gram_pass2_kl_loss()
    test_gram_pass2_training()
    test_gram_pass2_constraint_satisfaction()

    print("\n✅ All tests passed!")
