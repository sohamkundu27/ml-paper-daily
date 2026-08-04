import torch
import torch.optim as optim
from gram import GRAMPass1, GRAMPass2, GRAMPass3


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


def test_gram_pass3_parallel_sampling():
    """Test Pass 3 parallel trajectory sampling"""
    model = GRAMPass3(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    x = torch.randn(4, 5)  # batch_size=4, input_dim=5

    trajectories = model.sample_trajectories(x, num_trajectories=5, resample=False)

    # Check we got 5 trajectories
    assert len(trajectories) == 5, f"Expected 5 trajectories, got {len(trajectories)}"

    # Check structure of each trajectory
    for traj in trajectories:
        assert traj["output"].shape == (4, 5), "Output shape mismatch"
        assert len(traj["trajectory"]) == 4, "Trajectory length should be 4 (initial + 3 steps)"
        assert traj["score"].shape == (4,), "Score shape should be (batch_size,)"

    print("✓ Pass 3 parallel sampling test passed")


def test_gram_pass3_resampling():
    """Test Pass 3 trajectory resampling based on likelihood"""
    model = GRAMPass3(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    x = torch.randn(4, 5)

    trajectories = model.sample_trajectories(x, num_trajectories=10, resample=True, keep_ratio=0.5)

    # After resampling with keep_ratio=0.5, should have ~5 trajectories
    assert (
        4 <= len(trajectories) <= 6
    ), f"Expected ~5 trajectories after resampling, got {len(trajectories)}"

    # Trajectories should be sorted by score (higher first)
    scores = [traj["score"].mean().item() for traj in trajectories]
    assert scores == sorted(scores, reverse=True), "Trajectories should be sorted by score"

    print(f"✓ Pass 3 resampling test passed (kept {len(trajectories)} trajectories)")


def test_gram_pass3_variable_depth():
    """Test Pass 3 variable recursion depth with early stopping"""
    model = GRAMPass3(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=5)
    x = torch.randn(4, 5)

    trajectories = model.forward_variable_depth(
        x, num_trajectories=5, max_depth=5, early_stopping=True, convergence_threshold=0.001
    )

    assert len(trajectories) == 5, "Should have 5 trajectories"

    # Check that trajectories have variable depths (some may converge early)
    depths = [traj["depth"] for traj in trajectories]
    print(f"  Trajectory depths: {depths}")

    # At least some trajectories should have depth <= max_depth
    assert all(d <= 5 for d in depths), "All depths should be <= max_depth"
    assert all(d > 0 for d in depths), "All depths should be > 0"

    # Check outputs
    for traj in trajectories:
        assert traj["output"].shape == (4, 5), "Output shape mismatch"

    print(f"✓ Pass 3 variable depth test passed (depths: {depths})")


def test_gram_pass3_ensemble():
    """Test Pass 3 trajectory ensemble methods"""
    model = GRAMPass3(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
    x = torch.randn(4, 5)

    trajectories = model.sample_trajectories(x, num_trajectories=5, resample=False)

    # Test mean ensemble
    ensemble_mean = model.ensemble_outputs(trajectories, method="mean")
    assert ensemble_mean.shape == (4, 5), f"Expected shape (4, 5), got {ensemble_mean.shape}"

    # Test best ensemble
    ensemble_best = model.ensemble_outputs(trajectories, method="best")
    assert ensemble_best.shape == (4, 5), f"Expected shape (4, 5), got {ensemble_best.shape}"

    # Mean ensemble should be between min and max of individual trajectories
    all_outputs = torch.stack([t["output"] for t in trajectories])
    assert (
        ensemble_mean.min() >= all_outputs.min() - 0.1
    ), "Ensemble mean should be reasonable"

    print("✓ Pass 3 ensemble test passed")


def test_gram_pass3_constraint_satisfaction():
    """
    Test Pass 3 on constraint satisfaction with trajectory ensemble.
    """
    model = GRAMPass3(input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3)
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

    # Train using ensemble of trajectories
    losses = []
    for step in range(150):
        # Sample trajectories and compute ensemble
        trajectories = model.sample_trajectories(x_noisy, num_trajectories=3, resample=False)
        ensemble_output = model.ensemble_outputs(trajectories, method="mean")

        # Compute loss on ensemble output
        recon_loss = torch.mean((ensemble_output - x_target) ** 2)
        kl_loss = sum(t["kl_loss"] for t in trajectories) / len(trajectories)

        total_loss = recon_loss + 0.01 * kl_loss
        losses.append(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Verify loss decreases
    initial_loss = losses[0]
    final_loss = losses[-1]
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    # Verify ensemble output satisfies constraints
    with torch.no_grad():
        trajectories = model.sample_trajectories(x_noisy, num_trajectories=3, resample=False)
        ensemble_output = model.ensemble_outputs(trajectories, method="mean")
        in_range = (ensemble_output >= -0.1) & (ensemble_output <= 2.1)
        in_range_ratio = in_range.float().mean().item()
        assert in_range_ratio > 0.65, f"Expected >65% in range, got {in_range_ratio*100:.1f}%"

    print(
        f"✓ Pass 3 constraint satisfaction test passed "
        f"(loss: {initial_loss:.4f} -> {final_loss:.4f}, "
        f"{in_range_ratio*100:.1f}% in valid range)"
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

    print("\nTesting GRAM Pass 3 (Scaling and Trajectory Management)...")
    test_gram_pass3_parallel_sampling()
    test_gram_pass3_resampling()
    test_gram_pass3_variable_depth()
    test_gram_pass3_ensemble()
    test_gram_pass3_constraint_satisfaction()

    print("\n✅ All tests passed!")
