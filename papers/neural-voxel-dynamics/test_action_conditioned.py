import torch
from action_conditioned_flow import (
    ActionConditionedFlowPredictor,
    ActionConditionedAdvection,
    SyntheticVideoSequenceGenerator,
)


def test_action_conditioned_flow_predictor():
    """Test that the action-conditioned flow predictor works."""
    print("Testing action-conditioned flow predictor...")

    grid_size = 16
    feature_dim = 2
    action_dim = 3
    batch_size = 2
    num_points = 5

    predictor = ActionConditionedFlowPredictor(grid_size, feature_dim, action_dim)

    # Create random inputs
    coords = torch.rand(batch_size, num_points, 3) * (grid_size - 1)
    features = torch.randn(batch_size, num_points, feature_dim)
    actions = torch.randn(batch_size, action_dim)

    # Forward pass
    velocity = predictor(coords, features, actions)

    assert velocity.shape == (batch_size, num_points, 3), f"Expected shape {(batch_size, num_points, 3)}, got {velocity.shape}"
    print("✓ Action-conditioned flow predictor output shape is correct")
    print(f"  Velocity magnitude range: [{velocity.norm(dim=-1).min():.3f}, {velocity.norm(dim=-1).max():.3f}]")


def test_action_conditioned_advection():
    """Test that action-conditioned advection works."""
    print("Testing action-conditioned advection...")

    grid_size = 16
    feature_dim = 1
    action_dim = 3
    batch_size = 2

    advection = ActionConditionedAdvection(grid_size, feature_dim, action_dim)

    # Create a simple Gaussian blob in the voxel grid
    device = torch.device('cpu')
    features = torch.zeros(1, feature_dim, grid_size, grid_size, grid_size, device=device)
    z, y, x = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32, device=device),
        torch.arange(grid_size, dtype=torch.float32, device=device),
        torch.arange(grid_size, dtype=torch.float32, device=device),
        indexing='ij'
    )
    center = [grid_size / 2, grid_size / 2, grid_size / 2]
    dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    gaussian = torch.exp(-dist_sq / (2 * 2.0 ** 2))
    features[0, 0, :, :, :] = gaussian

    advection.voxel_grid.features.data = features

    # Create test inputs
    coords = torch.tensor([
        [[grid_size / 2, grid_size / 2, grid_size / 2]],
        [[grid_size / 2, grid_size / 2, grid_size / 2]]
    ], dtype=torch.float32)  # (batch_size, 1, 3)

    actions = torch.tensor([
        [1.0, 0.0, 0.0],  # Move right
        [-1.0, 0.0, 0.0]  # Move left
    ], dtype=torch.float32)  # (batch_size, 3)

    # Forward pass
    advected_features, new_coords, velocity = advection(coords, actions, dt=0.5)

    assert advected_features.shape == (batch_size, 1, feature_dim), f"Expected shape {(batch_size, 1, feature_dim)}, got {advected_features.shape}"
    assert new_coords.shape == (batch_size, 1, 3), f"Expected shape {(batch_size, 1, 3)}, got {new_coords.shape}"
    assert velocity.shape == (batch_size, 1, 3), f"Expected shape {(batch_size, 1, 3)}, got {velocity.shape}"

    print("✓ Action-conditioned advection output shapes are correct")
    print(f"  Predicted velocity for action [1,0,0]: {velocity[0, 0].tolist()}")
    print(f"  Predicted velocity for action [-1,0,0]: {velocity[1, 0].tolist()}")

    # Check that different actions produce different velocities
    vel_diff = (velocity[0] - velocity[1]).norm().item()
    assert vel_diff > 0.01, "Different actions should produce different velocities"
    print("✓ Different actions produce different velocities")


def test_synthetic_sequence_generator():
    """Test that synthetic sequence generation works."""
    print("Testing synthetic sequence generator...")

    grid_size = 16
    feature_dim = 1
    sequence_length = 5

    generator = SyntheticVideoSequenceGenerator(
        grid_size=grid_size,
        feature_dim=feature_dim,
        sequence_length=sequence_length
    )

    # Generate single sequence
    frames, actions = generator.generate_sequence()

    assert len(frames) == sequence_length, f"Expected {sequence_length} frames, got {len(frames)}"
    assert actions.shape == (sequence_length - 1, 3), f"Expected shape {(sequence_length - 1, 3)}, got {actions.shape}"

    for i, frame in enumerate(frames):
        assert frame.shape == (1, feature_dim, grid_size, grid_size, grid_size), f"Frame {i} has wrong shape: {frame.shape}"

    print(f"✓ Generated sequence with {sequence_length} frames and {actions.shape[0]} actions")
    print(f"  Action magnitudes: {[f'{a.norm():.3f}' for a in actions]}")

    # Generate batch
    batch_size = 3
    frames_batch, actions_batch = generator.generate_batch(batch_size)

    assert frames_batch.shape == (batch_size, sequence_length, feature_dim, grid_size, grid_size, grid_size), f"Wrong batch frames shape: {frames_batch.shape}"
    assert actions_batch.shape == (batch_size, sequence_length - 1, 3), f"Wrong batch actions shape: {actions_batch.shape}"

    print(f"✓ Generated batch of {batch_size} sequences")
    print(f"  Batch frames shape: {frames_batch.shape}")
    print(f"  Batch actions shape: {actions_batch.shape}")


def test_training_step():
    """Test that a single training step works."""
    print("Testing training step...")

    import torch.optim as optim
    import torch.nn as nn

    grid_size = 16
    feature_dim = 1
    action_dim = 3
    batch_size = 2

    model = ActionConditionedAdvection(grid_size, feature_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    mse_loss = nn.MSELoss()

    generator = SyntheticVideoSequenceGenerator(
        grid_size=grid_size,
        feature_dim=feature_dim,
        sequence_length=3
    )

    # Generate a small batch
    frames, actions = generator.generate_batch(batch_size)

    # Run one training step
    current_frame = frames[:, 0]  # (batch_size, feature_dim, grid_size, grid_size, grid_size)
    next_frame = frames[:, 1]
    action = actions[:, 0]  # (batch_size, 3)

    # Set current frame in voxel grid
    model.voxel_grid.features.data = current_frame

    # Sample points at blob center
    center_coord = torch.tensor([grid_size / 2.0, grid_size / 2.0, grid_size / 2.0])
    coords = center_coord.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)  # (batch_size, 1, 3)

    # Predict next features
    predicted_features, new_coords, velocity = model(coords, action, dt=1.0)

    # Get target features
    model.voxel_grid.features.data = next_frame
    target_features = model.voxel_grid(new_coords)

    # Compute loss and step
    loss = mse_loss(predicted_features, target_features)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✓ Training step completed successfully")
    print(f"  Loss: {loss.item():.6f}")


if __name__ == '__main__':
    print("Running Neural Voxel Dynamics - Pass 3 Tests\n")
    test_action_conditioned_flow_predictor()
    print()
    test_action_conditioned_advection()
    print()
    test_synthetic_sequence_generator()
    print()
    test_training_step()
    print("\n✅ All Pass 3 tests passed!")
