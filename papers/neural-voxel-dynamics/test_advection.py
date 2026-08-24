import torch
import torch.nn.functional as F
from voxel_advection import VoxelGrid, VelocityPredictor, FeatureAdvection


def create_gaussian_blob(grid_size, center, std=1.0, feature_dim=1):
    """Create a 3D Gaussian blob in a voxel grid."""
    device = torch.device('cpu')
    features = torch.zeros(1, feature_dim, grid_size, grid_size, grid_size, device=device)

    # Create grid of coordinates
    z, y, x = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32, device=device),
        torch.arange(grid_size, dtype=torch.float32, device=device),
        torch.arange(grid_size, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # Compute Gaussian
    dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    gaussian = torch.exp(-dist_sq / (2 * std ** 2))

    # Set features
    for c in range(feature_dim):
        features[0, c, :, :, :] = gaussian

    return features


def test_voxel_grid_sampling():
    """Test that voxel grid sampling works correctly."""
    print("Testing voxel grid sampling...")
    grid_size = 16
    feature_dim = 1
    batch_size = 2

    voxel_grid = VoxelGrid(grid_size, feature_dim)

    # Sample at the center of a few voxels
    coords = torch.tensor([
        [[grid_size / 2 - 0.5, grid_size / 2 - 0.5, grid_size / 2 - 0.5]],
        [[grid_size / 2, grid_size / 2, grid_size / 2]]
    ], dtype=torch.float32)  # (2, 1, 3)

    features = voxel_grid(coords)
    assert features.shape == (2, 1, feature_dim), f"Expected shape (2, 1, {feature_dim}), got {features.shape}"
    print("✓ Voxel grid sampling shape is correct")


def test_velocity_predictor():
    """Test that velocity predictor produces reasonable output."""
    print("Testing velocity predictor...")
    grid_size = 16
    batch_size = 2
    num_points = 10

    velocity_predictor = VelocityPredictor(grid_size, input_dim=0)

    coords = torch.rand(batch_size, num_points, 3) * (grid_size - 1)
    velocity = velocity_predictor(coords)

    assert velocity.shape == (batch_size, num_points, 3), f"Expected shape {(batch_size, num_points, 3)}, got {velocity.shape}"
    print("✓ Velocity predictor output shape is correct")
    print(f"  Velocity magnitude range: [{velocity.norm(dim=-1).min():.3f}, {velocity.norm(dim=-1).max():.3f}]")


def test_feature_advection():
    """Test that feature advection moves features correctly."""
    print("Testing feature advection...")
    grid_size = 32
    feature_dim = 1

    advection = FeatureAdvection(grid_size, feature_dim, input_dim=0)

    # Initialize voxel grid with a Gaussian blob at position (8, 8, 8)
    blob_center = [8.0, 8.0, 8.0]
    advection.voxel_grid.features.data = create_gaussian_blob(grid_size, blob_center, std=1.5, feature_dim=feature_dim)

    # Create a batch with one point at the blob center
    batch_size = 1
    coords = torch.tensor([[[8.0, 8.0, 8.0]]], dtype=torch.float32)

    # Sample initial feature at blob center
    initial_feature = advection.voxel_grid(coords)
    initial_value = initial_feature[0, 0, 0].item()

    # Perform one advection step
    dt = 1.0
    advected_feature, new_coords = advection(coords, dt=dt)

    print(f"  Initial coords: {coords[0, 0].tolist()}")
    print(f"  New coords: {new_coords[0, 0].tolist()}")
    print(f"  Initial feature value: {initial_value:.4f}")
    print(f"  Advected feature value: {advected_feature[0, 0, 0].item():.4f}")

    # The coordinates should have changed (velocity was non-zero)
    coord_change = (new_coords - coords).norm().item()
    assert coord_change > 0.01, "Coordinates did not change during advection"
    print("✓ Advection moved coordinates")

    # The feature should still be in the valid range [0, 1] (Gaussian is normalized)
    assert 0 <= advected_feature.min() <= 1 and 0 <= advected_feature.max() <= 1, "Feature values out of expected range"
    print("✓ Advected features are in valid range")


def test_multiple_advection_steps():
    """Test that multiple advection steps work correctly."""
    print("Testing multiple advection steps...")
    grid_size = 32
    feature_dim = 2

    advection = FeatureAdvection(grid_size, feature_dim, input_dim=0)

    # Initialize with a blob
    blob_center = [16.0, 16.0, 16.0]
    advection.voxel_grid.features.data = create_gaussian_blob(grid_size, blob_center, std=2.0, feature_dim=feature_dim)

    # Start with a single point
    coords = torch.tensor([[[16.0, 16.0, 16.0]]], dtype=torch.float32)
    dt = 0.5

    positions_history = [coords[0, 0].clone()]

    # Perform multiple advection steps
    num_steps = 5
    for step in range(num_steps):
        features, coords = advection(coords, dt=dt)
        positions_history.append(coords[0, 0].clone())

    # Check that coordinates stayed within bounds
    final_coords = coords[0, 0]
    assert torch.all(final_coords >= 0) and torch.all(final_coords < grid_size), f"Coordinates went out of bounds: {final_coords}"
    print("✓ Multiple advection steps stayed within bounds")

    # Check that we moved from starting position
    total_displacement = (positions_history[-1] - positions_history[0]).norm().item()
    assert total_displacement > 0.1, "No displacement after multiple steps"
    print(f"✓ Total displacement after {num_steps} steps: {total_displacement:.3f}")


if __name__ == '__main__':
    print("Running Neural Voxel Dynamics - Pass 1 Tests\n")
    test_voxel_grid_sampling()
    print()
    test_velocity_predictor()
    print()
    test_feature_advection()
    print()
    test_multiple_advection_steps()
    print("\n✅ All tests passed!")
