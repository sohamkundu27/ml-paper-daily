import torch
from voxel_advection import VoxelGrid, FeatureAdvection


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


def demo_blob_advection():
    """Demonstrate advecting a Gaussian blob through voxel space."""
    print("=" * 60)
    print("Neural Voxel Dynamics - Pass 1: Feature Advection Demo")
    print("=" * 60)

    # Setup
    grid_size = 32
    feature_dim = 1
    advection = FeatureAdvection(grid_size, feature_dim, input_dim=0)

    # Initialize voxel grid with a Gaussian blob
    blob_center = [16.0, 16.0, 16.0]
    advection.voxel_grid.features.data = create_gaussian_blob(
        grid_size, blob_center, std=2.0, feature_dim=feature_dim
    )

    print(f"\nGrid size: {grid_size}³")
    print(f"Feature dimension: {feature_dim}")
    print(f"Initial blob center: {blob_center}")

    # Create a tracking point at the blob center
    tracking_point = torch.tensor([[[16.0, 16.0, 16.0]]], dtype=torch.float32)
    dt = 0.5
    num_steps = 20

    print(f"\nAdvecting a point through the voxel grid for {num_steps} steps (dt={dt}):")
    print(f"{'Step':>5} | {'Position':>30} | {'Feature Value':>14}")
    print("-" * 55)

    positions = [tracking_point[0, 0].clone().detach().cpu().numpy()]
    feature_values = []

    # Sample initial feature
    initial_feature = advection.voxel_grid(tracking_point)
    feature_values.append(initial_feature[0, 0, 0].item())
    pos_str = [f"{x:.3f}" for x in positions[0]]
    print(f"{0:5d} | [{', '.join(pos_str)}] | {feature_values[0]:14.6f}")

    # Perform advection steps
    coords = tracking_point
    for step in range(1, num_steps + 1):
        features, coords = advection(coords, dt=dt)
        positions.append(coords[0, 0].clone().detach().cpu().numpy())
        feature_values.append(features[0, 0, 0].item())

        if step % 5 == 0 or step == num_steps:
            pos_str = [f"{x:.3f}" for x in positions[-1]]
            print(f"{step:5d} | [{', '.join(pos_str)}] | {feature_values[-1]:14.6f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary:")
    initial_pos = positions[0]
    final_pos = positions[-1]
    total_displacement = sum((final_pos[i] - initial_pos[i]) ** 2 for i in range(3)) ** 0.5
    print(f"  Total displacement: {total_displacement:.3f} voxels")
    print(f"  Initial feature value: {feature_values[0]:.6f}")
    print(f"  Final feature value: {feature_values[-1]:.6f}")
    print(f"  Min feature value encountered: {min(feature_values):.6f}")
    print(f"  Max feature value encountered: {max(feature_values):.6f}")
    print("\nAdvection completed successfully! ✓")
    print("=" * 60)


if __name__ == '__main__':
    demo_blob_advection()
