import torch
from image_to_voxel import ImageToVoxelProjector, VoxelToImageUnprojector


def demo_image_to_voxel_lifting():
    """Demonstrate lifting 2D images to 3D voxel grid."""
    print("=" * 70)
    print("Neural Voxel Dynamics - Pass 2: Image-to-Voxel Lifting Demo")
    print("=" * 70)

    # Setup
    image_height, image_width = 64, 64
    grid_size = 16
    feature_dim = 8

    projector = ImageToVoxelProjector(
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        feature_dim=feature_dim,
        fx=400.0, fy=400.0
    )
    unprojector = VoxelToImageUnprojector(
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        feature_dim=feature_dim,
        fx=400.0, fy=400.0
    )

    projector.eval()
    unprojector.eval()

    print(f"\nConfiguration:")
    print(f"  Image size: {image_height} x {image_width}")
    print(f"  Voxel grid: {grid_size}³")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Camera intrinsics: fx=400.0, fy=400.0, cx={image_width/2}, cy={image_height/2}")

    # Create synthetic RGB images
    batch_size = 2
    images = torch.randn(batch_size, 3, image_height, image_width)

    # Create corresponding depth maps
    # Simple case: all pixels at same depth
    depth_maps = torch.ones(batch_size, 1, image_height, image_width) * 5.0

    print(f"\nInput images: shape {images.shape}")
    print(f"Input depth maps: shape {depth_maps.shape}")
    print(f"  Depth range: [{depth_maps.min():.3f}, {depth_maps.max():.3f}]")

    # Lift images to voxel space
    with torch.no_grad():
        voxel_features = projector(images, depth_maps)

    print(f"\n✓ Lifted to voxel space")
    print(f"  Voxel features shape: {voxel_features.shape}")
    print(f"  Feature value range: [{voxel_features.min():.4f}, {voxel_features.max():.4f}]")

    # Count non-zero voxels
    non_zero_voxels = (voxel_features.abs() > 1e-6).sum(dim=1).float()
    print(f"  Non-zero voxels: {non_zero_voxels.mean().item():.0f} (avg across batch)")

    # Unproject back to image space
    with torch.no_grad():
        reconstructed_features = unprojector(voxel_features)

    print(f"\n✓ Unprojected back to image space")
    print(f"  Reconstructed image features shape: {reconstructed_features.shape}")
    print(f"  Feature value range: [{reconstructed_features.min():.4f}, {reconstructed_features.max():.4f}]")

    # Print summary
    print("\n" + "=" * 70)
    print("Lifting Pipeline Summary:")
    print(f"  2D image (RGB) → extract features")
    print(f"  + depth map (from estimator or ground truth)")
    print(f"  → lift pixels to 3D using camera model")
    print(f"  → project 3D features into voxel grid (via trilinear interpolation)")
    print(f"  ✓ Now features are in 3D space for advection!")
    print(f"\n  Reverse pipeline (voxel → image) also implemented")
    print(f"  for visualization and potential auxiliary supervision")
    print("=" * 70)


def demo_varying_depth_lifting():
    """Demonstrate lifting with spatially-varying depth."""
    print("\n" + "=" * 70)
    print("Demo 2: Lifting with Varying Depth Map")
    print("=" * 70)

    image_height, image_width = 64, 64
    grid_size = 16
    feature_dim = 4

    projector = ImageToVoxelProjector(
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        feature_dim=feature_dim
    )
    projector.eval()

    # Create synthetic image
    images = torch.randn(1, 3, image_height, image_width)

    # Create a depth map with gradient (closer on the left, farther on the right)
    v_coords, u_coords = torch.meshgrid(
        torch.arange(image_height, dtype=torch.float32),
        torch.arange(image_width, dtype=torch.float32),
        indexing='ij'
    )
    # Linear gradient from depth 2.0 to 8.0
    depth_maps = 2.0 + (u_coords / image_width) * 6.0
    depth_maps = depth_maps.unsqueeze(0).unsqueeze(0)

    print(f"\nInput image shape: {images.shape}")
    print(f"Depth map: gradient from left (depth={depth_maps.min():.1f}) to right (depth={depth_maps.max():.1f})")

    # Lift to voxel space
    with torch.no_grad():
        voxel_features = projector(images, depth_maps)

    print(f"\n✓ Voxel features: {voxel_features.shape}")
    print(f"  Non-zero voxels: {(voxel_features.abs() > 1e-6).sum().item()}")

    # Show distribution across depth layers
    voxel_occupancy = (voxel_features.abs() > 1e-6).float().sum(dim=1)  # Sum over feature channels
    occupancy_per_depth = voxel_occupancy[0].sum(dim=(1, 2))  # Sum over spatial dims

    print(f"\nVoxel occupancy per depth layer:")
    for d in range(min(5, grid_size)):
        occ = occupancy_per_depth[d].item()
        bar = "█" * int(occ / 10)
        print(f"  Depth layer {d:2d}: {occ:6.0f} {bar}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    demo_image_to_voxel_lifting()
    demo_varying_depth_lifting()
    print("\n✅ Pass 2 demos completed successfully!")
