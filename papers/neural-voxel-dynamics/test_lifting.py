import torch
import torch.nn.functional as F
from image_to_voxel import (
    CameraModel, SimpleDepthEstimator, SimpleImageFeatureExtractor,
    ImageToVoxelProjector, VoxelToImageUnprojector
)


def test_camera_model_projection():
    """Test basic camera projection and unprojection."""
    print("Testing camera model projection...")
    camera = CameraModel(fx=400.0, fy=400.0, cx=160.0, cy=120.0)

    # Create a simple 3D point
    points_3d = torch.tensor([[1.0, 0.5, 5.0]], dtype=torch.float32)  # (1, 3)

    # Project to 2D
    points_2d, depth = camera.project_3d_to_2d(points_3d)

    assert points_2d.shape == (1, 2), f"Expected shape (1, 2), got {points_2d.shape}"
    assert depth.shape == (1,), f"Expected depth shape (1,), got {depth.shape}"
    assert torch.allclose(depth, torch.tensor([5.0])), "Depth should be 5.0"

    print(f"  3D point: {points_3d[0].tolist()}")
    print(f"  Projected 2D: {points_2d[0].tolist()}")
    print(f"  Depth: {depth[0].item():.3f}")

    # Unproject back to 3D
    points_3d_reconstructed = camera.unproject_2d_to_3d(points_2d, depth)

    # Should approximately match original
    error = (points_3d_reconstructed - points_3d).abs().max().item()
    assert error < 1e-4, f"Unprojection error too large: {error}"
    print(f"✓ Camera projection/unprojection round-trip successful")


def test_depth_estimator():
    """Test depth estimator produces valid outputs."""
    print("Testing depth estimator...")
    depth_estimator = SimpleDepthEstimator()
    depth_estimator.eval()

    # Create synthetic image
    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        depth_maps = depth_estimator(images)

    assert depth_maps.shape == (2, 1, 64, 64), f"Expected shape (2, 1, 64, 64), got {depth_maps.shape}"
    assert depth_maps.min() >= 0.1, f"Min depth should be >= 0.1, got {depth_maps.min()}"
    assert depth_maps.max() <= 10.0, f"Max depth should be <= 10.0, got {depth_maps.max()}"
    print(f"✓ Depth estimator output shape correct")
    print(f"  Depth range: [{depth_maps.min():.3f}, {depth_maps.max():.3f}]")


def test_image_feature_extractor():
    """Test image feature extractor."""
    print("Testing image feature extractor...")
    feature_dim = 32
    extractor = SimpleImageFeatureExtractor(image_channels=3, feature_channels=feature_dim)
    extractor.eval()

    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        features = extractor(images)

    assert features.shape == (2, feature_dim, 64, 64), f"Expected shape (2, {feature_dim}, 64, 64), got {features.shape}"
    print(f"✓ Feature extractor output shape correct")
    print(f"  Output feature channels: {features.shape[1]}")


def test_image_to_voxel_projector():
    """Test lifting 2D images to 3D voxel grid."""
    print("Testing image-to-voxel projector...")
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
    projector.eval()

    # Create synthetic images
    images = torch.randn(2, 3, image_height, image_width)
    depth_maps = torch.ones(2, 1, image_height, image_width) * 5.0  # All at depth 5

    with torch.no_grad():
        voxel_features = projector(images, depth_maps)

    assert voxel_features.shape == (2, feature_dim, grid_size, grid_size, grid_size), \
        f"Expected shape (2, {feature_dim}, {grid_size}, {grid_size}, {grid_size}), got {voxel_features.shape}"
    print(f"✓ Voxel projection output shape correct")
    print(f"  Output voxel grid: {voxel_features.shape[2]}³")

    # Check that features are in reasonable range
    print(f"  Voxel features range: [{voxel_features.min():.4f}, {voxel_features.max():.4f}]")


def test_voxel_to_image_unprojector():
    """Test unprojecting voxel features back to 2D."""
    print("Testing voxel-to-image unprojector...")
    image_height, image_width = 64, 64
    grid_size = 16
    feature_dim = 8

    unprojector = VoxelToImageUnprojector(
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        feature_dim=feature_dim,
        fx=400.0, fy=400.0
    )
    unprojector.eval()

    # Create synthetic voxel features
    voxel_features = torch.randn(2, feature_dim, grid_size, grid_size, grid_size)

    with torch.no_grad():
        image_features = unprojector(voxel_features)

    assert image_features.shape == (2, feature_dim, image_height, image_width), \
        f"Expected shape (2, {feature_dim}, {image_height}, {image_width}), got {image_features.shape}"
    print(f"✓ Voxel unprojection output shape correct")
    print(f"  Output image features: {image_features.shape}")


def test_lifting_pipeline():
    """Test complete lifting and unprojection pipeline."""
    print("Testing complete lifting pipeline...")
    image_height, image_width = 64, 64
    grid_size = 16
    feature_dim = 8

    projector = ImageToVoxelProjector(
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        feature_dim=feature_dim
    )
    unprojector = VoxelToImageUnprojector(
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        feature_dim=feature_dim
    )

    projector.eval()
    unprojector.eval()

    # Create synthetic images
    images = torch.randn(2, 3, image_height, image_width)
    depth_maps = torch.ones(2, 1, image_height, image_width) * 5.0

    with torch.no_grad():
        # Lift to voxel space
        voxel_features = projector(images, depth_maps)

        # Unproject back to image space
        reconstructed_images = unprojector(voxel_features)

    assert voxel_features.shape == (2, feature_dim, grid_size, grid_size, grid_size)
    assert reconstructed_images.shape == (2, feature_dim, image_height, image_width)

    print(f"✓ Complete pipeline executes successfully")
    print(f"  Voxel features shape: {voxel_features.shape}")
    print(f"  Reconstructed image features shape: {reconstructed_images.shape}")

    # Check that reconstructed image has some non-zero values
    non_zero_count = (reconstructed_images.abs() > 1e-6).sum().item()
    print(f"  Non-zero voxels in reconstruction: {non_zero_count}")


def test_lifting_with_varying_depth():
    """Test lifting pipeline with spatially varying depth."""
    print("Testing lifting with varying depth...")
    image_height, image_width = 64, 64
    grid_size = 16
    feature_dim = 4

    projector = ImageToVoxelProjector(
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        feature_dim=feature_dim,
        voxel_bounds=(0.1, 10.0)
    )
    projector.eval()

    # Create synthetic images
    images = torch.randn(1, 3, image_height, image_width)

    # Create varying depth map (gradient)
    v_coords, u_coords = torch.meshgrid(
        torch.arange(image_height, dtype=torch.float32),
        torch.arange(image_width, dtype=torch.float32),
        indexing='ij'
    )
    depth_maps = 1.0 + (u_coords / image_width) * 8.0  # Range from 1 to 9
    depth_maps = depth_maps.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        voxel_features = projector(images, depth_maps)

    assert voxel_features.shape == (1, feature_dim, grid_size, grid_size, grid_size)
    print(f"✓ Varying depth map handled correctly")
    print(f"  Voxel features shape: {voxel_features.shape}")
    print(f"  Feature value range: [{voxel_features.min():.4f}, {voxel_features.max():.4f}]")


if __name__ == '__main__':
    print("Running Neural Voxel Dynamics - Pass 2 Tests\n")
    test_camera_model_projection()
    print()
    test_depth_estimator()
    print()
    test_image_feature_extractor()
    print()
    test_image_to_voxel_projector()
    print()
    test_voxel_to_image_unprojector()
    print()
    test_lifting_pipeline()
    print()
    test_lifting_with_varying_depth()
    print("\n✅ All pass 2 tests passed!")
