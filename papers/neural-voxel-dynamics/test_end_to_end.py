import torch
from end_to_end_demo import (
    SyntheticRGBVideoGenerator,
    EndToEndFramePredictor,
    demo_feature_lifting,
    demo_frame_prediction,
)


def test_synthetic_rgb_video_generator():
    """Test that the RGB video generator works correctly."""
    print("Testing synthetic RGB video generator...")

    generator = SyntheticRGBVideoGenerator(height=64, width=64)

    # Generate a sequence
    frames, positions, actions = generator.generate_sequence(num_frames=5)

    assert len(frames) == 5, f"Expected 5 frames, got {len(frames)}"
    assert len(actions) == 4, f"Expected 4 actions, got {len(actions)}"
    assert positions.shape == (5, 2), f"Expected positions shape (5, 2), got {positions.shape}"
    assert actions.shape == (4, 3), f"Expected actions shape (4, 3), got {actions.shape}"

    # Check frame shapes
    for i, frame in enumerate(frames):
        assert frame.shape == (3, 64, 64), f"Frame {i} has shape {frame.shape}, expected (3, 64, 64)"
        assert frame.min() >= 0 and frame.max() <= 1, f"Frame {i} values out of [0, 1] range"

    print("✓ RGB video generator output shapes are correct")
    print(f"  Generated {len(frames)} frames with {len(actions)} actions")


def test_end_to_end_frame_predictor():
    """Test that the end-to-end predictor works."""
    print("Testing end-to-end frame predictor...")

    image_height, image_width = 64, 64
    grid_size = 32
    feature_dim = 16
    action_dim = 3

    model = EndToEndFramePredictor(image_height, image_width, grid_size, feature_dim, action_dim)
    model.eval()

    # Create test data
    batch_size = 2
    image = torch.randn(batch_size, 3, image_height, image_width)
    action = torch.randn(batch_size, action_dim)

    # Forward pass
    with torch.no_grad():
        predicted_image, voxel_current, voxel_next = model(image, action)

    assert predicted_image.shape == (batch_size, feature_dim, image_height, image_width), \
        f"Expected predicted_image shape {(batch_size, feature_dim, image_height, image_width)}, got {predicted_image.shape}"
    assert voxel_current.shape == (batch_size, feature_dim, grid_size, grid_size, grid_size), \
        f"Expected voxel_current shape {(batch_size, feature_dim, grid_size, grid_size, grid_size)}, got {voxel_current.shape}"
    assert voxel_next.shape == (batch_size, feature_dim, grid_size, grid_size, grid_size), \
        f"Expected voxel_next shape {(batch_size, feature_dim, grid_size, grid_size, grid_size)}, got {voxel_next.shape}"

    print("✓ End-to-end predictor output shapes are correct")
    print(f"  Predicted image norm: {predicted_image.norm():.4f}")
    print(f"  Voxel current norm: {voxel_current.norm():.4f}")


def test_end_to_end_with_real_video():
    """Test end-to-end prediction on real generated video."""
    print("Testing end-to-end prediction with synthetic video...")

    device = torch.device('cpu')

    # Generate synthetic video
    video_generator = SyntheticRGBVideoGenerator(height=64, width=64)
    frames, positions, actions = video_generator.generate_sequence(num_frames=3)

    # Initialize model
    model = EndToEndFramePredictor(64, 64, 32, 16, 3)
    model.to(device)
    model.eval()

    print(f"  Generated sequence with {len(frames)} frames")

    # Process frames
    with torch.no_grad():
        for i in range(len(frames) - 1):
            frame = frames[i].unsqueeze(0).to(device)
            action = actions[i].unsqueeze(0).to(device)

            predicted_image, voxel_current, voxel_next = model(frame, action)

            # Check outputs are valid
            assert not torch.isnan(predicted_image).any(), f"NaN in predicted_image at frame {i}"
            assert not torch.isnan(voxel_current).any(), f"NaN in voxel_current at frame {i}"
            assert predicted_image.norm() > 0, f"Zero predicted_image at frame {i}"

    print("✓ End-to-end prediction on video works correctly")


if __name__ == '__main__':
    test_synthetic_rgb_video_generator()
    test_end_to_end_frame_predictor()
    test_end_to_end_with_real_video()

    print("\n" + "=" * 70)
    print("All Pass 4 tests passed!")
    print("=" * 70)
