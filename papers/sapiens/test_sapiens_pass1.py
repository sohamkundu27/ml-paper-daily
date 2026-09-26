import torch
import numpy as np
from pose_detector import (
    KeypointDetector, create_synthetic_image_with_points, points_to_heatmap
)


def test_backbone_forward_pass():
    """Test that the CNN backbone processes images correctly"""
    detector = KeypointDetector(num_keypoints=17, image_size=256)
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256)

    output = detector(images)

    assert output.shape == (batch_size, 17, 64, 64), f"Expected shape (2, 17, 64, 64), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print("✓ Backbone forward pass test passed")


def test_keypoint_extraction():
    """Test keypoint coordinate extraction from heatmaps"""
    detector = KeypointDetector(num_keypoints=17, image_size=256)
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256)

    heatmaps = detector(images)
    keypoints = detector.get_keypoints_from_heatmaps(heatmaps)

    assert keypoints.shape == (batch_size, 17, 2), f"Expected shape (2, 17, 2), got {keypoints.shape}"
    assert torch.all(keypoints >= 0) and torch.all(keypoints <= 256), "Keypoints out of image bounds"
    print("✓ Keypoint extraction test passed")


def test_synthetic_image_creation():
    """Test synthetic image and keypoint generation"""
    image, keypoints = create_synthetic_image_with_points(image_size=256, num_points=17)

    assert image.shape == (256, 256, 3), f"Expected image shape (256, 256, 3), got {image.shape}"
    assert keypoints.shape == (17, 2), f"Expected keypoints shape (17, 2), got {keypoints.shape}"
    assert np.all(keypoints >= 0) and np.all(keypoints <= 256), "Keypoints out of bounds"
    print("✓ Synthetic image creation test passed")


def test_heatmap_generation():
    """Test conversion of keypoints to heatmaps"""
    image, keypoints = create_synthetic_image_with_points(image_size=256, num_points=17)
    heatmaps = points_to_heatmap(keypoints, image_size=256, heatmap_size=64)

    assert heatmaps.shape == (17, 64, 64), f"Expected shape (17, 64, 64), got {heatmaps.shape}"
    assert np.all(heatmaps >= 0) and np.all(heatmaps <= 1), "Heatmap values out of [0, 1]"
    for k in range(17):
        assert np.max(heatmaps[k]) > 0.5, f"Keypoint {k} heatmap too low"
    print("✓ Heatmap generation test passed")


def test_forward_backward_pass():
    """Test that gradients flow correctly through the network"""
    detector = KeypointDetector(num_keypoints=17, image_size=256)
    images = torch.randn(2, 3, 256, 256, requires_grad=True)
    heatmaps = detector(images)

    loss = heatmaps.mean()
    loss.backward()

    assert images.grad is not None, "Gradients not flowing to input"
    assert not torch.isnan(images.grad).any(), "Gradient contains NaN"
    print("✓ Forward-backward pass test passed")


def test_end_to_end_pipeline():
    """Test complete pipeline: image -> heatmaps -> keypoints"""
    detector = KeypointDetector(num_keypoints=17, image_size=256)

    image_np, keypoints_np = create_synthetic_image_with_points(image_size=256, num_points=17)
    heatmaps_np = points_to_heatmap(keypoints_np, image_size=256, heatmap_size=64)

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)

    predicted_heatmaps = detector(image_tensor)
    predicted_keypoints = detector.get_keypoints_from_heatmaps(predicted_heatmaps)

    assert predicted_heatmaps.shape == (1, 17, 64, 64)
    assert predicted_keypoints.shape == (1, 17, 2)
    assert torch.all(predicted_keypoints >= 0) and torch.all(predicted_keypoints <= 256)
    print("✓ End-to-end pipeline test passed")


if __name__ == "__main__":
    test_backbone_forward_pass()
    test_keypoint_extraction()
    test_synthetic_image_creation()
    test_heatmap_generation()
    test_forward_backward_pass()
    test_end_to_end_pipeline()
    print("\n✅ All Pass 1 tests passed!")
