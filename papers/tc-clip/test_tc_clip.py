"""Test suite for TC-CLIP Pass 1."""

import torch
import numpy as np
from tc_clip import TCClipPass1, get_video_transforms


def create_synthetic_video(batch_size=2, num_frames=8, height=224, width=224, channels=3):
    """Create a batch of random synthetic videos for testing."""
    video = torch.randn(batch_size, num_frames, channels, height, width)
    video = torch.clamp(video, 0, 1)
    return video


def test_frame_encoder():
    """Test FrameEncoder produces correct output shapes."""
    from tc_clip import FrameEncoder

    batch_size, num_frames, feature_dim = 2, 8, 512
    video = create_synthetic_video(batch_size=batch_size, num_frames=num_frames)

    encoder = FrameEncoder(feature_dim=feature_dim, freeze_backbone=True)
    frame_features = encoder(video)

    assert frame_features.shape == (batch_size, num_frames, feature_dim), \
        f"Expected shape ({batch_size}, {num_frames}, {feature_dim}), got {frame_features.shape}"
    print("✓ FrameEncoder test passed")


def test_temporal_aggregator():
    """Test TemporalAggregator reduces time dimension correctly."""
    from tc_clip import TemporalAggregator

    batch_size, num_frames, feature_dim = 2, 8, 512
    frame_features = torch.randn(batch_size, num_frames, feature_dim)

    aggregator = TemporalAggregator(aggregation_type="mean")
    video_features = aggregator(frame_features)

    assert video_features.shape == (batch_size, feature_dim), \
        f"Expected shape ({batch_size}, {feature_dim}), got {video_features.shape}"

    expected = frame_features.mean(dim=1)
    assert torch.allclose(video_features, expected), \
        "Mean aggregation did not match expected values"
    print("✓ TemporalAggregator test passed")


def test_full_pipeline():
    """Test the full TC-CLIP Pass 1 pipeline end-to-end."""
    batch_size, num_frames, feature_dim = 2, 8, 512
    video = create_synthetic_video(batch_size=batch_size, num_frames=num_frames)

    model = TCClipPass1(feature_dim=feature_dim, freeze_backbone=True, aggregation="mean")
    model.eval()

    with torch.no_grad():
        output = model(video)

    assert "video_features" in output, "Missing 'video_features' in output"
    assert "frame_features" in output, "Missing 'frame_features' in output"

    video_features = output["video_features"]
    frame_features = output["frame_features"]

    assert video_features.shape == (batch_size, feature_dim), \
        f"Expected video_features shape ({batch_size}, {feature_dim}), got {video_features.shape}"
    assert frame_features.shape == (batch_size, num_frames, feature_dim), \
        f"Expected frame_features shape ({batch_size}, {num_frames}, {feature_dim}), " \
        f"got {frame_features.shape}"

    print("✓ Full pipeline test passed")


def test_gradient_flow():
    """Test gradient computation when backbone is trainable."""
    batch_size, num_frames = 2, 8
    video = create_synthetic_video(batch_size=batch_size, num_frames=num_frames)

    model = TCClipPass1(feature_dim=512, freeze_backbone=False, aggregation="mean")
    model.train()

    output = model(video)
    video_features = output["video_features"]

    loss = video_features.sum()
    loss.backward()

    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients, "Some parameters should have gradients"

    print("✓ Gradient flow test passed")


def test_inference_modes():
    """Test that model works in both train and eval modes."""
    batch_size, num_frames = 2, 8
    video = create_synthetic_video(batch_size=batch_size, num_frames=num_frames)

    model = TCClipPass1(feature_dim=512)

    model.eval()
    with torch.no_grad():
        output_eval = model(video)

    model.train()
    output_train = model(video)

    assert output_eval["video_features"].shape == output_train["video_features"].shape, \
        "Output shapes should match between eval and train modes"

    print("✓ Inference modes test passed")


if __name__ == "__main__":
    print("Running TC-CLIP Pass 1 tests...\n")

    test_frame_encoder()
    test_temporal_aggregator()
    test_full_pipeline()
    test_gradient_flow()
    test_inference_modes()

    print("\n✅ All tests passed!")
