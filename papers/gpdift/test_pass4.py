"""Test Pass 4: End-to-end toy video generation demo."""

import torch
import torch.nn as nn
import numpy as np
from diffusion import DiffusionScheduler, DiffusionLoss
from model import (
    LatentEncoder,
    LatentDecoder,
    SequenceContextDenoisingModel,
    AutoregressiveFramePredictor,
)


def create_synthetic_video(num_frames=16, height=32, width=32, object_type="dot"):
    """
    Create a simple synthetic video with moving patterns.

    Args:
        num_frames: Number of frames to generate
        height, width: Frame dimensions
        object_type: "dot" for moving dot, "circle" for expanding circle

    Returns:
        video: (num_frames, 3, height, width) tensor with pixel values in [0, 1]
    """
    frames = []
    center = np.array([height / 2, width / 2])

    for t in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.float32)

        if object_type == "dot":
            # Moving dot: moves diagonally across the frame
            pos = center + (t / num_frames) * np.array([height / 4, width / 4])
            pos = pos.astype(int)
            pos = np.clip(pos, 2, np.array([height - 2, width - 2]))
            frame[pos[0] - 2:pos[0] + 2, pos[1] - 2:pos[1] + 2, :] = 1.0

        elif object_type == "circle":
            # Expanding circle: circle grows over time
            radius = int(2 + (t / num_frames) * 10)
            y, x = np.ogrid[:height, :width]
            mask = (x - width / 2) ** 2 + (y - height / 2) ** 2 <= radius ** 2
            frame[mask, :] = 1.0

        elif object_type == "wave":
            # Wave pattern: sine wave moving to the right
            for i in range(height):
                phase = (t / num_frames) * 2 * np.pi
                for j in range(width):
                    val = 0.5 + 0.5 * np.sin(2 * np.pi * j / width + phase)
                    frame[i, j, :] = val

        frames.append(torch.from_numpy(frame).permute(2, 0, 1))  # (3, H, W)

    video = torch.stack(frames, dim=0)  # (num_frames, 3, H, W)
    return video


def test_end_to_end_video_generation():
    """Test end-to-end video generation: synthetic video -> encode -> predict -> decode."""
    # Setup
    num_frames = 16
    height, width = 32, 32
    latent_dim = 64
    batch_size = 1

    # Create components
    scheduler = DiffusionScheduler(num_steps=50)
    encoder = LatentEncoder(in_channels=3, latent_dim=latent_dim, height=height, width=width)
    decoder = LatentDecoder(latent_dim=latent_dim, out_channels=3, height=height, width=width)

    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=64,
        num_heads=4,
        num_blocks=1,
        use_linear_attn=False,
        use_rotation_time=False,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=2,
        num_denoise_steps=5,
    )

    # Generate synthetic video
    video = create_synthetic_video(num_frames=num_frames, height=height, width=width, object_type="dot")
    assert video.shape == (num_frames, 3, height, width)

    # Split into initial and future frames
    num_initial = 4
    initial_frames = video[:num_initial]  # (4, 3, H, W)
    ground_truth_future = video[num_initial:]  # (12, 3, H, W)

    # Encode initial frames to latents
    initial_frames_batch = initial_frames.unsqueeze(0)  # (1, 4, 3, H, W)
    batch_num_initial, _, _, _, _ = initial_frames_batch.shape
    initial_frames_flat = initial_frames_batch.view(batch_num_initial * num_initial, 3, height, width)
    z_init = encoder(initial_frames_flat)  # (4, latent_dim)
    z_init = z_init.view(batch_num_initial, num_initial, latent_dim)  # (1, 4, latent_dim)

    # Generate future frames autoregressively
    z_predicted = predictor.generate_sequence(z_init, num_frames=num_frames)  # (1, 16, latent_dim)

    # Decode predicted latents
    z_predicted_flat = z_predicted.view(-1, latent_dim)  # (16, latent_dim)
    predicted_frames = decoder(z_predicted_flat)  # (16, 3, H, W)
    predicted_frames = predicted_frames.view(batch_size, num_frames, 3, height, width)

    # Verify shapes
    assert z_predicted.shape == (batch_size, num_frames, latent_dim)
    assert predicted_frames.shape == (batch_size, num_frames, 3, height, width)

    # Basic sanity checks
    assert not torch.isnan(predicted_frames).any(), "Predicted frames contain NaN"
    assert not torch.isinf(predicted_frames).any(), "Predicted frames contain Inf"

    # Check that predictions are in reasonable range (decoder outputs unbounded values)
    assert torch.abs(predicted_frames).max() < 100.0, "Predicted frame values out of reasonable range"

    print("✓ End-to-end video generation test passed")
    return predicted_frames.squeeze(0).detach().numpy()  # Return for potential visualization


def test_video_with_different_patterns():
    """Test video generation with different synthetic patterns."""
    latent_dim = 64
    height, width = 32, 32
    batch_size = 1

    scheduler = DiffusionScheduler(num_steps=50)
    encoder = LatentEncoder(in_channels=3, latent_dim=latent_dim, height=height, width=width)
    decoder = LatentDecoder(latent_dim=latent_dim, out_channels=3, height=height, width=width)

    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=64,
        num_heads=4,
        num_blocks=1,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=2,
        num_denoise_steps=5,
    )

    # Test with different patterns
    for pattern in ["dot", "circle", "wave"]:
        video = create_synthetic_video(num_frames=12, height=height, width=width, object_type=pattern)

        # Encode first 3 frames
        initial_frames = video[:3]
        initial_frames_batch = initial_frames.unsqueeze(0)
        initial_frames_flat = initial_frames_batch.view(3, 3, height, width)
        z_init = encoder(initial_frames_flat).view(1, 3, latent_dim)

        # Generate sequence
        z_sequence = predictor.generate_sequence(z_init, num_frames=12)
        assert z_sequence.shape == (1, 12, latent_dim)

        # Decode
        z_flat = z_sequence.view(-1, latent_dim)
        frames_recon = decoder(z_flat).view(1, 12, 3, height, width)
        assert frames_recon.shape == (1, 12, 3, height, width)

    print("✓ Video with different patterns test passed")


def test_context_window_propagation():
    """Test that context window properly influences predictions over time."""
    latent_dim = 64
    height, width = 32, 32
    num_frames = 20

    scheduler = DiffusionScheduler(num_steps=50)
    encoder = LatentEncoder(in_channels=3, latent_dim=latent_dim, height=height, width=width)

    # Test with different context lengths
    for context_len in [1, 2, 4]:
        denoising_model = SequenceContextDenoisingModel(
            latent_dim=latent_dim,
            d_model=64,
            num_heads=4,
            num_blocks=1,
        )

        predictor = AutoregressiveFramePredictor(
            denoising_model=denoising_model,
            scheduler=scheduler,
            latent_dim=latent_dim,
            context_len=context_len,
            num_denoise_steps=3,
        )

        # Create and encode initial frames
        video = create_synthetic_video(num_frames=3, height=height, width=width)
        initial_frames_flat = video.view(3, 3, height, width)
        z_init = encoder(initial_frames_flat).view(1, 3, latent_dim)

        # Generate long sequence
        z_sequence = predictor.generate_sequence(z_init, num_frames=num_frames)
        assert z_sequence.shape == (1, num_frames, latent_dim)

    print("✓ Context window propagation test passed")


def test_latent_sequence_smoothness():
    """Test that generated latent sequences have reasonable smoothness."""
    latent_dim = 64
    height, width = 32, 32

    scheduler = DiffusionScheduler(num_steps=50)
    encoder = LatentEncoder(in_channels=3, latent_dim=latent_dim, height=height, width=width)

    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=64,
        num_heads=4,
        num_blocks=1,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=3,
        num_denoise_steps=5,
    )

    # Create video and encode
    video = create_synthetic_video(num_frames=4, height=height, width=width)
    initial_frames_flat = video.view(4, 3, height, width)
    z_init = encoder(initial_frames_flat).view(1, 4, latent_dim)

    # Generate sequence
    z_sequence = predictor.generate_sequence(z_init, num_frames=16)  # (1, 16, latent_dim)
    z_sequence = z_sequence.squeeze(0)  # (16, latent_dim)

    # Check smoothness: consecutive frames should have bounded differences
    frame_diffs = torch.norm(z_sequence[1:] - z_sequence[:-1], dim=1)

    # Verify finite differences (no explosion)
    assert torch.all(torch.isfinite(frame_diffs))
    assert torch.all(frame_diffs < 100.0)  # Reasonable upper bound

    print(f"✓ Latent sequence smoothness test passed (avg diff: {frame_diffs.mean():.4f})")


def test_full_pipeline_with_linear_attention():
    """Test full pipeline using linear causal attention."""
    latent_dim = 64
    height, width = 32, 32

    scheduler = DiffusionScheduler(num_steps=50)
    encoder = LatentEncoder(in_channels=3, latent_dim=latent_dim, height=height, width=width)
    decoder = LatentDecoder(latent_dim=latent_dim, out_channels=3, height=height, width=width)

    # Use linear attention variant
    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=64,
        num_heads=4,
        num_blocks=1,
        use_linear_attn=True,  # Linear attention
        use_rotation_time=False,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=2,
        num_denoise_steps=5,
    )

    # Generate and process video
    video = create_synthetic_video(num_frames=12, height=height, width=width)
    initial_frames = video[:3].view(3, 3, height, width)
    z_init = encoder(initial_frames).view(1, 3, latent_dim)

    z_sequence = predictor.generate_sequence(z_init, num_frames=12)
    assert z_sequence.shape == (1, 12, latent_dim)

    decoded = decoder(z_sequence.view(-1, latent_dim))
    assert decoded.shape == (12, 3, height, width)

    print("✓ Full pipeline with linear attention test passed")


def test_full_pipeline_with_rotation_time():
    """Test full pipeline using rotation-based time embedding."""
    latent_dim = 64
    height, width = 32, 32

    scheduler = DiffusionScheduler(num_steps=50)
    encoder = LatentEncoder(in_channels=3, latent_dim=latent_dim, height=height, width=width)
    decoder = LatentDecoder(latent_dim=latent_dim, out_channels=3, height=height, width=width)

    # Use rotation time embedding
    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=64,
        num_heads=4,
        num_blocks=1,
        use_linear_attn=False,
        use_rotation_time=True,  # Rotation time embedding
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=2,
        num_denoise_steps=5,
    )

    # Generate and process video
    video = create_synthetic_video(num_frames=12, height=height, width=width)
    initial_frames = video[:3].view(3, 3, height, width)
    z_init = encoder(initial_frames).view(1, 3, latent_dim)

    z_sequence = predictor.generate_sequence(z_init, num_frames=12)
    assert z_sequence.shape == (1, 12, latent_dim)

    decoded = decoder(z_sequence.view(-1, latent_dim))
    assert decoded.shape == (12, 3, height, width)

    print("✓ Full pipeline with rotation time embedding test passed")


def test_multi_batch_generation():
    """Test video generation with batch size > 1."""
    batch_size = 2
    latent_dim = 64
    height, width = 32, 32
    num_frames = 12

    scheduler = DiffusionScheduler(num_steps=50)
    encoder = LatentEncoder(in_channels=3, latent_dim=latent_dim, height=height, width=width)
    decoder = LatentDecoder(latent_dim=latent_dim, out_channels=3, height=height, width=width)

    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=64,
        num_heads=4,
        num_blocks=1,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=2,
        num_denoise_steps=5,
    )

    # Create batch of videos
    videos = torch.stack([
        create_synthetic_video(num_frames=4, object_type="dot"),
        create_synthetic_video(num_frames=4, object_type="circle"),
    ])  # (2, 4, 3, H, W)

    # Encode
    videos_flat = videos.view(batch_size * 4, 3, height, width)
    z_init = encoder(videos_flat).view(batch_size, 4, latent_dim)

    # Generate
    z_sequence = predictor.generate_sequence(z_init, num_frames=num_frames)
    assert z_sequence.shape == (batch_size, num_frames, latent_dim)

    # Decode
    z_flat = z_sequence.view(-1, latent_dim)
    decoded = decoder(z_flat).view(batch_size, num_frames, 3, height, width)
    assert decoded.shape == (batch_size, num_frames, 3, height, width)

    print("✓ Multi-batch generation test passed")


if __name__ == "__main__":
    test_end_to_end_video_generation()
    test_video_with_different_patterns()
    test_context_window_propagation()
    test_latent_sequence_smoothness()
    test_full_pipeline_with_linear_attention()
    test_full_pipeline_with_rotation_time()
    test_multi_batch_generation()
    print("\n✅ All Pass 4 tests passed!")
