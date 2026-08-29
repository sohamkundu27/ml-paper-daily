"""Test Pass 3: Autoregressive frame sequence prediction."""

import torch
import torch.nn as nn
from diffusion import DiffusionScheduler, DiffusionLoss
from model import (
    LatentEncoder,
    LatentDecoder,
    DenoisingModel,
    SequenceContextDenoisingModel,
    AutoregressiveFramePredictor,
)


def test_sequence_context_denoising_model_basic():
    """Test sequence context denoising model forward pass."""
    batch_size = 2
    latent_dim = 256
    context_len = 4

    model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=256,
        num_heads=8,
        num_blocks=1,
        use_linear_attn=False,
        use_rotation_time=False,
    )

    # Context: previous frames
    z_context = torch.randn(batch_size, context_len, latent_dim)
    # Current noisy frame
    z_t = torch.randn(batch_size, latent_dim)
    t = torch.randint(0, 1000, (batch_size,))

    # Predict noise
    noise_pred = model(z_context, z_t, t)

    assert noise_pred.shape == z_t.shape
    assert noise_pred.requires_grad

    print("✓ Sequence context denoising model basic test passed")


def test_sequence_context_denoising_with_linear_attention():
    """Test sequence context denoising with linear attention."""
    batch_size = 2
    latent_dim = 256
    context_len = 3

    model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=128,
        num_heads=4,
        num_blocks=1,
        use_linear_attn=True,
        use_rotation_time=False,
    )

    z_context = torch.randn(batch_size, context_len, latent_dim)
    z_t = torch.randn(batch_size, latent_dim)
    t = torch.randint(0, 1000, (batch_size,))

    noise_pred = model(z_context, z_t, t)

    assert noise_pred.shape == z_t.shape
    assert noise_pred.requires_grad

    print("✓ Sequence context denoising with linear attention test passed")


def test_sequence_context_denoising_with_rotation_time():
    """Test sequence context denoising with rotation time embedding."""
    batch_size = 2
    latent_dim = 256
    context_len = 4

    model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=256,
        num_heads=8,
        num_blocks=1,
        use_linear_attn=False,
        use_rotation_time=True,
    )

    z_context = torch.randn(batch_size, context_len, latent_dim)
    z_t = torch.randn(batch_size, latent_dim)
    t = torch.randint(0, 1000, (batch_size,))

    noise_pred = model(z_context, z_t, t)

    assert noise_pred.shape == z_t.shape

    print("✓ Sequence context denoising with rotation time embedding test passed")


def test_autoregressive_predictor_with_standard_model():
    """Test autoregressive predictor using standard denoising model."""
    batch_size = 2
    latent_dim = 256
    context_len = 4

    scheduler = DiffusionScheduler(num_steps=50)
    denoising_model = DenoisingModel(
        latent_dim=latent_dim,
        d_model=128,
        num_heads=4,
        num_blocks=1,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=context_len,
        num_denoise_steps=10,
    )

    # Context frames
    z_context = torch.randn(batch_size, context_len, latent_dim)

    # Predict next frame
    z_next = predictor.predict_next_frame(z_context, t_start=10)

    assert z_next.shape == (batch_size, latent_dim)
    assert z_next.requires_grad == False  # Should be in eval mode

    print("✓ Autoregressive predictor with standard model test passed")


def test_autoregressive_predictor_with_sequence_model():
    """Test autoregressive predictor using sequence context denoising model."""
    batch_size = 2
    latent_dim = 256
    context_len = 4

    scheduler = DiffusionScheduler(num_steps=50)
    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=128,
        num_heads=4,
        num_blocks=1,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=context_len,
        num_denoise_steps=10,
    )

    # Context frames
    z_context = torch.randn(batch_size, context_len, latent_dim)

    # Predict next frame
    z_next = predictor.predict_next_frame(z_context, t_start=10)

    assert z_next.shape == (batch_size, latent_dim)
    assert z_next.requires_grad == False

    print("✓ Autoregressive predictor with sequence model test passed")


def test_generate_sequence_from_single_frame():
    """Test generating a sequence starting from a single frame."""
    batch_size = 2
    latent_dim = 256
    context_len = 4
    num_frames = 8

    scheduler = DiffusionScheduler(num_steps=50)
    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=128,
        num_heads=4,
        num_blocks=1,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=context_len,
        num_denoise_steps=10,
    )

    # Initial frame (single frame)
    z_init = torch.randn(batch_size, latent_dim)

    # Generate sequence
    z_sequence = predictor.generate_sequence(z_init, num_frames)

    assert z_sequence.shape == (batch_size, num_frames, latent_dim)

    print("✓ Generate sequence from single frame test passed")


def test_generate_sequence_from_multiple_frames():
    """Test generating a sequence starting from multiple frames."""
    batch_size = 2
    latent_dim = 256
    context_len = 4
    num_frames = 10
    init_len = 3

    scheduler = DiffusionScheduler(num_steps=50)
    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=128,
        num_heads=4,
        num_blocks=1,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=context_len,
        num_denoise_steps=10,
    )

    # Initial frames (multiple)
    z_init = torch.randn(batch_size, init_len, latent_dim)

    # Generate sequence
    z_sequence = predictor.generate_sequence(z_init, num_frames)

    assert z_sequence.shape == (batch_size, num_frames, latent_dim)
    # First init_len frames should match z_init
    assert torch.allclose(z_sequence[:, :init_len, :], z_init, atol=1e-5)

    print("✓ Generate sequence from multiple frames test passed")


def test_context_window_handling():
    """Test that context window is properly limited."""
    batch_size = 1
    latent_dim = 64
    context_len = 3
    num_frames = 10

    scheduler = DiffusionScheduler(num_steps=50)
    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=64,
        num_heads=2,
        num_blocks=1,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=context_len,  # Only use last 3 frames as context
        num_denoise_steps=5,
    )

    # Generate long sequence with small context window
    z_init = torch.randn(batch_size, latent_dim)
    z_sequence = predictor.generate_sequence(z_init, num_frames)

    assert z_sequence.shape == (batch_size, num_frames, latent_dim)

    # Verify context window is working by checking model can handle
    # a sequence with more frames than context_len

    print("✓ Context window handling test passed")


def test_latent_trajectory_inference():
    """Test inference of latent trajectory from initial frames."""
    batch_size = 2
    latent_dim = 256
    context_len = 4
    num_frames = 16

    scheduler = DiffusionScheduler(num_steps=50)
    encoder = LatentEncoder(in_channels=3, latent_dim=latent_dim, height=32, width=32)
    denoising_model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=128,
        num_heads=4,
        num_blocks=1,
    )

    predictor = AutoregressiveFramePredictor(
        denoising_model=denoising_model,
        scheduler=scheduler,
        latent_dim=latent_dim,
        context_len=context_len,
        num_denoise_steps=10,
    )

    # Simulate starting with encoded frames
    init_images = torch.randn(batch_size, 2, 3, 32, 32)  # 2 initial frames
    z_init = encoder(init_images.view(batch_size * 2, 3, 32, 32))  # Encode each frame
    z_init = z_init.view(batch_size, 2, latent_dim)

    # Infer full latent trajectory
    z_trajectory = predictor.generate_sequence(z_init, num_frames)

    assert z_trajectory.shape == (batch_size, num_frames, latent_dim)

    # Verify consistency: initial frames should match input
    assert torch.allclose(z_trajectory[:, :2, :], z_init, atol=1e-5)

    print("✓ Latent trajectory inference test passed")


def test_multi_block_sequence_model():
    """Test sequence model with multiple transformer blocks."""
    batch_size = 2
    latent_dim = 256
    context_len = 4

    model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=256,
        num_heads=8,
        num_blocks=3,  # Multiple blocks
        use_linear_attn=False,
        use_rotation_time=False,
    )

    z_context = torch.randn(batch_size, context_len, latent_dim)
    z_t = torch.randn(batch_size, latent_dim)
    t = torch.randint(0, 1000, (batch_size,))

    noise_pred = model(z_context, z_t, t)

    assert noise_pred.shape == z_t.shape

    print("✓ Multi-block sequence model test passed")


def test_variable_context_lengths():
    """Test that sequence model works with various context lengths."""
    batch_size = 2
    latent_dim = 256

    model = SequenceContextDenoisingModel(
        latent_dim=latent_dim,
        d_model=128,
        num_heads=4,
        num_blocks=1,
    )

    z_t = torch.randn(batch_size, latent_dim)
    t = torch.randint(0, 1000, (batch_size,))

    # Test with different context lengths
    for context_len in [1, 2, 4, 8]:
        z_context = torch.randn(batch_size, context_len, latent_dim)
        noise_pred = model(z_context, z_t, t)
        assert noise_pred.shape == z_t.shape

    print("✓ Variable context lengths test passed")


if __name__ == "__main__":
    test_sequence_context_denoising_model_basic()
    test_sequence_context_denoising_with_linear_attention()
    test_sequence_context_denoising_with_rotation_time()
    test_autoregressive_predictor_with_standard_model()
    test_autoregressive_predictor_with_sequence_model()
    test_generate_sequence_from_single_frame()
    test_generate_sequence_from_multiple_frames()
    test_context_window_handling()
    test_latent_trajectory_inference()
    test_multi_block_sequence_model()
    test_variable_context_lengths()
    print("\n✅ All Pass 3 tests passed!")
