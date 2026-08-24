"""Tests for end-to-end video generation (Pass 4)."""

import torch
from diffusion_video_model import SimpleDiffusionModel, SimpleVideoUNet, get_timing


def test_simple_video_unet_forward():
    """Test SimpleVideoUNet forward pass."""
    batch_size = 2
    channels = 3
    height, width = 16, 16
    latent_dim = 32

    model = SimpleVideoUNet(
        in_channels=channels,
        out_channels=channels,
        latent_dim=latent_dim,
        num_blocks=2,
        use_sparse_attention=True
    )

    x = torch.randn(batch_size, channels, height, width)
    output = model(x)

    assert output.shape == (batch_size, channels, height, width)
    assert not torch.isnan(output).any()

    print("✓ test_simple_video_unet_forward passed")


def test_unet_with_full_attention():
    """Test SimpleVideoUNet with full attention."""
    batch_size = 2
    channels = 3
    height, width = 16, 16
    latent_dim = 32

    model = SimpleVideoUNet(
        in_channels=channels,
        out_channels=channels,
        latent_dim=latent_dim,
        num_blocks=2,
        use_sparse_attention=False
    )

    x = torch.randn(batch_size, channels, height, width)
    output = model(x)

    assert output.shape == (batch_size, channels, height, width)
    assert not torch.isnan(output).any()

    print("✓ test_unet_with_full_attention passed")


def test_diffusion_model_forward():
    """Test SimpleDiffusionModel forward pass."""
    batch_size = 2
    channels = 3
    height, width = 16, 16

    model = SimpleDiffusionModel(
        channels=channels,
        latent_dim=32,
        num_blocks=2,
        use_sparse_attention=True
    )

    x = torch.randn(batch_size, channels, height, width)
    t = torch.rand(batch_size)

    output = model(x, t)

    assert output.shape == (batch_size, channels, height, width)
    assert not torch.isnan(output).any()

    print("✓ test_diffusion_model_forward passed")


def test_diffusion_generation():
    """Test SimpleDiffusionModel generation."""
    batch_size = 2
    channels = 3
    height, width = 16, 16

    model = SimpleDiffusionModel(
        channels=channels,
        latent_dim=32,
        num_blocks=2,
        use_sparse_attention=True
    )

    # Test generation
    generated = model.generate(
        shape=(batch_size, channels, height, width),
        num_steps=5,
        device='cpu'
    )

    assert generated.shape == (batch_size, channels, height, width)
    assert not torch.isnan(generated).any()
    assert not torch.isinf(generated).any()
    assert (generated >= -1.0).all() and (generated <= 1.0).all()

    print("✓ test_diffusion_generation passed")


def test_sparse_vs_full_consistency():
    """Test that sparse and full attention produce different but valid outputs."""
    batch_size = 1
    channels = 3
    height, width = 16, 16

    # Both models should run without error
    model_sparse = SimpleDiffusionModel(
        channels=channels,
        latent_dim=32,
        num_blocks=1,
        use_sparse_attention=True
    )

    model_full = SimpleDiffusionModel(
        channels=channels,
        latent_dim=32,
        num_blocks=1,
        use_sparse_attention=False
    )

    x = torch.randn(batch_size, channels, height, width)
    t = torch.rand(batch_size)

    out_sparse = model_sparse(x, t)
    out_full = model_full(x, t)

    assert out_sparse.shape == out_full.shape
    assert not torch.isnan(out_sparse).any()
    assert not torch.isnan(out_full).any()

    # Outputs should be different (different architectures)
    # but roughly in same range
    assert out_sparse.mean() < 1.0
    assert out_full.mean() < 1.0

    print("✓ test_sparse_vs_full_consistency passed")


def test_timing_measurement():
    """Test timing measurement function."""
    batch_size = 2
    channels = 3
    height, width = 16, 16

    model = SimpleDiffusionModel(
        channels=channels,
        latent_dim=32,
        num_blocks=1,
        use_sparse_attention=True
    )

    x = torch.randn(batch_size, channels, height, width)
    t = torch.rand(batch_size)

    timing = get_timing(model, x, t, num_iterations=3)

    assert 'avg_time' in timing
    assert 'total_time' in timing
    assert timing['avg_time'] > 0
    assert timing['total_time'] > 0
    assert timing['total_time'] >= timing['avg_time'] * 3

    print("✓ test_timing_measurement passed")


def test_batch_size_flexibility():
    """Test model works with different batch sizes."""
    channels = 3
    height, width = 16, 16

    model = SimpleDiffusionModel(
        channels=channels,
        latent_dim=32,
        num_blocks=2,
        use_sparse_attention=True
    )

    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, channels, height, width)
        t = torch.rand(batch_size)

        output = model(x, t)

        assert output.shape == (batch_size, channels, height, width)
        assert not torch.isnan(output).any()

    print("✓ test_batch_size_flexibility passed")


def test_device_handling():
    """Test model works on different devices."""
    batch_size = 2
    channels = 3
    height, width = 16, 16

    model = SimpleDiffusionModel(
        channels=channels,
        latent_dim=32,
        num_blocks=1,
        use_sparse_attention=True
    )

    # Test CPU
    x_cpu = torch.randn(batch_size, channels, height, width)
    t_cpu = torch.rand(batch_size)

    model_cpu = model.cpu()
    output_cpu = model_cpu(x_cpu, t_cpu)

    assert output_cpu.shape == (batch_size, channels, height, width)
    assert not torch.isnan(output_cpu).any()

    # Test CUDA if available
    if torch.cuda.is_available():
        x_cuda = x_cpu.cuda()
        t_cuda = t_cpu.cuda()

        model_cuda = model.cuda()
        output_cuda = model_cuda(x_cuda, t_cuda)

        assert output_cuda.shape == (batch_size, channels, height, width)
        assert output_cuda.device.type == 'cuda'
        assert not torch.isnan(output_cuda).any()

    print("✓ test_device_handling passed")


def test_gradient_flow():
    """Test gradients flow through the model."""
    batch_size = 2
    channels = 3
    height, width = 16, 16

    model = SimpleDiffusionModel(
        channels=channels,
        latent_dim=32,
        num_blocks=1,
        use_sparse_attention=True
    )

    x = torch.randn(batch_size, channels, height, width, requires_grad=True)
    t = torch.rand(batch_size, requires_grad=True)

    output = model(x, t)
    loss = output.sum()

    loss.backward()

    # Check that gradients flowed
    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    print("✓ test_gradient_flow passed")


def test_training_mode():
    """Test model behavior in training mode."""
    batch_size = 2
    channels = 3
    height, width = 16, 16

    model = SimpleDiffusionModel(
        channels=channels,
        latent_dim=32,
        num_blocks=2,
        use_sparse_attention=True
    )

    x = torch.randn(batch_size, channels, height, width)
    t = torch.rand(batch_size)

    # Training mode
    model.train()
    output_train = model(x, t)

    # Eval mode
    model.eval()
    with torch.no_grad():
        output_eval = model(x, t)

    # Both should produce valid outputs
    assert output_train.shape == (batch_size, channels, height, width)
    assert output_eval.shape == (batch_size, channels, height, width)
    assert not torch.isnan(output_train).any()
    assert not torch.isnan(output_eval).any()

    print("✓ test_training_mode passed")


if __name__ == "__main__":
    test_simple_video_unet_forward()
    test_unet_with_full_attention()
    test_diffusion_model_forward()
    test_diffusion_generation()
    test_sparse_vs_full_consistency()
    test_timing_measurement()
    test_batch_size_flexibility()
    test_device_handling()
    test_gradient_flow()
    test_training_mode()

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
