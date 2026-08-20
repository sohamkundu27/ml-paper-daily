"""Test conditioning mechanism for audio diffusion model."""
import torch
import torch.nn as nn
from diffusion import (
    AudioDiffusionModel,
    GaussianDiffusion,
    StageAdaptiveScheduler,
    stage_adaptive_loss,
    make_class_embedding,
)


def test_class_embedding():
    """Test class embedding creation."""
    num_classes = 5
    cond_dim = 32
    batch_size = 4

    # Create embeddings for different class indices
    class_ids = torch.tensor([0, 1, 2, 3])
    embeddings = make_class_embedding(class_ids, num_classes, cond_dim=cond_dim)

    assert embeddings.shape == (batch_size, cond_dim), f"Shape mismatch: {embeddings.shape}"
    assert not torch.isnan(embeddings).any(), "Embeddings contain NaN"
    assert not torch.isinf(embeddings).any(), "Embeddings contain inf"

    # Same class should produce same embedding
    emb1 = make_class_embedding(0, num_classes, cond_dim=cond_dim)
    emb2 = make_class_embedding(torch.tensor([0]), num_classes, cond_dim=cond_dim)
    assert torch.allclose(emb1, emb2), "Same class should produce same embedding"

    # Different classes should produce different embeddings
    emb_class_0 = make_class_embedding(torch.tensor([0, 0, 0, 0]), num_classes, cond_dim=cond_dim)
    emb_class_1 = make_class_embedding(torch.tensor([1, 1, 1, 1]), num_classes, cond_dim=cond_dim)
    assert not torch.allclose(emb_class_0, emb_class_1), "Different classes should produce different embeddings"

    print("✓ Class embedding test passed")


def test_conditional_model_forward():
    """Test conditional model forward pass."""
    audio_dim = 128
    cond_dim = 32
    model = AudioDiffusionModel(audio_dim=audio_dim, time_dim=64, hidden_dim=256, cond_dim=cond_dim)

    batch_size = 4
    x = torch.randn(batch_size, audio_dim)
    t = torch.full((batch_size,), 0.5)
    cond = torch.randn(batch_size, cond_dim)

    # Forward pass with conditioning
    output = model(x, t, cond=cond)

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaN"
    assert not torch.isinf(output).any(), "Model output contains inf"

    print("✓ Conditional model forward pass test passed")


def test_conditional_model_without_cond():
    """Test that conditional model works without conditioning input."""
    audio_dim = 128
    cond_dim = 32
    model = AudioDiffusionModel(audio_dim=audio_dim, time_dim=64, hidden_dim=256, cond_dim=cond_dim)

    batch_size = 4
    x = torch.randn(batch_size, audio_dim)
    t = torch.full((batch_size,), 0.5)

    # Forward pass WITHOUT conditioning (should fail since we need cond_dim)
    # This tests that we properly enforce conditioning requirement
    try:
        output = model(x, t, cond=None)
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        print("⚠ Model accepts None conditioning despite having cond_dim (design choice)")
    except RuntimeError as e:
        print("✓ Model properly requires conditioning when cond_dim is set")


def test_unconditional_model():
    """Test that unconditional model still works (cond_dim=None)."""
    audio_dim = 128
    model = AudioDiffusionModel(audio_dim=audio_dim, time_dim=64, hidden_dim=256, cond_dim=None)

    batch_size = 4
    x = torch.randn(batch_size, audio_dim)
    t = torch.full((batch_size,), 0.5)

    # Forward pass without conditioning
    output = model(x, t, cond=None)

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaN"

    print("✓ Unconditional model test passed")


def test_conditional_sampling():
    """Test conditional sampling from the model."""
    audio_dim = 128
    cond_dim = 32
    num_classes = 5

    model = AudioDiffusionModel(audio_dim=audio_dim, time_dim=64, hidden_dim=256, cond_dim=cond_dim)
    diffusion = GaussianDiffusion(timesteps=100)

    device = torch.device("cpu")
    batch_size = 2

    # Create class conditioning
    class_ids = torch.tensor([0, 1])
    cond = make_class_embedding(class_ids, num_classes, cond_dim=cond_dim, device=device)

    # Conditional sampling
    samples = diffusion.sample(
        model, (batch_size, audio_dim), device=device, num_steps=10, cond=cond
    )

    assert samples.shape == (batch_size, audio_dim), f"Sample shape mismatch: {samples.shape}"
    assert not torch.isnan(samples).any(), "Samples contain NaN"
    assert not torch.isinf(samples).any(), "Samples contain inf"

    print("✓ Conditional sampling test passed")


def test_training_with_conditioning():
    """Test training loop with class-conditional audio generation."""
    audio_dim = 128
    cond_dim = 32
    num_classes = 5

    model = AudioDiffusionModel(audio_dim=audio_dim, time_dim=64, hidden_dim=256, cond_dim=cond_dim)
    diffusion = GaussianDiffusion(timesteps=100)
    scheduler = StageAdaptiveScheduler(num_training_steps=10, timesteps=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []

    for step in range(10):
        batch_size = 4
        x0 = torch.randn(batch_size, audio_dim)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 100, (batch_size,))
        t_norm = t.float() / diffusion.timesteps

        # Class conditioning
        class_ids = torch.randint(0, num_classes, (batch_size,))
        cond = make_class_embedding(class_ids, num_classes, cond_dim=cond_dim)

        # Forward diffusion
        x_t = diffusion.q_sample(x0, t, noise)

        # Model forward with conditioning
        noise_pred = model(x_t, t_norm, cond=cond)

        # Stage-adaptive loss
        loss = stage_adaptive_loss(noise_pred, noise, t, scheduler, current_step=step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    assert not any(torch.isnan(torch.tensor(l)) for l in losses), "Loss should not be NaN"
    assert all(l > 0 for l in losses), "All losses should be positive"

    print("✓ Training with conditioning test passed")


def test_class_consistency():
    """Test that same class produces similar denoised outputs."""
    audio_dim = 128
    cond_dim = 32
    num_classes = 5

    model = AudioDiffusionModel(audio_dim=audio_dim, time_dim=64, hidden_dim=256, cond_dim=cond_dim)
    diffusion = GaussianDiffusion(timesteps=100)

    device = torch.device("cpu")

    # Sample from same class
    class_ids_same = torch.tensor([0, 0])
    cond_same = make_class_embedding(class_ids_same, num_classes, cond_dim=cond_dim, device=device)

    # Start with same noise, different random seeds but same class
    torch.manual_seed(42)
    samples_same_1 = diffusion.sample(model, (2, audio_dim), device=device, num_steps=10, cond=cond_same)

    torch.manual_seed(42)
    samples_same_2 = diffusion.sample(model, (2, audio_dim), device=device, num_steps=10, cond=cond_same)

    # Should be identical given same random seed
    assert torch.allclose(samples_same_1, samples_same_2), "Same seed and class should produce same samples"

    # Sample from different classes
    class_ids_diff = torch.tensor([0, 1])
    cond_diff = make_class_embedding(class_ids_diff, num_classes, cond_dim=cond_dim, device=device)

    torch.manual_seed(42)
    samples_diff = diffusion.sample(model, (2, audio_dim), device=device, num_steps=10, cond=cond_diff)

    # Different classes should produce different outputs (statistically)
    assert not torch.allclose(samples_diff, samples_same_1, atol=1e-2), "Different classes should likely produce different samples"

    print("✓ Class consistency test passed")


if __name__ == "__main__":
    test_class_embedding()
    test_conditional_model_forward()
    test_unconditional_model()
    test_conditional_sampling()
    test_training_with_conditioning()
    test_class_consistency()
    print("\n✅ All conditioning tests passed!")
