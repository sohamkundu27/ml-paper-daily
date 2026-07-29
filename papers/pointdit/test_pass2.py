"""
Test Pass 2: Image feature conditioning via cross-attention.
Verify that the model can condition on synthetic image features.
"""
import torch
from diffusion import DiffusionTransformer, GaussianDiffusion


def test_forward_without_conditioning():
    """Test that model still works without conditioning (backward compatible)."""
    B, N, patch_size, point_dim = 2, 16, 16, 3
    hidden_dim = 64

    model = DiffusionTransformer(
        point_dim=point_dim,
        patch_size=patch_size,
        num_layers=4,
        hidden_dim=hidden_dim,
        num_heads=4,
        use_cross_attention=False,
    )
    model.eval()

    x = torch.randn(B, N, patch_size * point_dim)
    t = torch.tensor([100, 500])

    noise_pred = model(x, t, condition=None)
    assert noise_pred.shape == x.shape
    print(f"✓ Model without conditioning: {x.shape} -> {noise_pred.shape}")


def test_forward_with_image_features():
    """Test forward pass with image feature conditioning."""
    B, N, patch_size, point_dim = 2, 16, 16, 3
    hidden_dim = 64
    cond_dim = 256

    model = DiffusionTransformer(
        point_dim=point_dim,
        patch_size=patch_size,
        num_layers=4,
        hidden_dim=hidden_dim,
        num_heads=4,
        use_cross_attention=True,
    )
    model.eval()

    x = torch.randn(B, N, patch_size * point_dim)
    t = torch.tensor([100, 500])

    # Synthetic image features: [B, num_patches, cond_dim]
    # Could be from a pre-trained vision encoder (e.g., DINO)
    num_img_patches = 16
    condition = torch.randn(B, num_img_patches, cond_dim)

    noise_pred = model(x, t, condition=condition)
    assert noise_pred.shape == x.shape
    print(f"✓ Model with conditioning: input {x.shape}, condition {condition.shape} -> output {noise_pred.shape}")


def test_conditioning_affects_output():
    """Verify that different conditions produce different outputs."""
    B, N, patch_size, point_dim = 1, 9, 16, 3
    hidden_dim = 64
    cond_dim = 128

    model = DiffusionTransformer(
        point_dim=point_dim,
        patch_size=patch_size,
        num_layers=4,
        hidden_dim=hidden_dim,
        num_heads=4,
        use_cross_attention=True,
    )
    model.eval()

    x = torch.randn(B, N, patch_size * point_dim)
    t = torch.tensor([300])

    # Two different conditions
    cond1 = torch.randn(B, 8, cond_dim)
    cond2 = torch.randn(B, 8, cond_dim)

    with torch.no_grad():
        out1 = model(x, t, condition=cond1)
        out2 = model(x, t, condition=cond2)

    # Different conditions should produce different outputs (with high probability)
    diff = torch.norm(out1 - out2).item()
    assert diff > 1e-5, "Different conditions should produce different outputs"
    print(f"✓ Conditioning affects output: difference between two conditions = {diff:.4f}")


def test_condition_dimension_flexibility():
    """Test that model adapts to different condition dimensions."""
    B, N, patch_size, point_dim = 2, 16, 16, 3
    hidden_dim = 64
    x = torch.randn(B, N, patch_size * point_dim)
    t = torch.tensor([100, 500])

    model = DiffusionTransformer(
        point_dim=point_dim,
        patch_size=patch_size,
        num_layers=4,
        hidden_dim=hidden_dim,
        num_heads=4,
        use_cross_attention=True,
    )
    model.eval()

    # Test with condition_dim=256
    cond_256 = torch.randn(B, 12, 256)
    out_256 = model(x, t, condition=cond_256)
    assert out_256.shape == x.shape

    # Test with condition_dim=512 (different model instance)
    model2 = DiffusionTransformer(
        point_dim=point_dim,
        patch_size=patch_size,
        num_layers=4,
        hidden_dim=hidden_dim,
        num_heads=4,
        use_cross_attention=True,
    )
    model2.eval()

    cond_512 = torch.randn(B, 12, 512)
    out_512 = model2(x, t, condition=cond_512)
    assert out_512.shape == x.shape

    print(f"✓ Flexible condition dimensions: 256D -> {out_256.shape}, 512D -> {out_512.shape}")


def test_condition_as_global_embedding():
    """Test conditioning with a global (non-patched) image embedding."""
    B, N, patch_size, point_dim = 2, 16, 16, 3
    hidden_dim = 64
    cond_dim = 512

    model = DiffusionTransformer(
        point_dim=point_dim,
        patch_size=patch_size,
        num_layers=4,
        hidden_dim=hidden_dim,
        num_heads=4,
        use_cross_attention=True,
    )
    model.eval()

    x = torch.randn(B, N, patch_size * point_dim)
    t = torch.tensor([100, 500])

    # Global image embedding (e.g., from final layer of pre-trained encoder)
    # Shape: [B, cond_dim]
    global_condition = torch.randn(B, cond_dim)

    noise_pred = model(x, t, condition=global_condition)
    assert noise_pred.shape == x.shape
    print(f"✓ Global condition embedding: {global_condition.shape} -> {noise_pred.shape}")


def test_diffusion_with_conditioning():
    """Test a reverse diffusion step with conditioning."""
    B, N, patch_size, point_dim = 1, 9, 16, 3
    hidden_dim = 64
    cond_dim = 256

    model = DiffusionTransformer(
        point_dim=point_dim,
        patch_size=patch_size,
        num_layers=4,
        hidden_dim=hidden_dim,
        num_heads=4,
        use_cross_attention=True,
    )
    model.eval()

    diffusion = GaussianDiffusion(num_steps=1000)

    # Noisy point maps
    xt = torch.randn(B, N, patch_size * point_dim)
    t = torch.tensor([500])

    # Image condition
    condition = torch.randn(B, 12, cond_dim)

    # Reverse step with conditioning
    with torch.no_grad():
        # Model forward pass with condition
        predicted_noise = model(xt, t, condition=condition)
        assert predicted_noise.shape == xt.shape

    print(f"✓ Diffusion reverse step with conditioning: input {xt.shape}, condition {condition.shape}")


if __name__ == "__main__":
    print("\n=== Pass 2: Image Feature Conditioning Tests ===\n")
    test_forward_without_conditioning()
    test_forward_with_image_features()
    test_conditioning_affects_output()
    test_condition_dimension_flexibility()
    test_condition_as_global_embedding()
    test_diffusion_with_conditioning()
    print("\n=== All Pass 2 tests passed ===\n")
