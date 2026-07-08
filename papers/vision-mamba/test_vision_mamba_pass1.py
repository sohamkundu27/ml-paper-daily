import torch
from vision_mamba import (
    ImagePatcher, PositionalEmbedding, LinearSSMBlock,
    VisionMambaBlock, VisionMambaPass1
)


def test_image_patcher():
    """Test image patching."""
    patcher = ImagePatcher(image_size=32, patch_size=4, in_channels=3, embed_dim=64)
    x = torch.randn(2, 3, 32, 32)
    patches = patcher(x)
    assert patches.shape == (2, 64, 64), f"Expected (2, 64, 64), got {patches.shape}"
    print("✓ ImagePatcher test passed")


def test_positional_embedding():
    """Test positional embeddings."""
    pe = PositionalEmbedding(num_patches=64, embed_dim=64)
    x = torch.randn(2, 64, 64)
    x_with_pe = pe(x)
    assert x_with_pe.shape == (2, 64, 64), f"Expected (2, 64, 64), got {x_with_pe.shape}"
    # Verify that PE was added (not just returned input)
    assert not torch.allclose(x, x_with_pe), "Positional embedding was not applied"
    print("✓ PositionalEmbedding test passed")


def test_linear_ssm_block():
    """Test linear SSM block."""
    ssm = LinearSSMBlock(embed_dim=64, hidden_dim=128)
    x = torch.randn(2, 64, 64)
    output = ssm(x)
    assert output.shape == (2, 64, 64), f"Expected (2, 64, 64), got {output.shape}"
    print("✓ LinearSSMBlock test passed")


def test_vision_mamba_block():
    """Test Vision Mamba block with residual."""
    block = VisionMambaBlock(embed_dim=64, ssm_hidden_dim=128)
    x = torch.randn(2, 64, 64)
    output = block(x)
    assert output.shape == (2, 64, 64), f"Expected (2, 64, 64), got {output.shape}"
    print("✓ VisionMambaBlock test passed")


def test_vision_mamba_pass1():
    """Test full Vision Mamba Pass 1 model."""
    model = VisionMambaPass1(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    # Output should be (batch, num_patches, embed_dim)
    num_patches = (32 // 4) ** 2
    expected_shape = (4, num_patches, 64)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print("✓ VisionMambaPass1 test passed")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    model = VisionMambaPass1(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients were computed
    assert x.grad is not None, "Gradients did not flow to input"
    assert model.patcher.proj.weight.grad is not None, "Gradients did not flow to patcher"
    print("✓ Gradient flow test passed")


def test_batch_consistency():
    """Test that model produces consistent results."""
    torch.manual_seed(42)
    model = VisionMambaPass1(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)

    assert torch.allclose(output1, output2), "Model output is not deterministic"
    print("✓ Batch consistency test passed")


if __name__ == "__main__":
    test_image_patcher()
    test_positional_embedding()
    test_linear_ssm_block()
    test_vision_mamba_block()
    test_vision_mamba_pass1()
    test_gradient_flow()
    test_batch_consistency()
    print("\n✅ All tests passed!")
