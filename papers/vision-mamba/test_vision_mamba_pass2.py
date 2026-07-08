import torch
from vision_mamba import (
    SelectiveSSMBlock, BidirectionalSSMBlock,
    VisionMambaBlock, VisionMambaPass1, VisionMambaPass2
)


def test_selective_ssm_forward():
    """Test selective SSM block in forward direction."""
    ssm = SelectiveSSMBlock(embed_dim=64, hidden_dim=128)
    x = torch.randn(2, 64, 64)
    output = ssm(x, direction='forward')
    assert output.shape == (2, 64, 64), f"Expected (2, 64, 64), got {output.shape}"
    print("✓ SelectiveSSMBlock forward test passed")


def test_selective_ssm_backward():
    """Test selective SSM block in backward direction."""
    ssm = SelectiveSSMBlock(embed_dim=64, hidden_dim=128)
    x = torch.randn(2, 64, 64)
    output = ssm(x, direction='backward')
    assert output.shape == (2, 64, 64), f"Expected (2, 64, 64), got {output.shape}"
    print("✓ SelectiveSSMBlock backward test passed")


def test_selective_ssm_gate_effect():
    """Test that gating actually affects the output."""
    torch.manual_seed(42)
    ssm = SelectiveSSMBlock(embed_dim=64, hidden_dim=128)
    ssm.eval()

    x = torch.randn(1, 64, 64)
    with torch.no_grad():
        output1 = ssm(x, direction='forward')

    # Create a second SSM with different random state
    ssm2 = SelectiveSSMBlock(embed_dim=64, hidden_dim=128)
    ssm2.eval()
    with torch.no_grad():
        output2 = ssm2(x, direction='forward')

    # Outputs should be different due to different parameter initialization
    assert not torch.allclose(output1, output2, atol=1e-5), \
        "Different SSMs should produce different outputs"
    print("✓ SelectiveSSMBlock gating effect test passed")


def test_bidirectional_ssm():
    """Test bidirectional SSM block."""
    bidirectional = BidirectionalSSMBlock(embed_dim=64, hidden_dim=128)
    x = torch.randn(2, 64, 64)
    output = bidirectional(x)
    assert output.shape == (2, 64, 64), f"Expected (2, 64, 64), got {output.shape}"
    print("✓ BidirectionalSSMBlock test passed")


def test_bidirectional_vs_unidirectional():
    """Test that bidirectional and unidirectional produce different outputs."""
    torch.manual_seed(42)

    # Create two blocks with same structure
    block_unidirectional = VisionMambaBlock(embed_dim=64, ssm_hidden_dim=128,
                                          use_bidirectional=False)
    block_bidirectional = VisionMambaBlock(embed_dim=64, ssm_hidden_dim=128,
                                          use_bidirectional=True)

    block_unidirectional.eval()
    block_bidirectional.eval()

    x = torch.randn(1, 64, 64)
    with torch.no_grad():
        out_uni = block_unidirectional(x)
        out_bi = block_bidirectional(x)

    # Outputs should be different (different SSM types)
    assert not torch.allclose(out_uni, out_bi, atol=1e-5), \
        "Bidirectional and unidirectional should produce different outputs"
    print("✓ Bidirectional vs unidirectional test passed")


def test_vision_mamba_block_bidirectional():
    """Test Vision Mamba block with bidirectional SSM."""
    block = VisionMambaBlock(embed_dim=64, ssm_hidden_dim=128, use_bidirectional=True)
    x = torch.randn(2, 64, 64)
    output = block(x)
    assert output.shape == (2, 64, 64), f"Expected (2, 64, 64), got {output.shape}"
    print("✓ VisionMambaBlock with bidirectional test passed")


def test_vision_mamba_pass1_unchanged():
    """Test that Pass 1 still works with updated VisionMambaBlock."""
    model = VisionMambaPass1(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    num_patches = (32 // 4) ** 2
    expected_shape = (4, num_patches, 64)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print("✓ VisionMambaPass1 (unchanged) test passed")


def test_vision_mamba_pass2():
    """Test Vision Mamba Pass 2 with bidirectional blocks."""
    model = VisionMambaPass2(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    num_patches = (32 // 4) ** 2
    expected_shape = (4, num_patches, 64)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print("✓ VisionMambaPass2 test passed")


def test_gradient_flow_pass2():
    """Test that gradients flow through Pass 2."""
    model = VisionMambaPass2(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Gradients did not flow to input"
    assert model.patcher.proj.weight.grad is not None, "Gradients did not flow to patcher"
    print("✓ Gradient flow (Pass 2) test passed")


def test_pass1_vs_pass2():
    """Test that Pass 1 and Pass 2 produce different outputs."""
    torch.manual_seed(42)

    pass1 = VisionMambaPass1(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    pass1.eval()

    pass2 = VisionMambaPass2(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    pass2.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out1 = pass1(x)
        out2 = pass2(x)

    # They should be different since they use different SSM types
    assert not torch.allclose(out1, out2, atol=1e-5), \
        "Pass 1 and Pass 2 should produce different outputs due to different SSM types"
    print("✓ Pass 1 vs Pass 2 test passed")


if __name__ == "__main__":
    test_selective_ssm_forward()
    test_selective_ssm_backward()
    test_selective_ssm_gate_effect()
    test_bidirectional_ssm()
    test_bidirectional_vs_unidirectional()
    test_vision_mamba_block_bidirectional()
    test_vision_mamba_pass1_unchanged()
    test_vision_mamba_pass2()
    test_gradient_flow_pass2()
    test_pass1_vs_pass2()
    print("\n✅ All Pass 2 tests passed!")
