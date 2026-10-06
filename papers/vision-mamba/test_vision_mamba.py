import torch
import numpy as np
from vision_mamba import (
    PositionalEmbedding,
    SSMBlock,
    BidirectionalSSMBlock,
    VisionMambaBlock,
)


def test_positional_embedding():
    """Test positional embeddings."""
    seq_len = 196  # 14x14 patches
    dim = 768
    
    pos_embed = PositionalEmbedding(seq_len, dim)
    x = torch.randn(2, seq_len, dim)
    y = pos_embed(x)
    
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
    assert not torch.allclose(y, x), "Output should differ from input due to positional embedding"
    print("✓ Positional embedding test passed")


def test_ssm_block():
    """Test basic SSM block."""
    batch_size = 2
    seq_len = 196
    dim = 768
    state_dim = 64
    
    ssm = SSMBlock(dim, state_dim)
    x = torch.randn(batch_size, seq_len, dim)
    y = ssm(x)
    
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
    assert y.dtype == x.dtype, f"Expected dtype {x.dtype}, got {y.dtype}"
    assert not torch.isnan(y).any(), "Output contains NaN"
    print("✓ SSM block test passed")


def test_bidirectional_ssm_block():
    """Test bidirectional SSM block."""
    batch_size = 4
    seq_len = 196
    dim = 768
    state_dim = 64
    
    bi_ssm = BidirectionalSSMBlock(dim, state_dim)
    x = torch.randn(batch_size, seq_len, dim)
    y = bi_ssm(x)
    
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
    assert not torch.isnan(y).any(), "Output contains NaN"
    print("✓ Bidirectional SSM block test passed")


def test_vision_mamba_block():
    """Test complete Vision Mamba block."""
    batch_size = 2
    seq_len = 196  # 14x14 patches from 224x224 image
    dim = 768
    state_dim = 64
    
    mamba = VisionMambaBlock(dim, state_dim)
    x = torch.randn(batch_size, seq_len, dim)
    
    # Forward pass
    y = mamba(x)
    
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
    assert not torch.isnan(y).any(), "Output contains NaN"
    assert not torch.isinf(y).any(), "Output contains Inf"
    
    # Check that residual connection works
    assert not torch.allclose(y, x), "Output should be different from input"
    
    print("✓ Vision Mamba block test passed")


def test_gradient_flow():
    """Test that gradients flow through the block."""
    batch_size = 2
    seq_len = 196
    dim = 768
    state_dim = 64
    
    mamba = VisionMambaBlock(dim, state_dim)
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    
    y = mamba(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients should flow to input"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaN"
    
    # Check that model parameters have gradients
    for param in mamba.parameters():
        if param.requires_grad:
            assert param.grad is not None, "All params should have gradients"
            assert not torch.isnan(param.grad).any(), "Param gradient contains NaN"
    
    print("✓ Gradient flow test passed")


def test_multi_patch_sizes():
    """Test with different patch sequence lengths."""
    dim = 768
    state_dim = 64
    mamba = VisionMambaBlock(dim, state_dim)
    
    for seq_len in [49, 196, 256]:  # Different image resolutions
        x = torch.randn(1, seq_len, dim)
        y = mamba(x)
        assert y.shape == x.shape, f"Failed for seq_len={seq_len}"
    
    print("✓ Multi-patch size test passed")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_positional_embedding()
    test_ssm_block()
    test_bidirectional_ssm_block()
    test_vision_mamba_block()
    test_gradient_flow()
    test_multi_patch_sizes()
    
    print("\n✅ All tests passed!")
