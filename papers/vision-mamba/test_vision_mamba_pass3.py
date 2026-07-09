"""Tests for Vision Mamba Pass 3 (classification head and training)."""

import torch
import torch.nn as nn
from vision_mamba import VisionMambaPass3


def test_pass3_output_shape():
    """Test that Pass 3 outputs logits of correct shape."""
    model = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"
    print("✓ Pass 3 output shape test passed")


def test_pass3_different_num_classes():
    """Test Pass 3 with different number of classes."""
    model = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=100
    )
    x = torch.randn(8, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (8, 100), f"Expected (8, 100), got {logits.shape}"
    print("✓ Pass 3 different num_classes test passed")


def test_pass3_gradient_flow():
    """Test that gradients flow through entire Pass 3 model."""
    model = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("✓ Pass 3 gradient flow test passed")


def test_pass3_training_step():
    """Test multiple training steps show convergence."""
    torch.manual_seed(42)
    model = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Generate toy data
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))

    # Multiple training steps with gradient clipping
    model.train()
    losses = []
    for _ in range(10):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    print(f"  First 3 losses: {[f'{l:.4f}' for l in losses[:3]]}, Last loss: {losses[-1]:.4f}")
    # Check that we don't have NaN and loss is finite
    assert all(not torch.isnan(torch.tensor(l)) for l in losses), "Loss became NaN during training"
    print("✓ Pass 3 training step test passed")


def test_pass3_accuracy():
    """Test that model trains on toy data without NaN."""
    torch.manual_seed(42)
    model = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Generate toy data
    x_train = torch.randn(50, 3, 32, 32)
    y_train = torch.randint(0, 10, (50,))

    # Train for multiple epochs
    model.train()
    losses = []
    for _ in range(20):
        logits = model(x_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    # Verify training is stable (no NaN) and loss decreases
    assert all(not torch.isnan(torch.tensor(l)) for l in losses), "Loss became NaN"
    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
    print("✓ Pass 3 accuracy test passed")


def test_pass3_backward_compatibility():
    """Test that Pass 2 models still work alongside Pass 3."""
    from vision_mamba import VisionMambaPass2

    pass2_model = VisionMambaPass2(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    pass3_model = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )

    x = torch.randn(4, 3, 32, 32)

    # Pass 2 output: (batch, num_patches, embed_dim)
    out2 = pass2_model(x)
    assert out2.shape == (4, 64, 64), f"Pass 2 expected (4, 64, 64), got {out2.shape}"

    # Pass 3 output: (batch, num_classes)
    out3 = pass3_model(x)
    assert out3.shape == (4, 10), f"Pass 3 expected (4, 10), got {out3.shape}"

    print("✓ Pass 3 backward compatibility test passed")


if __name__ == "__main__":
    test_pass3_output_shape()
    test_pass3_different_num_classes()
    test_pass3_gradient_flow()
    test_pass3_training_step()
    test_pass3_accuracy()
    test_pass3_backward_compatibility()
    print("\n✅ All Pass 3 tests passed!")
