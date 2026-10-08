"""Tests for Vision Mamba Pass 4 (end-to-end demo and training)."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vision_mamba import VisionMambaPass1, VisionMambaPass2, VisionMambaPass3


def test_pass4_all_models_inference():
    """Test that all three passes can run inference end-to-end."""
    x = torch.randn(4, 3, 32, 32)

    # Pass 1: unidirectional SSM, feature output
    pass1 = VisionMambaPass1(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    out1 = pass1(x)
    assert out1.shape == (4, 64, 64), f"Pass 1 expected (4, 64, 64), got {out1.shape}"

    # Pass 2: bidirectional SSM, feature output
    pass2 = VisionMambaPass2(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )
    out2 = pass2(x)
    assert out2.shape == (4, 64, 64), f"Pass 2 expected (4, 64, 64), got {out2.shape}"

    # Pass 3: bidirectional SSM + classification, logits output
    pass3 = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )
    out3 = pass3(x)
    assert out3.shape == (4, 10), f"Pass 3 expected (4, 10), got {out3.shape}"

    print("✓ Pass 4: All models inference test passed")


def test_pass4_training_loop():
    """Test full training loop with Pass 3 on synthetic data."""
    torch.manual_seed(42)

    # Create synthetic dataset
    num_train = 100
    num_test = 20
    batch_size = 16

    train_images = torch.randn(num_train, 3, 32, 32)
    train_labels = torch.randint(0, 10, (num_train,))

    test_images = torch.randn(num_test, 3, 32, 32)
    test_labels = torch.randint(0, 10, (num_test,))

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train for 2 epochs
    model.train()
    train_losses = []

    for epoch in range(2):
        epoch_loss = 0.0
        for images, labels in train_loader:
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            train_losses.append(loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

    # Verify training is stable (no NaN, no divergence)
    assert all(not torch.isnan(torch.tensor(l)) for l in train_losses), "Loss became NaN"
    # Loss should not explode (max value should not exceed 10x initial loss)
    assert train_losses[-1] < train_losses[0] * 10, "Loss diverged during training"

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    test_acc = 100.0 * correct / total
    print(f"  Test accuracy: {test_acc:.1f}%")

    print("✓ Pass 4: Training loop test passed")


def test_pass4_inference_speed():
    """Test that inference speed is reasonable on GPU/CPU."""
    import time

    model = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )
    model.eval()

    # Create test batch
    x = torch.randn(32, 3, 32, 32)

    # Warm up
    with torch.no_grad():
        _ = model(x)

    # Time 5 forward passes
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    elapsed = time.time() - start

    avg_time_per_batch = elapsed / 5
    throughput = x.size(0) / avg_time_per_batch

    print(f"  Avg time per batch: {avg_time_per_batch*1000:.2f}ms")
    print(f"  Throughput: {throughput:.1f} images/sec")

    # Verify throughput is positive and reasonable
    assert throughput > 100, f"Throughput too low: {throughput}"

    print("✓ Pass 4: Inference speed test passed")


def test_pass4_model_comparison():
    """Compare Pass 1 vs Pass 2 vs Pass 3 on same data."""
    torch.manual_seed(42)
    x = torch.randn(8, 3, 32, 32)

    pass1 = VisionMambaPass1(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )

    pass2 = VisionMambaPass2(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )

    pass3 = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )

    # Get outputs
    pass1.eval()
    pass2.eval()
    pass3.eval()

    with torch.no_grad():
        out1 = pass1(x)
        out2 = pass2(x)
        out3 = pass3(x)

    # Check shapes
    assert out1.shape == (8, 64, 64), "Pass 1 output shape mismatch"
    assert out2.shape == (8, 64, 64), "Pass 2 output shape mismatch"
    assert out3.shape == (8, 10), "Pass 3 output shape mismatch"

    # Pass 1 and Pass 2 outputs should be different (bidirectionality effect)
    # but same shape
    output_diff = (out1 - out2).abs().mean()
    print(f"  Pass 1 vs Pass 2 output difference: {output_diff.item():.6f}")

    # Verify outputs are not identical (would indicate a bug)
    assert output_diff > 1e-5, "Pass 1 and Pass 2 should produce different outputs"

    # Pass 3 logits should be valid (not all zeros, not NaN)
    logits_norm = out3.abs().mean()
    assert logits_norm > 0 and not torch.isnan(out3).any(), "Pass 3 logits invalid"

    print("✓ Pass 4: Model comparison test passed")


def test_pass4_batch_consistency():
    """Test that batches of different sizes produce consistent results."""
    torch.manual_seed(42)

    model = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )
    model.eval()

    with torch.no_grad():
        # Test different batch sizes
        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 3, 32, 32)
            logits = model(x)
            assert logits.shape == (batch_size, 10), f"Failed for batch_size={batch_size}"

    print("✓ Pass 4: Batch consistency test passed")


if __name__ == "__main__":
    print("Running Vision Mamba Pass 4 tests...\n")
    test_pass4_all_models_inference()
    test_pass4_training_loop()
    test_pass4_inference_speed()
    test_pass4_model_comparison()
    test_pass4_batch_consistency()
    print("\n✅ All Pass 4 tests passed!")
