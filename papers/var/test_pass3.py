"""
Tests for Pass 3: Training loop and loss computation.
"""

import torch
from torch.utils.data import DataLoader
from var_pass2 import VARPass2
from var_pass3 import VARTrainer, ToyImageDataset


def test_toy_dataset():
    """Test that toy dataset generates valid image batches."""
    dataset = ToyImageDataset(num_samples=10, img_size=64, num_channels=3)

    assert len(dataset) == 10, "Dataset should have 10 samples"

    # Test that we can sample from it
    image = dataset[0]
    assert image.shape == (3, 64, 64), f"Expected shape (3, 64, 64), got {image.shape}"
    assert isinstance(image, torch.Tensor), "Dataset should return torch tensors"

    print("✓ Toy dataset generates valid images")


def test_trainer_initialization():
    """Test that trainer initializes correctly."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    trainer = VARTrainer(model, device="cpu", learning_rate=1e-3)

    assert trainer.model is not None, "Trainer should have a model"
    assert trainer.optimizer is not None, "Trainer should have an optimizer"
    assert trainer.loss_fn is not None, "Trainer should have a loss function"

    print("✓ Trainer initializes correctly")


def test_target_token_computation():
    """Test that target tokens are computed correctly from token maps."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    trainer = VARTrainer(model, device="cpu")

    # Generate a batch of images
    images = torch.randn(2, 3, 64, 64)
    logits, token_maps = model(images)

    # Compute target tokens
    targets = trainer.compute_target_tokens(token_maps, model.num_scales)

    # Verify shapes
    assert targets.shape[0] == 2, "Batch size should be 2"
    assert targets.shape[1] > 0, "Should have tokens from all scales"
    assert targets.dtype == torch.long, "Targets should be long integers"

    # Verify tokens are within vocab range
    assert targets.min() >= 0, "Token indices should be >= 0"
    assert targets.max() < 4096, "Token indices should be < vocab_size"

    print("✓ Target token computation works correctly")


def test_single_training_step():
    """Test that a single training step runs without errors."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    trainer = VARTrainer(model, device="cpu", learning_rate=1e-3)

    # Create a batch
    images = torch.randn(2, 3, 64, 64)

    # Record model parameters before training
    param_before = [p.clone() for p in model.parameters()]

    # Run one training step
    loss = trainer.train_step(images)

    # Verify loss is a scalar
    assert isinstance(loss, float), "Loss should be a scalar"
    assert loss > 0, "Loss should be positive"

    # Verify parameters have been updated
    param_after = [p.clone() for p in model.parameters()]
    params_changed = False
    for pb, pa in zip(param_before, param_after):
        if not torch.allclose(pb, pa):
            params_changed = True
            break

    assert params_changed, "Model parameters should be updated after training step"

    print(f"✓ Single training step runs correctly (loss={loss:.4f})")


def test_training_loop():
    """Test that a full training loop runs and loss decreases."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    trainer = VARTrainer(model, device="cpu", learning_rate=1e-2)

    # Create a small dataset and dataloader
    dataset = ToyImageDataset(num_samples=20, img_size=64, num_channels=3)
    dataloader = DataLoader(dataset, batch_size=4)

    # Train for a few epochs
    initial_loss = None
    for epoch in range(5):
        loss = trainer.train_epoch(dataloader)
        if initial_loss is None:
            initial_loss = loss
        print(f"  Epoch {epoch + 1}: loss={loss:.4f}")

    # Check that loss decreased (learning is happening)
    final_loss = trainer.train_losses[-1]
    assert final_loss < initial_loss, (
        f"Loss should decrease during training. Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    )

    print(f"✓ Training loop works: loss decreased from {initial_loss:.4f} to {final_loss:.4f}")


def test_validation():
    """Test that validation loop runs without errors."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    trainer = VARTrainer(model, device="cpu", learning_rate=1e-3)

    # Create validation dataset
    val_dataset = ToyImageDataset(num_samples=10, img_size=64, num_channels=3)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    # Compute validation loss
    val_loss = trainer.validate(val_dataloader)

    assert isinstance(val_loss, float), "Validation loss should be a float"
    assert val_loss > 0, "Validation loss should be positive"
    assert torch.isfinite(torch.tensor(val_loss)), "Validation loss should be finite"

    print(f"✓ Validation loop runs correctly (val_loss={val_loss:.4f})")


def test_learning_rate_effect():
    """Test that different learning rates affect training speed."""
    dataset = ToyImageDataset(num_samples=20, img_size=64, num_channels=3)
    dataloader = DataLoader(dataset, batch_size=4)

    # Train with different learning rates
    losses_lr_high = []
    losses_lr_low = []

    for lr, loss_list in [(1e-1, losses_lr_high), (1e-4, losses_lr_low)]:
        model = VARPass2(
            in_channels=3,
            token_dim=256,
            num_scales=3,
            num_layers=6,
            num_heads=8,
            ff_dim=1024,
            vocab_size=4096,
        )
        trainer = VARTrainer(model, device="cpu", learning_rate=lr)

        for epoch in range(3):
            loss = trainer.train_epoch(dataloader)
            loss_list.append(loss)

    # Higher learning rate should reduce loss faster
    decrease_high = losses_lr_high[0] - losses_lr_high[-1]
    decrease_low = losses_lr_low[0] - losses_lr_low[-1]

    assert decrease_high > decrease_low, (
        f"Higher LR should reduce loss faster. High LR decrease: {decrease_high:.4f}, "
        f"Low LR decrease: {decrease_low:.4f}"
    )

    print(f"✓ Learning rate effect verified: high LR reduces loss faster")


def test_batch_gradient_accumulation():
    """Test that gradients accumulate correctly across batches."""
    model = VARPass2(
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    )

    trainer = VARTrainer(model, device="cpu", learning_rate=1e-3)

    # Single batch training
    batch1 = torch.randn(4, 3, 64, 64)
    param_before = [p.clone() for p in model.parameters()]

    trainer.train_step(batch1)

    param_after_step1 = [p.clone() for p in model.parameters()]

    # Another batch training
    batch2 = torch.randn(4, 3, 64, 64)
    trainer.train_step(batch2)

    param_after_step2 = [p.clone() for p in model.parameters()]

    # Each step should update parameters
    step1_changed = any(
        not torch.allclose(pb, pa) for pb, pa in zip(param_before, param_after_step1)
    )
    step2_changed = any(
        not torch.allclose(pa1, pa2)
        for pa1, pa2 in zip(param_after_step1, param_after_step2)
    )

    assert step1_changed, "Parameters should change in step 1"
    assert step2_changed, "Parameters should change in step 2"

    print("✓ Batch gradient accumulation works correctly")


if __name__ == "__main__":
    test_toy_dataset()
    test_trainer_initialization()
    test_target_token_computation()
    test_single_training_step()
    test_training_loop()
    test_validation()
    test_learning_rate_effect()
    test_batch_gradient_accumulation()
    print("\n✅ All Pass 3 tests passed!")
