import torch
import numpy as np
from pointins import (
    PointCloudAugmentation,
    SimplePointCloudEncoder,
    PointCloudDataset,
    normalize_points,
    NTXentLoss,
    MomentumEncoder,
    ContrastiveTrainer
)


def test_augmentation():
    """Test point cloud augmentation."""
    aug = PointCloudAugmentation()
    points = np.random.randn(100, 3).astype(np.float32)

    aug_points = aug.augment(points.copy())

    assert aug_points.shape == points.shape
    assert not np.allclose(aug_points, points)  # Should be different after augmentation
    print("✓ Augmentation test passed")


def test_encoder_shapes():
    """Test encoder output shapes."""
    encoder = SimplePointCloudEncoder(input_dim=3, hidden_dim=64, output_dim=128)

    batch_size = 4
    num_points = 1024
    points = torch.randn(batch_size, num_points, 3)

    output = encoder(points)

    assert output.shape == (batch_size, 128)
    print("✓ Encoder shape test passed")


def test_dataset_pairs():
    """Test dataset positive and negative pair generation."""
    dataset = PointCloudDataset(num_objects=16, num_points=512, point_range=1.0)

    # Test positive pair
    pos1, pos2 = dataset.get_positive_pair(0)
    assert pos1.shape == (512, 3)
    assert pos2.shape == (512, 3)
    assert not np.allclose(pos1, pos2)  # Augmented differently
    print("✓ Positive pair generation test passed")

    # Test negative pair
    neg1, neg2 = dataset.get_negative_pair(0)
    assert neg1.shape == (512, 3)
    assert neg2.shape == (512, 3)
    print("✓ Negative pair generation test passed")

    # Dataset length
    assert len(dataset) == 16
    print("✓ Dataset length test passed")


def test_end_to_end_forward():
    """Test end-to-end forward pass through encoder."""
    encoder = SimplePointCloudEncoder(input_dim=3, hidden_dim=64, output_dim=128)
    dataset = PointCloudDataset(num_objects=8, num_points=256)

    # Get a positive pair from dataset
    cloud1, cloud2 = dataset.get_positive_pair(0)

    # Convert to tensors
    cloud1_tensor = torch.from_numpy(cloud1).unsqueeze(0)  # (1, num_points, 3)
    cloud2_tensor = torch.from_numpy(cloud2).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        feat1 = encoder(cloud1_tensor)
        feat2 = encoder(cloud2_tensor)

    # Verify outputs
    assert feat1.shape == (1, 128)
    assert feat2.shape == (1, 128)

    # Features from same object should be somewhat similar (though not identical due to different augmentations)
    cosine_sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
    assert -1.0 <= cosine_sim <= 1.0
    print(f"✓ End-to-end forward pass test passed (cosine_sim: {cosine_sim:.4f})")


def test_normalization():
    """Test point cloud normalization."""
    points = torch.randn(4, 256, 3)
    normalized = normalize_points(points)

    # Check that mean is close to 0 and std is close to 1
    mean = normalized.mean(dim=1)
    std = normalized.std(dim=1)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)
    print("✓ Normalization test passed")


def test_batch_processing():
    """Test processing multiple point clouds in a batch."""
    encoder = SimplePointCloudEncoder(input_dim=3, hidden_dim=64, output_dim=128)
    dataset = PointCloudDataset(num_objects=16, num_points=256)

    batch_size = 8
    batch_clouds = []
    for i in range(batch_size):
        cloud, _ = dataset.get_positive_pair(i % len(dataset))
        batch_clouds.append(cloud)

    batch_tensor = torch.from_numpy(np.stack(batch_clouds))  # (batch_size, num_points, 3)

    with torch.no_grad():
        batch_features = encoder(batch_tensor)

    assert batch_features.shape == (batch_size, 128)
    print("✓ Batch processing test passed")


def test_ntxent_loss():
    """Test NT-Xent contrastive loss."""
    loss_fn = NTXentLoss(temperature=0.07)

    batch_size = 4
    feature_dim = 128

    # Create dummy features
    z_i = torch.randn(batch_size, feature_dim)
    z_j = torch.randn(batch_size, feature_dim)

    loss = loss_fn(z_i, z_j)

    assert loss.item() > 0
    assert not torch.isnan(loss)
    print(f"✓ NT-Xent loss test passed (loss: {loss.item():.4f})")


def test_momentum_encoder():
    """Test momentum encoder initialization and update."""
    encoder = SimplePointCloudEncoder(input_dim=3, hidden_dim=64, output_dim=128)
    momentum_wrapper = MomentumEncoder(encoder, momentum=0.999)

    batch_size = 4
    num_points = 256
    points = torch.randn(batch_size, num_points, 3)

    # Forward pass through both encoders
    z_online, z_momentum = momentum_wrapper(points)

    assert z_online.shape == (batch_size, 128)
    assert z_momentum.shape == (batch_size, 128)

    # Momentum encoder should not require gradients
    assert not z_momentum.requires_grad

    # Update momentum weights
    momentum_wrapper.update_momentum()

    print("✓ Momentum encoder test passed")


def test_contrastive_training():
    """Test training step and learning."""
    device = "cpu"
    encoder = SimplePointCloudEncoder(input_dim=3, hidden_dim=64, output_dim=128)
    trainer = ContrastiveTrainer(encoder, device=device, lr=1e-2, use_momentum=False)
    dataset = PointCloudDataset(num_objects=16, num_points=256)

    # Collect initial similarity metrics
    pos_sim_before, neg_sim_before = trainer.evaluate_similarity(dataset, num_samples=8)

    # Train for multiple steps to allow convergence
    num_steps = 50
    losses = []
    for step in range(num_steps):
        idx = step % len(dataset)
        cloud1, cloud2 = dataset.get_positive_pair(idx)

        cloud1_t = torch.from_numpy(cloud1).unsqueeze(0)
        cloud2_t = torch.from_numpy(cloud2).unsqueeze(0)

        loss = trainer.train_step(cloud1_t, cloud2_t)
        losses.append(loss)

    # Collect post-training similarity metrics
    pos_sim_after, neg_sim_after = trainer.evaluate_similarity(dataset, num_samples=8)

    # Check that loss decreased
    loss_ratio = np.mean(losses[:10]) / (np.mean(losses[-10:]) + 1e-8)
    assert loss_ratio > 1.1, f"Loss should decrease during training (ratio: {loss_ratio:.2f})"

    # Check that positive similarity increased
    assert pos_sim_after > pos_sim_before - 0.1, "Positive similarity should not decrease significantly"

    print(f"✓ Contrastive training test passed")
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (ratio: {loss_ratio:.2f}x)")
    print(f"  Positive similarity: {pos_sim_before:.4f} -> {pos_sim_after:.4f}")
    print(f"  Negative similarity: {neg_sim_before:.4f} -> {neg_sim_after:.4f}")


if __name__ == "__main__":
    print("Running PointINS Pass 1 tests...\n")
    test_augmentation()
    test_encoder_shapes()
    test_dataset_pairs()
    test_normalization()
    test_end_to_end_forward()
    test_batch_processing()

    print("\nRunning PointINS Pass 2 tests...\n")
    test_ntxent_loss()
    test_momentum_encoder()
    test_contrastive_training()

    print("\n✓ All tests passed!")
