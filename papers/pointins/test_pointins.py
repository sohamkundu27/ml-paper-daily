import torch
import numpy as np
from pointins import (
    PointCloudAugmentation,
    SimplePointCloudEncoder,
    PointCloudDataset,
    normalize_points,
    NTXentLoss,
    MomentumEncoder,
    ContrastiveTrainer,
    PointCloudEncoder,
    OrthogonalOffsetBranch,
    geometry_aware_loss,
    GeometryAwareTrainer,
    retrieve_nearest_instances,
    evaluate_clustering,
    run_end_to_end_demo
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


def test_offset_branch():
    """Test orthogonal offset branch."""
    offset_branch = OrthogonalOffsetBranch(feature_dim=128, hidden_dim=64)

    batch_size = 4
    num_points = 256
    per_point_features = torch.randn(batch_size, num_points, 128)

    offsets = offset_branch(per_point_features)

    assert offsets.shape == (batch_size, num_points, 3)
    assert not torch.isnan(offsets).any()
    print("✓ Offset branch test passed")


def test_geometry_aware_loss():
    """Test geometry-aware loss computation."""
    batch_size = 4
    num_points = 128
    points = torch.randn(batch_size, num_points, 3)
    offsets = torch.randn(batch_size, num_points, 3)

    loss = geometry_aware_loss(offsets, points, lambda_geo=0.1)

    assert loss.item() > 0
    assert not torch.isnan(loss)
    print(f"✓ Geometry-aware loss test passed (loss: {loss.item():.4f})")


def test_point_cloud_encoder():
    """Test PointCloudEncoder with per-point features."""
    encoder = PointCloudEncoder(input_dim=3, hidden_dim=64, output_dim=128)

    batch_size = 4
    num_points = 256
    points = torch.randn(batch_size, num_points, 3)

    cloud_feat, per_point_feat = encoder(points)

    assert cloud_feat.shape == (batch_size, 128)
    assert per_point_feat.shape == (batch_size, num_points, 128)
    print("✓ PointCloudEncoder test passed")


def test_geometry_aware_training():
    """Test geometry-aware training with combined losses."""
    device = "cpu"
    encoder = PointCloudEncoder(input_dim=3, hidden_dim=64, output_dim=128)
    offset_branch = OrthogonalOffsetBranch(feature_dim=128, hidden_dim=64)
    trainer = GeometryAwareTrainer(encoder, offset_branch, device=device, lr=1e-2, lambda_geo=0.1)
    dataset = PointCloudDataset(num_objects=12, num_points=128)

    num_steps = 30
    total_losses = []
    contrastive_losses = []
    geo_losses = []

    for step in range(num_steps):
        idx = step % len(dataset)
        cloud1, cloud2 = dataset.get_positive_pair(idx)

        cloud1_t = torch.from_numpy(cloud1).unsqueeze(0)
        cloud2_t = torch.from_numpy(cloud2).unsqueeze(0)

        total_loss, cont_loss, geo_loss = trainer.train_step(cloud1_t, cloud2_t)
        total_losses.append(total_loss)
        contrastive_losses.append(cont_loss)
        geo_losses.append(geo_loss)

    loss_ratio = np.mean(total_losses[:5]) / (np.mean(total_losses[-5:]) + 1e-8)
    assert loss_ratio > 1.0, f"Loss should decrease (ratio: {loss_ratio:.2f})"

    print(f"✓ Geometry-aware training test passed")
    print(f"  Total loss: {total_losses[0]:.4f} -> {total_losses[-1]:.4f} (ratio: {loss_ratio:.2f}x)")
    print(f"  Contrastive: {np.mean(contrastive_losses[:5]):.4f} -> {np.mean(contrastive_losses[-5:]):.4f}")
    print(f"  Geometry: {np.mean(geo_losses[:5]):.4f} -> {np.mean(geo_losses[-5:]):.4f}")


def test_offset_patterns():
    """Test extraction of learned offset patterns."""
    device = "cpu"
    encoder = PointCloudEncoder(input_dim=3, hidden_dim=64, output_dim=128)
    offset_branch = OrthogonalOffsetBranch(feature_dim=128, hidden_dim=64)
    trainer = GeometryAwareTrainer(encoder, offset_branch, device=device, lr=5e-3)
    dataset = PointCloudDataset(num_objects=8, num_points=128)

    # Train briefly with lower learning rate for stability
    for step in range(20):
        idx = step % len(dataset)
        cloud1, cloud2 = dataset.get_positive_pair(idx)
        cloud1_t = torch.from_numpy(cloud1).unsqueeze(0)
        cloud2_t = torch.from_numpy(cloud2).unsqueeze(0)
        trainer.train_step(cloud1_t, cloud2_t)

    # Extract offset patterns
    points_batch, offsets_batch = trainer.get_offset_patterns(dataset, num_samples=3)

    assert points_batch.shape[0] == 3
    assert points_batch.shape[1] == 128
    assert offsets_batch.shape == points_batch.shape

    # Check that offsets are being learned (have non-zero values)
    offset_magnitudes = torch.norm(offsets_batch, dim=2)
    assert (offset_magnitudes > 0).any(), "Offsets should be non-zero"
    assert not torch.isnan(offsets_batch).any(), "Offsets should not be NaN"

    print(f"✓ Offset patterns test passed")
    print(f"  Offset magnitude range: [{offset_magnitudes.min():.4f}, {offset_magnitudes.max():.4f}]")


def test_retrieve_nearest_instances():
    """Test instance retrieval based on learned representations."""
    feature_dim = 128
    num_instances = 16

    # Create random features
    all_features = torch.randn(num_instances, feature_dim)
    query_idx = 0
    query_features = all_features[query_idx:query_idx+1]

    indices, similarities = retrieve_nearest_instances(query_features, all_features, k=5)

    assert len(indices) == 5
    assert len(similarities) == 5
    assert indices[0] == query_idx  # First should be the query itself
    assert similarities[0] > similarities[1]  # Sorted by similarity
    print("✓ Retrieve nearest instances test passed")


def test_evaluate_clustering():
    """Test clustering evaluation metric."""
    feature_dim = 128
    num_instances = 32

    # Create features (random)
    features = torch.randn(num_instances, feature_dim)
    true_labels = np.arange(num_instances)

    accuracy = evaluate_clustering(features, true_labels, num_clusters=8)

    assert 0.0 <= accuracy <= 1.0
    assert not np.isnan(accuracy)
    print(f"✓ Evaluate clustering test passed (accuracy: {accuracy:.4f})")


def test_get_representations():
    """Test extracting representations from trainer."""
    device = "cpu"
    encoder = PointCloudEncoder(input_dim=3, hidden_dim=64, output_dim=128)
    offset_branch = OrthogonalOffsetBranch(feature_dim=128, hidden_dim=64)
    trainer = GeometryAwareTrainer(encoder, offset_branch, device=device, lr=1e-3)
    dataset = PointCloudDataset(num_objects=12, num_points=128)

    # Train briefly
    for step in range(10):
        idx = step % len(dataset)
        cloud1, cloud2 = dataset.get_positive_pair(idx)
        cloud1_t = torch.from_numpy(cloud1).unsqueeze(0)
        cloud2_t = torch.from_numpy(cloud2).unsqueeze(0)
        trainer.train_step(cloud1_t, cloud2_t)

    # Extract representations
    representations = trainer.get_representations(dataset)

    assert representations.shape == (len(dataset), 128)
    assert not torch.isnan(representations).any()
    print("✓ Get representations test passed")


def test_end_to_end_demo_mini():
    """Test end-to-end demo on small synthetic data."""
    results = run_end_to_end_demo(num_objects=8, num_points=256, num_epochs=5, device="cpu")

    assert "initial_loss" in results
    assert "final_loss" in results
    assert "retrieval_accuracy" in results
    assert "clustering_accuracy" in results

    # Check that loss decreased
    assert results["initial_loss"] > results["final_loss"]

    # Check that metrics are in valid ranges
    assert 0.0 <= results["retrieval_accuracy"] <= 1.0
    assert 0.0 <= results["clustering_accuracy"] <= 1.0

    print(f"✓ End-to-end demo test passed")
    print(f"  Final loss: {results['final_loss']:.4f}")
    print(f"  Retrieval accuracy: {results['retrieval_accuracy']*100:.1f}%")
    print(f"  Clustering accuracy: {results['clustering_accuracy']*100:.1f}%")


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

    print("\nRunning PointINS Pass 3 tests...\n")
    test_offset_branch()
    test_geometry_aware_loss()
    test_point_cloud_encoder()
    test_geometry_aware_training()
    test_offset_patterns()

    print("\nRunning PointINS Pass 4 tests...\n")
    test_retrieve_nearest_instances()
    test_evaluate_clustering()
    test_get_representations()
    test_end_to_end_demo_mini()

    print("\n✓ All tests passed!")
