import torch
import numpy as np
from pose_detector import (
    MultiTaskPoseDepthNormal, MultiTaskLoss,
    SyntheticBodyDataset, create_data_batch, apply_data_augmentation,
    compute_pck, compute_depth_metrics, compute_normal_metrics,
    Trainer, create_synthetic_image_with_points, points_to_heatmap,
    create_synthetic_depth, create_synthetic_normals
)


def test_synthetic_dataset_creation():
    """Test that dataset creates valid samples"""
    dataset = SyntheticBodyDataset(num_samples=5, image_size=256, heatmap_size=64)

    assert len(dataset) == 5

    sample = dataset[0]
    assert 'image' in sample
    assert 'keypoints' in sample
    assert 'depth' in sample
    assert 'normals' in sample

    assert sample['image'].shape == (3, 256, 256)
    assert sample['keypoints'].shape == (17, 64, 64)
    assert sample['depth'].shape == (1, 64, 64)
    assert sample['normals'].shape == (3, 64, 64)

    assert sample['image'].min() >= 0.0 and sample['image'].max() <= 1.0
    assert sample['depth'].min() >= 0.5 and sample['depth'].max() <= 2.0

    print("✓ Synthetic dataset creation test passed")


def test_batch_creation():
    """Test batch creation from dataset"""
    dataset = SyntheticBodyDataset(num_samples=10)
    batch_size = 4
    indices = np.array([0, 1, 2, 3])

    images, targets = create_data_batch(dataset, batch_size, indices)

    assert images.shape == (4, 3, 256, 256)
    assert targets['keypoints'].shape == (4, 17, 64, 64)
    assert targets['depth'].shape == (4, 1, 64, 64)
    assert targets['normals'].shape == (4, 3, 64, 64)

    assert torch.all(images >= 0.0) and torch.all(images <= 1.0)
    assert torch.all(targets['depth'] >= 0.5) and torch.all(targets['depth'] <= 2.0)

    print("✓ Batch creation test passed")


def test_data_augmentation():
    """Test data augmentation pipeline"""
    dataset = SyntheticBodyDataset(num_samples=1)
    sample = dataset[0]

    image = sample['image'].clone()
    heatmaps = sample['keypoints'].clone()
    depth = sample['depth'].clone()
    normals = sample['normals'].clone()

    aug_image, aug_heatmaps, aug_depth, aug_normals = apply_data_augmentation(
        image, heatmaps, depth, normals
    )

    assert aug_image.shape == image.shape
    assert aug_heatmaps.shape == heatmaps.shape
    assert aug_depth.shape == depth.shape
    assert aug_normals.shape == normals.shape

    assert aug_image.min() >= 0.0 and aug_image.max() <= 1.0

    print("✓ Data augmentation test passed")


def test_pck_metric():
    """Test PCK (Percentage of Correct Keypoints) computation"""
    pred_keypoints = np.array([
        [[128, 128], [130, 130], [100, 100]],
        [[256, 256], [200, 200], [50, 50]]
    ])

    target_keypoints = np.array([
        [[128, 128], [128, 128], [256, 256]],
        [[256, 256], [256, 256], [256, 256]]
    ])

    pck = compute_pck(pred_keypoints, target_keypoints, threshold=0.2, image_size=256)

    assert 0.0 <= pck <= 1.0
    print(f"  PCK score: {pck:.4f}")
    print("✓ PCK metric test passed")


def test_depth_metrics():
    """Test depth prediction metrics"""
    pred_depth = np.random.uniform(0.5, 2.0, (2, 1, 64, 64))
    target_depth = np.random.uniform(0.5, 2.0, (2, 1, 64, 64))

    metrics = compute_depth_metrics(pred_depth, target_depth)

    assert 'mse' in metrics
    assert 'mae' in metrics
    assert metrics['mse'] >= 0.0
    assert metrics['mae'] >= 0.0

    print(f"  Depth MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")
    print("✓ Depth metrics test passed")


def test_normal_metrics():
    """Test surface normal prediction metrics"""
    pred_normals = np.random.randn(2, 3, 64, 64)
    pred_normals /= np.linalg.norm(pred_normals, axis=1, keepdims=True) + 1e-8

    target_normals = np.random.randn(2, 3, 64, 64)
    target_normals /= np.linalg.norm(target_normals, axis=1, keepdims=True) + 1e-8

    metrics = compute_normal_metrics(pred_normals, target_normals)

    assert 'mae_angle' in metrics
    assert 0.0 <= metrics['mae_angle'] <= 180.0

    print(f"  Normal MAE angle: {metrics['mae_angle']:.2f}°")
    print("✓ Normal metrics test passed")


def test_trainer_initialization():
    """Test trainer setup"""
    device = 'cpu'
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = MultiTaskLoss()

    trainer = Trainer(model, optimizer, loss_fn, device=device)

    assert trainer.model is model
    assert trainer.optimizer is optimizer
    assert trainer.loss_fn is loss_fn
    assert trainer.device == device
    assert len(trainer.train_losses) == 0
    assert len(trainer.val_losses) == 0

    print("✓ Trainer initialization test passed")


def test_training_epoch():
    """Test training for one epoch"""
    device = 'cpu'
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MultiTaskLoss()

    trainer = Trainer(model, optimizer, loss_fn, device=device)

    train_dataset = SyntheticBodyDataset(num_samples=8, image_size=256)

    epoch_losses = trainer.train_epoch(train_dataset, batch_size=4)

    assert 'total' in epoch_losses
    assert 'keypoint' in epoch_losses
    assert 'depth' in epoch_losses
    assert 'normal' in epoch_losses

    assert all(v >= 0 for v in epoch_losses.values())
    assert len(trainer.train_losses) == 1

    print(f"  Epoch losses - Total: {epoch_losses['total']:.4f}, "
          f"Keypoint: {epoch_losses['keypoint']:.4f}, "
          f"Depth: {epoch_losses['depth']:.4f}, "
          f"Normal: {epoch_losses['normal']:.4f}")
    print("✓ Training epoch test passed")


def test_validation():
    """Test validation loop"""
    device = 'cpu'
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MultiTaskLoss()

    trainer = Trainer(model, optimizer, loss_fn, device=device)

    val_dataset = SyntheticBodyDataset(num_samples=8, image_size=256)

    val_losses, metrics = trainer.validate(val_dataset, batch_size=4)

    assert 'total' in val_losses
    assert 'depth_mse' in metrics
    assert 'depth_mae' in metrics
    assert 'normal_mae_angle' in metrics

    assert len(trainer.val_losses) == 1
    assert len(trainer.metrics) == 1

    print(f"  Val losses - Total: {val_losses['total']:.4f}")
    print(f"  Metrics - Depth MSE: {metrics['depth_mse']:.4f}, "
          f"Normal angle MAE: {metrics['normal_mae_angle']:.2f}°")
    print("✓ Validation test passed")


def test_full_training_loop():
    """Test complete training loop with multiple epochs and validation"""
    device = 'cpu'
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MultiTaskLoss()

    trainer = Trainer(model, optimizer, loss_fn, device=device)

    train_dataset = SyntheticBodyDataset(num_samples=12, image_size=256)
    val_dataset = SyntheticBodyDataset(num_samples=8, image_size=256)

    num_epochs = 3
    batch_size = 4

    for epoch in range(num_epochs):
        train_losses = trainer.train_epoch(train_dataset, batch_size=batch_size)
        val_losses, metrics = trainer.validate(val_dataset, batch_size=batch_size)

        print(f"  Epoch {epoch + 1} - Train loss: {train_losses['total']:.4f}, "
              f"Val loss: {val_losses['total']:.4f}")

    history = trainer.get_training_history()
    assert len(history['train_losses']) == num_epochs
    assert len(history['val_losses']) == num_epochs
    assert len(history['metrics']) == num_epochs

    print("✓ Full training loop test passed")


def test_training_convergence():
    """Test that training actually reduces loss"""
    device = 'cpu'
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = MultiTaskLoss()

    trainer = Trainer(model, optimizer, loss_fn, device=device)
    train_dataset = SyntheticBodyDataset(num_samples=16, image_size=256)

    first_epoch_loss = None
    for epoch in range(5):
        epoch_losses = trainer.train_epoch(train_dataset, batch_size=8)
        if first_epoch_loss is None:
            first_epoch_loss = epoch_losses['total']

    last_epoch_loss = trainer.train_losses[-1]['total']

    print(f"  First epoch loss: {first_epoch_loss:.4f}, Last epoch loss: {last_epoch_loss:.4f}")
    assert last_epoch_loss < first_epoch_loss, "Loss should decrease over training"

    print("✓ Training convergence test passed")


if __name__ == "__main__":
    test_synthetic_dataset_creation()
    test_batch_creation()
    test_data_augmentation()
    test_pck_metric()
    test_depth_metrics()
    test_normal_metrics()
    test_trainer_initialization()
    test_training_epoch()
    test_validation()
    test_full_training_loop()
    test_training_convergence()
    print("\n✅ All Pass 3 tests passed!")
