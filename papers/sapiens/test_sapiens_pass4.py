import torch
import numpy as np
from pose_detector import MultiTaskPoseDepthNormal, MultiTaskLoss, SyntheticBodyDataset, Trainer
from demo_sapiens_pass4 import (
    visualize_keypoints_on_image, visualize_depth_map, visualize_normal_map,
    run_end_to_end_demo
)


def test_visualize_keypoints():
    """Test keypoint visualization"""
    image = np.random.rand(256, 256, 3)
    heatmaps = np.random.rand(17, 64, 64)

    viz = visualize_keypoints_on_image(image, heatmaps)

    assert viz.shape == (256, 256, 3)
    assert viz.dtype == np.uint8
    print("✓ Keypoint visualization test passed")


def test_visualize_depth():
    """Test depth map visualization"""
    depth = np.random.uniform(0.5, 2.0, (64, 64))

    viz = visualize_depth_map(depth)

    assert viz.shape == (64, 64, 3)
    assert viz.dtype == np.uint8
    print("✓ Depth visualization test passed")


def test_visualize_normals():
    """Test normal map visualization"""
    normals = np.random.uniform(-1, 1, (3, 64, 64))

    viz = visualize_normal_map(normals)

    assert viz.shape == (64, 64, 3)
    assert viz.dtype == np.uint8
    print("✓ Normal visualization test passed")


def test_end_to_end_training():
    """Test complete end-to-end training pipeline"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = SyntheticBodyDataset(num_samples=20, image_size=256)
    val_dataset = SyntheticBodyDataset(num_samples=5, image_size=256)

    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = MultiTaskLoss()
    trainer = Trainer(model, optimizer, loss_fn, device=device)

    train_loss = trainer.train_epoch(train_dataset, batch_size=4)
    val_loss, metrics = trainer.validate(val_dataset, batch_size=4)

    assert 'total' in train_loss
    assert 'keypoint' in train_loss
    assert 'depth' in train_loss
    assert 'normal' in train_loss

    assert 'total' in val_loss
    assert 'depth_mse' in metrics
    assert 'normal_mae_angle' in metrics

    print("✓ End-to-end training test passed")
    print(f"  Train loss: {train_loss['total']:.4f}")
    print(f"  Val loss: {val_loss['total']:.4f}")
    print(f"  Depth MSE: {metrics['depth_mse']:.6f}")
    print(f"  Normal MAE: {metrics['normal_mae_angle']:.2f}°")


def test_inference_shapes():
    """Test inference output shapes"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    model.to(device)
    model.eval()

    batch_size = 4
    test_images = torch.randn(batch_size, 3, 256, 256).to(device)

    with torch.no_grad():
        outputs = model(test_images)

    assert outputs['keypoints'].shape == (batch_size, 17, 64, 64)
    assert outputs['depth'].shape == (batch_size, 1, 64, 64)
    assert outputs['normals'].shape == (batch_size, 3, 64, 64)

    keypoints = model.get_keypoints_from_heatmaps(outputs['keypoints'])
    assert keypoints.shape == (batch_size, 17, 2)

    print("✓ Inference shapes test passed")


if __name__ == '__main__':
    print("Running Pass 4 tests...")
    test_visualize_keypoints()
    test_visualize_depth()
    test_visualize_normals()
    test_end_to_end_training()
    test_inference_shapes()
    print("\n✓ All Pass 4 tests passed!")
