import torch
import numpy as np
from pose_detector import (
    MultiTaskPoseDepthNormal, MultiTaskLoss, train_step,
    create_synthetic_image_with_points, points_to_heatmap,
    create_synthetic_depth, create_synthetic_normals
)


def test_multi_task_forward_pass():
    """Test multi-task model forward pass with all heads"""
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256)

    outputs = model(images)

    assert 'keypoints' in outputs
    assert 'depth' in outputs
    assert 'normals' in outputs

    assert outputs['keypoints'].shape == (batch_size, 17, 64, 64)
    assert outputs['depth'].shape == (batch_size, 1, 64, 64)
    assert outputs['normals'].shape == (batch_size, 3, 64, 64)

    assert not torch.isnan(outputs['keypoints']).any()
    assert not torch.isnan(outputs['depth']).any()
    assert not torch.isnan(outputs['normals']).any()

    print("✓ Multi-task forward pass test passed")


def test_depth_head():
    """Test depth prediction head produces valid depth values"""
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    batch_size = 1
    images = torch.randn(batch_size, 3, 256, 256)

    outputs = model(images)
    depth = outputs['depth']

    assert depth.shape == (batch_size, 1, 64, 64)
    assert torch.all(torch.isfinite(depth))
    print("✓ Depth head test passed")


def test_normal_head():
    """Test normal prediction head produces normalized vectors"""
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    batch_size = 1
    images = torch.randn(batch_size, 3, 256, 256)

    outputs = model(images)
    normals = outputs['normals']

    assert normals.shape == (batch_size, 3, 64, 64)

    norms = torch.sqrt(torch.sum(normals**2, dim=1, keepdim=True))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    print("✓ Normal head test passed")


def test_loss_function():
    """Test multi-task loss computation"""
    loss_fn = MultiTaskLoss(keypoint_weight=1.0, depth_weight=1.0, normal_weight=1.0)

    batch_size = 2
    keypoint_pred = torch.randn(batch_size, 17, 64, 64, requires_grad=True)
    depth_pred = torch.randn(batch_size, 1, 64, 64, requires_grad=True)
    normal_pred = torch.randn(batch_size, 3, 64, 64, requires_grad=True)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=1)

    predictions = {
        'keypoints': keypoint_pred,
        'depth': depth_pred,
        'normals': normal_pred
    }

    keypoint_target = torch.sigmoid(torch.randn(batch_size, 17, 64, 64))
    depth_target = torch.clamp(torch.randn(batch_size, 1, 64, 64), 0.5, 2.0)
    normal_target = torch.randn(batch_size, 3, 64, 64)
    normal_target = torch.nn.functional.normalize(normal_target, p=2, dim=1)

    targets = {
        'keypoints': keypoint_target,
        'depth': depth_target,
        'normals': normal_target
    }

    losses = loss_fn(predictions, targets)

    assert 'total' in losses
    assert 'keypoint' in losses
    assert 'depth' in losses
    assert 'normal' in losses

    assert losses['total'].requires_grad
    assert losses['keypoint'].item() >= 0
    assert losses['depth'].item() >= 0
    assert losses['normal'].item() >= 0

    print("✓ Loss function test passed")


def test_training_step():
    """Test a complete training step with all three heads"""
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = MultiTaskLoss(keypoint_weight=1.0, depth_weight=1.0, normal_weight=1.0)

    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256)

    keypoint_target = torch.sigmoid(torch.randn(batch_size, 17, 64, 64))
    depth_target = torch.clamp(torch.randn(batch_size, 1, 64, 64), 0.5, 2.0)
    normal_target = torch.randn(batch_size, 3, 64, 64)
    normal_target = torch.nn.functional.normalize(normal_target, p=2, dim=1)

    targets = {
        'keypoints': keypoint_target,
        'depth': depth_target,
        'normals': normal_target
    }

    losses = train_step(model, optimizer, images, targets, loss_fn)

    assert 'total' in losses
    assert 'keypoint' in losses
    assert 'depth' in losses
    assert 'normal' in losses

    assert losses['total'] > 0

    print("✓ Training step test passed")


def test_synthetic_data_generation():
    """Test generation of synthetic targets for all three tasks"""
    image_size = 256
    heatmap_size = 64

    image_np, keypoints_np = create_synthetic_image_with_points(image_size, 17)
    heatmaps_np = points_to_heatmap(keypoints_np, image_size, heatmap_size)
    depth_np = create_synthetic_depth(image_size, heatmap_size)
    normals_np = create_synthetic_normals(image_size, heatmap_size)

    assert image_np.shape == (256, 256, 3)
    assert keypoints_np.shape == (17, 2)
    assert heatmaps_np.shape == (17, 64, 64)
    assert depth_np.shape == (64, 64)
    assert normals_np.shape == (64, 64, 3)

    assert np.all(heatmaps_np >= 0) and np.all(heatmaps_np <= 1)
    assert np.all(depth_np >= 0.5) and np.all(depth_np <= 2.0)

    normal_norms = np.sqrt(np.sum(normals_np**2, axis=2))
    assert np.allclose(normal_norms, 1.0, atol=1e-5)

    print("✓ Synthetic data generation test passed")


def test_end_to_end_training():
    """Test complete training with synthetic data"""
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MultiTaskLoss()

    image_np, keypoints_np = create_synthetic_image_with_points(256, 17)
    heatmaps_np = points_to_heatmap(keypoints_np, 256, 64)
    depth_np = create_synthetic_depth(256, 64)
    normals_np = create_synthetic_normals(256, 64)

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)

    heatmap_tensor = torch.from_numpy(heatmaps_np).unsqueeze(0)
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
    normal_tensor = torch.from_numpy(normals_np).permute(2, 0, 1).unsqueeze(0)

    targets = {
        'keypoints': heatmap_tensor,
        'depth': depth_tensor,
        'normals': normal_tensor
    }

    initial_losses = []
    for step in range(5):
        losses = train_step(model, optimizer, image_tensor, targets, loss_fn)
        initial_losses.append(losses['total'])

    assert initial_losses[0] > 0
    print(f"  Training loss over 5 steps: {[f'{l:.4f}' for l in initial_losses]}")
    print("✓ End-to-end training test passed")


def test_model_gradient_flow():
    """Test that gradients flow to all heads"""
    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    images = torch.randn(1, 3, 256, 256, requires_grad=True)

    outputs = model(images)

    total_loss = (
        outputs['keypoints'].mean() +
        outputs['depth'].mean() +
        outputs['normals'].mean()
    )
    total_loss.backward()

    assert images.grad is not None
    assert not torch.isnan(images.grad).any()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"

    print("✓ Gradient flow test passed")


if __name__ == "__main__":
    test_multi_task_forward_pass()
    test_depth_head()
    test_normal_head()
    test_loss_function()
    test_training_step()
    test_synthetic_data_generation()
    test_end_to_end_training()
    test_model_gradient_flow()
    print("\n✅ All Pass 2 tests passed!")
