#!/usr/bin/env python3
"""
Pass 4: End-to-End Demo and Evaluation for Sapiens

Demonstrates the complete pipeline:
1. Generate small synthetic dataset of articulated bodies
2. Train multi-task model end-to-end
3. Perform inference and visualize outputs
4. Report evaluation metrics (keypoint accuracy, depth MSE, normal MAE)
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Tuple
from pose_detector import (
    MultiTaskPoseDepthNormal, MultiTaskLoss, SyntheticBodyDataset,
    create_data_batch, Trainer, compute_pck, compute_depth_metrics,
    compute_normal_metrics, create_synthetic_image_with_points, points_to_heatmap,
    create_synthetic_depth, create_synthetic_normals
)


def visualize_keypoints_on_image(image: np.ndarray, keypoint_heatmaps: np.ndarray,
                                 threshold: float = 0.3) -> np.ndarray:
    """Overlay keypoint predictions on image

    Args:
        image: (H, W, 3) RGB image in [0, 1]
        keypoint_heatmaps: (num_keypoints, H_hm, W_hm) heatmap predictions
        threshold: Minimum activation to display

    Returns:
        (H, W, 3) image with keypoints overlaid
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        viz = (image * 255).astype(np.uint8)
    else:
        viz = image

    H, W = viz.shape[:2]
    hm_h, hm_w = keypoint_heatmaps.shape[1:]
    scale_x = W / hm_w
    scale_y = H / hm_h

    num_keypoints = keypoint_heatmaps.shape[0]
    colors = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255]
    ])

    for k in range(num_keypoints):
        heatmap = keypoint_heatmaps[k]
        if np.max(heatmap) > threshold:
            y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            x, y = int(x_idx * scale_x), int(y_idx * scale_y)

            color = tuple(colors[k % len(colors)].tolist())
            radius = 3
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            viz[ny, nx] = color

    return viz


def visualize_depth_map(depth: np.ndarray) -> np.ndarray:
    """Convert depth map to RGB visualization

    Args:
        depth: (H, W) or (1, H, W) depth map

    Returns:
        (H, W, 3) RGB visualization (grayscale)
    """
    if len(depth.shape) == 3:
        depth = depth[0]

    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_rgb = np.stack([depth_normalized] * 3, axis=2)

    return (depth_rgb * 255).astype(np.uint8)


def visualize_normal_map(normals: np.ndarray) -> np.ndarray:
    """Convert normal map to RGB visualization

    Args:
        normals: (3, H, W) or (H, W, 3) normal vectors in [-1, 1]

    Returns:
        (H, W, 3) RGB visualization
    """
    if len(normals.shape) == 3 and normals.shape[0] == 3:
        normals = normals.transpose(1, 2, 0)

    normals_rgb = (normals + 1.0) / 2.0
    normals_rgb = np.clip(normals_rgb, 0, 1)

    return (normals_rgb * 255).astype(np.uint8)


def run_end_to_end_demo(num_train_samples: int = 100, num_val_samples: int = 20,
                        num_epochs: int = 10, batch_size: int = 8,
                        device: str = 'cpu', output_dir: str = 'outputs'):
    """Complete training and evaluation pipeline

    Args:
        num_train_samples: Number of synthetic training samples
        num_val_samples: Number of validation samples
        num_epochs: Training epochs
        batch_size: Batch size
        device: 'cpu' or 'cuda'
        output_dir: Directory to save visualizations
    """
    print("=" * 70)
    print("SAPIENS PASS 4: END-TO-END DEMO")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1] Generating synthetic dataset...")
    print(f"    Training samples: {num_train_samples}")
    print(f"    Validation samples: {num_val_samples}")

    train_dataset = SyntheticBodyDataset(num_samples=num_train_samples, image_size=256)
    val_dataset = SyntheticBodyDataset(num_samples=num_val_samples, image_size=256)

    print(f"✓ Dataset ready")

    print(f"\n[2] Initializing model and training...")
    print(f"    Device: {device}")
    print(f"    Epochs: {num_epochs}, Batch size: {batch_size}")

    model = MultiTaskPoseDepthNormal(num_keypoints=17, image_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = MultiTaskLoss()
    trainer = Trainer(model, optimizer, loss_fn, device=device)

    all_train_losses = []
    all_val_losses = []
    all_metrics = []

    for epoch in range(num_epochs):
        train_epoch_loss = trainer.train_epoch(train_dataset, batch_size=batch_size)
        val_epoch_loss, val_metrics = trainer.validate(val_dataset, batch_size=batch_size)

        all_train_losses.append(train_epoch_loss)
        all_val_losses.append(val_epoch_loss)
        all_metrics.append(val_metrics)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:2d}/{num_epochs} | "
                  f"Train Loss: {train_epoch_loss['total']:.4f} | "
                  f"Val Loss: {val_epoch_loss['total']:.4f} | "
                  f"Depth MSE: {val_metrics['depth_mse']:.4f} | "
                  f"Normal MAE: {val_metrics['normal_mae_angle']:.2f}°")

    print(f"✓ Training complete")

    print(f"\n[3] Running inference on test samples...")
    model.eval()

    test_dataset = SyntheticBodyDataset(num_samples=5, image_size=256)
    test_indices = np.arange(len(test_dataset))
    test_images, test_targets = create_data_batch(test_dataset, len(test_dataset), test_indices)
    test_images = test_images.to(device)

    with torch.no_grad():
        test_predictions = model(test_images)

    test_pred_keypoints = model.get_keypoints_from_heatmaps(test_predictions['keypoints'])
    test_pred_depth = test_predictions['depth'].cpu().numpy()
    test_pred_normals = test_predictions['normals'].cpu().numpy()

    test_targets_keypoints = torch.from_numpy(
        np.array([test_dataset[i]['keypoints'] for i in range(len(test_dataset))])
    )
    test_targets_depth = test_targets['depth'].numpy()
    test_targets_normals = test_targets['normals'].numpy()

    print(f"✓ Inference complete on {len(test_dataset)} test samples")

    print(f"\n[4] Computing evaluation metrics...")

    test_keypoints_np = test_pred_keypoints.cpu().numpy()
    target_keypoints_np = np.array([
        test_dataset[i]['keypoints'] for i in range(len(test_dataset))
    ])
    target_keypoints_np = np.mean(target_keypoints_np, axis=(2, 3))
    target_keypoints_np = np.expand_dims(target_keypoints_np, axis=2)

    pck_score = compute_pck(test_keypoints_np, target_keypoints_np, threshold=0.2, image_size=256)
    depth_metrics = compute_depth_metrics(test_pred_depth, test_targets_depth)
    normal_metrics = compute_normal_metrics(test_pred_normals, test_targets_normals)

    print(f"  PCK@0.2 (Keypoint Accuracy): {pck_score:.4f}")
    print(f"  Depth MSE: {depth_metrics['mse']:.6f}")
    print(f"  Depth MAE: {depth_metrics['mae']:.6f}")
    print(f"  Normal MAE (angle): {normal_metrics['mae_angle']:.2f}°")

    print(f"\n[5] Saving visualizations...")

    for i in range(min(3, len(test_dataset))):
        sample = test_dataset[i]
        image_np = (sample['image'].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        pred_hm = test_predictions['keypoints'][i].cpu().numpy()
        pred_depth = test_pred_depth[i]
        pred_normal = test_pred_normals[i]

        target_hm = sample['keypoints'].numpy()
        target_depth = sample['depth'].numpy()
        target_normal = sample['normals'].numpy()

        pred_keypoint_viz = visualize_keypoints_on_image(
            sample['image'].numpy().transpose(1, 2, 0), pred_hm
        )
        target_keypoint_viz = visualize_keypoints_on_image(
            sample['image'].numpy().transpose(1, 2, 0), target_hm
        )

        pred_depth_viz = visualize_depth_map(pred_depth)
        target_depth_viz = visualize_depth_map(target_depth)

        pred_normal_viz = visualize_normal_map(pred_normal)
        target_normal_viz = visualize_normal_map(target_normal)

        output_prefix = f"{output_dir}/sample_{i:02d}"
        np.save(f"{output_prefix}_pred_keypoint.npy", pred_keypoint_viz)
        np.save(f"{output_prefix}_target_keypoint.npy", target_keypoint_viz)
        np.save(f"{output_prefix}_pred_depth.npy", pred_depth_viz)
        np.save(f"{output_prefix}_target_depth.npy", target_depth_viz)
        np.save(f"{output_prefix}_pred_normal.npy", pred_normal_viz)
        np.save(f"{output_prefix}_target_normal.npy", target_normal_viz)

    print(f"✓ Saved visualizations to {output_dir}")

    print(f"\n[6] Training history summary...")
    print(f"  Initial train loss: {all_train_losses[0]['total']:.4f}")
    print(f"  Final train loss:   {all_train_losses[-1]['total']:.4f}")
    print(f"  Initial val loss:   {all_val_losses[0]['total']:.4f}")
    print(f"  Final val loss:     {all_val_losses[-1]['total']:.4f}")

    print("\n" + "=" * 70)
    print("END-TO-END DEMO COMPLETE")
    print("=" * 70)

    return {
        'pck_score': pck_score,
        'depth_metrics': depth_metrics,
        'normal_metrics': normal_metrics,
        'training_history': {
            'train_losses': all_train_losses,
            'val_losses': all_val_losses,
            'metrics': all_metrics
        }
    }


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = run_end_to_end_demo(
        num_train_samples=100,
        num_val_samples=20,
        num_epochs=10,
        batch_size=8,
        device=device,
        output_dir='outputs'
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"PCK@0.2 Score: {results['pck_score']:.4f}")
    print(f"Depth MSE: {results['depth_metrics']['mse']:.6f}")
    print(f"Depth MAE: {results['depth_metrics']['mae']:.6f}")
    print(f"Normal MAE (angle): {results['normal_metrics']['mae_angle']:.2f}°")
    print("=" * 70)
