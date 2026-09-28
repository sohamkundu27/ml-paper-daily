# Sapiens: Foundation Model for Human Body Understanding

## Paper Details
- **Title**: Sapiens: Foundation Models for Human Body Understanding
- **arXiv**: https://arxiv.org/abs/2411.14891
- **Authors**: Jing Lin, Ailing Zeng, Haoqian Wang, et al.
- **Published**: November 2024

## Summary
Sapiens is a foundation model for understanding human bodies in images. Rather than training separate models for different tasks (pose estimation, depth prediction, surface normal prediction), Sapiens learns a unified visual representation of human body geometry. The model uses a vision transformer backbone trained on diverse human body datasets, enabling strong zero-shot transfer to pose, depth, and normal estimation tasks. A key insight is that shared body geometry understanding benefits all human-centric tasks, making the approach efficient and generalizable.

## Plan: 4 Passes

### Pass 1: Basic Pose Keypoint Detection
Implement the foundational component: a vision transformer backbone that predicts 2D human body keypoints.
- Implement a lightweight ViT encoder (or simple CNN alternative) on images
- Add a keypoint heatmap prediction head that outputs 2D coordinates for body joints
- Implement image preprocessing and basic augmentation
- Write tests validating keypoint detection on synthetic/toy data
- Run inference on a test image

### Pass 2: Multi-Head Regression (Pose + Depth)
Extend to predict multiple geometric outputs from the same backbone:
- Keep the shared ViT backbone from Pass 1
- Add depth prediction head (pixel-wise depth regression)
- Add surface normal prediction head (3-channel normal vectors per pixel)
- Implement joint training with multi-task loss
- Test that all three heads train together

### Pass 3: Synthetic Data Training Pipeline
Build a realistic training system on toy/synthetic data:
- Create synthetic human body dataset (e.g., simple articulated figures or SMPL rendering)
- Implement full training loop with batch processing, validation, and loss tracking
- Add data loading, preprocessing, and augmentation
- Train on multiple geometric targets simultaneously
- Evaluate reconstruction metrics (MSE for depth/normals, PCK for keypoints)

### Pass 4: End-to-End Demo and Evaluation
Demonstrate the full system on toy data with results:
- Generate small-scale synthetic dataset of articulated bodies
- Train multi-task model end-to-end
- Perform inference and visualize outputs (keypoints, depth, normals)
- Report metrics (keypoint accuracy, depth MSE, normal MAE)
- Document what worked, simplified choices, and gaps vs. paper

## Implemented vs. Simplified

### Pass 1 Complete

✓ `SimpleCNNBackbone`: Lightweight 3-layer CNN encoder (stride-2 pooling at layers 1-2)
✓ `KeypointDetector`: Model that predicts heatmaps for 17 body keypoints
✓ Heatmap prediction head: Outputs (B, 17, 64, 64) heatmaps from CNN features
✓ Keypoint extraction: Argmax operation to extract (x, y) coordinates from heatmaps
✓ Synthetic data generation: Functions to create test images and Gaussian heatmaps
✓ 6 comprehensive tests validating backbone, extraction, heatmap generation, gradients, and end-to-end pipeline

### Pass 1 Simplified/Stubbed

- Uses 3-layer CNN instead of Vision Transformer (simpler, faster to train)
- No multi-scale features or feature pyramid network (single resolution heatmaps)
- Heatmap size fixed at 64×64 (image_size // 4)
- No soft-argmax for differentiable coordinate regression (uses hard argmax)
- No loss function implementation yet (Pass 2 will add training)
- Keypoint detection is inference-only (no training loop)
- Synthetic data uses simple Gaussian heatmaps (not SMPL rendering or real data)
- No depth or normal head (Pass 2 adds multi-task outputs)

### Pass 2 Complete

✓ `MultiTaskPoseDepthNormal`: Model with three prediction heads on shared CNN backbone
✓ `DepthHead`: Conv layers predicting per-pixel depth (1 channel)
✓ `NormalHead`: Conv layers predicting per-pixel surface normals (3 channels, L2-normalized)
✓ `create_synthetic_depth()`: Generate synthetic depth maps for training
✓ `create_synthetic_normals()`: Generate synthetic surface normal maps with realistic variation
✓ `MultiTaskLoss`: Combined loss function with separate components for keypoints (binary cross-entropy), depth (MSE), and normals (1 - cosine similarity)
✓ `train_step()`: Single training iteration with backward pass and optimizer step
✓ 8 comprehensive tests validating all three heads, loss computation, training, and gradient flow

### Pass 2 Simplified/Stubbed

- Depth and normal maps are synthetic with simple patterns (not from real SMPL renderings)
- Normal head outputs L2-normalized vectors (no special handling for ambiguous normals)
- Multi-task loss uses uniform weighting (1.0 for all tasks; could be tuned per task)
- Training function is minimal (no learning rate scheduling, early stopping, or validation)
- No evaluation metrics beyond loss values (no depth MAE, normal angle error, etc.)
- Synthetic targets are fixed patterns rather than from a realistic dataset or differentiable render

### Pass 3 Complete

✓ `SyntheticBodyDataset`: Dataset class that generates synthetic samples on-the-fly with image, keypoint heatmaps, depth, and normals
✓ `create_data_batch()`: Batching function to load multiple samples from dataset
✓ `apply_data_augmentation()`: Data augmentation pipeline including horizontal flip, rotation (applied to normals), and random brightness
✓ `compute_pck()`: Metric for percentage of correct keypoints (PCK) with distance threshold
✓ `compute_depth_metrics()`: Depth evaluation metrics (MSE and MAE)
✓ `compute_normal_metrics()`: Surface normal evaluation (mean angular error in degrees)
✓ `Trainer` class: Full training loop with:
  - `train_epoch()`: Single epoch training with batch processing and loss tracking
  - `validate()`: Validation loop with loss computation and metric evaluation
  - `get_training_history()`: Access to all training/validation history
✓ 11 comprehensive tests validating dataset, batching, augmentation, all metrics, trainer initialization, training, validation, and convergence

### Pass 3 Simplified/Stubbed

- Augmentation is minimal (only horizontal flip, brightness, no rotation/scale/perspective transforms)
- No learning rate scheduling (fixed learning rate during training)
- No early stopping mechanism (training runs fixed number of epochs)
- No optimizer state saving/loading (checkpoint functionality not implemented)
- Metrics computed per-batch without per-sample tracking
- PCK threshold fixed at 0.2 (standard but not configurable)
- Dataset generates random samples each time (no caching, slower for large datasets)
- No distributed training or GPU optimization
- Synthetic data still uses simple geometric patterns (not differentiable SMPL rendering)

