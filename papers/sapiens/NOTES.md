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

