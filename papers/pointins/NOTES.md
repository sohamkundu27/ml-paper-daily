# PointINS: Instance-Aware Self-Supervised Learning for Point Clouds

**arXiv**: https://arxiv.org/abs/2603.25165

**Submitted**: March 26, 2026

**Authors**: (See arXiv for full author list)

## Summary

PointINS proposes an instance-oriented self-supervised learning framework for learning rich 3D point cloud representations without human annotations. The key insight is to combine instance discrimination (learning to distinguish different 3D objects from one another) with geometric reasoning through an orthogonal offset branch. This allows the model to learn both high-level semantic understanding and fine-grained geometric structure from unlabeled 3D scene data.

## Plan: 4 passes

**Pass 1**: Basic point cloud processing and positive/negative pair generation
- Implement point cloud loading and augmentation (rotation, jittering, scaling)
- Create instance-level contrastive pairs from 3D scenes
- Simple baseline encoder (basic PointNet-style MLP on point coordinates)
- Test on synthetic 3D data

**Pass 2**: Instance discrimination loss and contrastive learning
- Implement contrastive loss (SimCLR-style or NT-Xent)
- Add memory bank or online momentum encoder for stability
- Train on unlabeled synthetic point clouds
- Demonstrate that the encoder learns discriminative features

**Pass 3**: Orthogonal offset branch and geometry-aware learning
- Add orthogonal offset prediction branch (predicts per-point geometric offsets)
- Implement geometry-aware regularization
- Combine instance discrimination with geometric reasoning
- Visualize learned offset patterns

**Pass 4**: End-to-end demo on synthetic data + summary
- Create a full pipeline: point cloud loading → augmentation → forward pass → loss computation
- Demonstrate downstream task performance (e.g., clustering, nearest-neighbor retrieval)
- Provide synthetic dataset generation
- Document what worked and what was simplified

## Implemented vs. simplified

### Pass 1 Status: COMPLETE

**Implemented**:
- Point cloud augmentation with random rotation, jittering, and isotropic scaling
- Synthetic point cloud dataset generation (num_objects, num_points configurable)
- Positive pair generation: two augmentations of the same instance
- Negative pair generation: augmentations from different instances
- SimplePointCloudEncoder: PointNet-inspired architecture with per-point MLPs and max-pooling
  - Input: (batch, num_points, 3) coordinates
  - Output: (batch, 128) global point cloud features
- Point cloud normalization (zero-mean, unit-variance)
- Comprehensive test suite with 8 test cases covering augmentation, encoding, batch processing, and end-to-end forward pass

**Simplified/Stubbed**:
- Encoder is very minimal: only 3 fully-connected layers on point coordinates (no edge features, no hierarchical pooling like PointNet++)
- No contrastive loss yet (Pass 2 will add NT-Xent loss)
- No memory bank or momentum encoder (Pass 2 will add)
- No geometric offset branch (Pass 3)
- Synthetic data is random uniform point clouds (not realistic 3D shapes)
- No downstream evaluation or clustering tasks yet
