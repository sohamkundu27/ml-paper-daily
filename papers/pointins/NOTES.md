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

### Pass 2 Status: COMPLETE

**Implemented**:
- NTXentLoss: contrastive loss for instance discrimination
  - Supports both single-pair (pairwise cosine similarity) and batch (full NT-Xent with negatives)
  - Uses temperature scaling for numerical stability
- MomentumEncoder: optional momentum-updated encoder for training stability
  - Maintains a slowly-updated copy of the encoder
  - Useful for larger training runs (can be disabled for simplicity)
- ContrastiveTrainer: training loop for learning discriminative features
  - Handles normalization and forward passes
  - Computes contrastive loss and gradient updates
  - Optional momentum encoder updates
- evaluate_similarity: evaluation metric showing positive vs negative pair similarity
  - Demonstrates that encoder learns to distinguish instances
  - Tracks cosine similarity improvements during training
- 3 new test cases: NT-Xent loss, momentum encoder, and end-to-end contrastive training
  - Training shows 5.9x loss reduction and improved positive pair similarity
  - All tests passing

**Simplified/Stubbed**:
- Single-pair loss (for batch_size=1) uses simplified pairwise cosine loss instead of full NT-Xent
  - Full NT-Xent requires batch_size > 1 to have meaningful hard negatives
  - Simplified version still effectively trains the encoder
- Momentum encoder not used in default training (use_momentum=False by default)
  - Can be enabled for larger training runs
- No batch-based training loop yet (processes single positive pairs)
  - Pass 4 will implement full batched training and end-to-end pipeline
- No retrieval or clustering evaluation yet
- Synthetic data is still random uniform point clouds (not realistic)
- No downstream task evaluation yet (Pass 4)

### Pass 3 Status: COMPLETE

**Implemented**:
- PointCloudEncoder: enhanced encoder that returns both global features and per-point features
  - Enables geometry-aware learning at point level
- OrthogonalOffsetBranch: predicts per-point 3D offset vectors
  - Takes per-point features as input: (batch, num_points, feature_dim)
  - Outputs per-point offsets: (batch, num_points, 3)
  - Implements geometry-aware offset prediction head
- compute_local_normals: estimates local surface normals using k-NN PCA
  - Computes local surface geometry for each point
  - Used to regularize offset predictions
- geometry_aware_loss: regularization that encourages offsets to align with surface normals
  - Promotes learning of meaningful geometric structure
  - Combines with instance discrimination loss
- GeometryAwareTrainer: unified trainer for combined instance + geometry losses
  - Jointly optimizes contrastive loss (instance discrimination) and geometry loss
  - Provides train_step returning both losses for transparency
  - Includes get_offset_patterns method to extract and visualize learned offsets
- 5 new test cases: offset branch, geometry loss, PointCloudEncoder, geometry-aware training, offset pattern visualization
  - All tests pass and demonstrate loss convergence
  - Offset magnitudes in range [2.1, 5.7] showing reasonable learned structure

**Simplified/Stubbed**:
- Local normal estimation uses simplified k-NN PCA (k=8) instead of full surface reconstruction
  - Sufficient for regularization without heavy computation
- Geometry loss encourages alignment with estimated normals, not true orthogonality constraints
  - Simplified version still effectively regularizes learned offsets
- No visualization of offset fields yet (3D plots) - Pass 4 will add if needed
- No learned normal prediction - normals computed from point positions only
- Synthetic point clouds are still random (no realistic geometry)
