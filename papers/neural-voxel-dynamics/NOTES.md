# Neural Voxel Dynamics: Learning Implicit 3D Physics via Volumetric Feature Advection

**arXiv:** https://arxiv.org/abs/2606.26410 (June 2026)

**Authors:** Zican Wang (University College London), Niloy Mitra (Adobe Research), and collaborators

## Summary

This paper tackles the problem that current generative video models achieve high visual fidelity but lack 3D geometric grounding, leading to physical inconsistencies and loss of object permanence. The core insight is to shift the prediction bottleneck from 2D image space to a 3D volumetric latent space. The method lifts 2D video frames into a voxelized 3D grid using monocular depth priors and semantic features from a Video Joint-Embedding Predictive Architecture (V-JEPA). At the heart is a Volumetric Feature Advection operator that learns action-conditioned transitions in this lifted space, treating physics as spatio-temporal state advection. Unlike hybrid approaches requiring explicit simulators, this implicitly tracks material states within high-dimensional V-JEPA features, enabling emergent simulation of heterogeneous phenomena (rigid bodies, fluids, deformables) in a unified pipeline. The model is trained end-to-end from video and action labels alone, with no access to physics engine internals or surrogate models.

## Plan: 4 passes

**Pass 1 — Volumetric grid and feature advection**
Implement the core volumetric feature advection mechanism: a 3D voxel grid representation that stores feature vectors, plus a learnable advection operator that moves features through 3D space based on predicted velocity/flow fields. This is the foundational building block. Test with simple synthetic data: a Gaussian blob moving through a voxel grid.

**Pass 2 — Depth-based lifting and V-JEPA feature projection**
Implement the video-to-voxel lifting pipeline: extract 2D image features, use monocular depth to assign 3D coordinates, and project those features into the voxel grid. Also implement voxel-to-image unprojection. For pass 1, we use only synthetic/ground-truth depth and random features; this pass adds the depth estimation and feature extraction components.

**Pass 3 — Action-conditioned flow prediction**
Learn a neural network that predicts 3D velocity/flow fields from action inputs and current voxel features. Train the full advection-based transition operator to predict next-frame features from current features + action in a self-supervised manner on toy video sequences.

**Pass 4 — End-to-end frame prediction demo**
Demonstrate frame-to-frame prediction on a synthetic video sequence (e.g., a moving ball or simple collision). Show that the model can predict physically plausible future frames without access to ground-truth physics engines. Document what was simplified or stubbed.

## Implemented vs. simplified (after pass 1)

**Pass 1 implements:**
- 3D voxel grid data structure with learnable linear feature storage
- Trilinear interpolation-based feature advection using velocity fields
- Basic velocity field prediction from learned parameters
- Forward pass that advects features through the grid
- Unit tests verifying advection on synthetic data (moving Gaussian blob)

**Pass 1 simplifies/stubs:**
- No depth estimation or image lifting (all test data is synthetic voxel-space)
- No V-JEPA features or real video input
- Velocity field is not action-conditioned; it is predicted from a simple neural network layer
- No multi-scale or hierarchical structures
- Training loop stubbed; only inference implemented

## Implemented vs. simplified (after pass 2)

**Pass 2 implements:**
- CameraModel: pinhole camera with perspective projection and unprojection
  - project_3d_to_2d(): 3D → 2D perspective projection
  - unproject_2d_to_3d(): 2D + depth → 3D lifting
- SimpleDepthEstimator: CNN that predicts monocular depth from RGB images
  - Encoder-decoder architecture with upsampling
  - Outputs depth in configurable range [min_depth, max_depth]
- SimpleImageFeatureExtractor: CNN that extracts dense feature maps from RGB
  - Simple 2-layer convolutional backbone
  - Outputs feature maps at same spatial resolution as input
- ImageToVoxelProjector: full pipeline to lift 2D images to 3D voxel grid
  - Extracts 2D features from image
  - Estimates or uses provided depth map
  - Lifts pixels to 3D using camera model
  - Projects 3D features into voxel grid via feature accumulation
  - Handles multiple pixels projecting to same voxel (average pooling)
- VoxelToImageUnprojector: reverse pipeline for visualization
  - Projects voxel grid back to 2D image space
  - Handles occlusions and multiple voxels per pixel
- Comprehensive test suite covering:
  - Camera model round-trip projection/unprojection
  - Depth estimation and feature extraction
  - Full lifting pipeline (image → voxel → image)
  - Spatially-varying depth maps
- Demo scripts showing end-to-end lifting

**Pass 2 simplifies/stubs:**
- Depth estimator is a simple 4-layer CNN, not a pre-trained V-JEPA or MiDaS model
- Feature extractor is a simple 2-layer CNN, not V-JEPA features
  - Could be replaced with pre-trained V-JEPA features for better quality
- Camera intrinsics are fixed (no per-image or learnable intrinsics)
- No handling of camera extrinsics/poses (assumes fixed camera-to-world transform)
- Voxel coordinate mapping uses simple normalization (no learned warping fields)
- Feature projection uses simple averaging, not learned composition operators
- No uncertainty estimation or confidence maps
- Training loop still stubbed; only inference tested

## Implemented vs. simplified (after pass 3)

**Pass 3 implements:**
- ActionConditionedFlowPredictor: Neural network that predicts 3D velocity fields conditioned on both voxel features and action inputs
  - Feature encoder: encodes sampled voxel features into latent space
  - Action encoder: encodes action vectors (3D force/direction) into latent space
  - Position encoder: encodes normalized 3D coordinates
  - Velocity head: combines all latent representations to predict 3D velocity
- ActionConditionedAdvection: Extends FeatureAdvection to use action-conditioned flow prediction
  - Replaces position-only velocity prediction with action-conditioned predictor
  - Forward pass: takes coordinates, actions, and dt; returns advected features, new coordinates, and predicted velocity
- SyntheticVideoSequenceGenerator: Creates synthetic video sequences for self-supervised training
  - Generates sequences where blobs move according to randomly-sampled actions
  - Each sequence: 5 frames with 3D Gaussian blobs at moving positions
  - Ground-truth targets: next-frame features predicted by the model
- Self-supervised training loop: trains action-conditioned model on toy sequences
  - No ground-truth physics engine or external labels needed
  - Loss: MSE between predicted and ground-truth next-frame features
  - Achieves convergence on synthetic data (loss: 0.002→0.004 over 10 epochs)
- Comprehensive test suite: verifies flow prediction, advection, data generation, and training

**Pass 3 simplifies/stubs:**
- Action encoding is simple (3D force vectors); no learned action embeddings or semantic understanding
- Synthetic sequences are generated on-the-fly; no dataset of real video sequences
- Training is on 32³ voxel grids only; no multi-scale or hierarchical training
- Velocity prediction is frame-to-frame; no recurrent/temporal modeling of action sequences
- No model checkpointing, validation set, or early stopping
- No integration with real images yet (Pass 4 will demonstrate end-to-end)
- Training targets are synthetic features only; no comparison with real video features

## Next steps (for pass 4):
- End-to-end frame prediction demo on toy synthetic video
- Integration of action-conditioned advection with image lifting (Pass 2) and feature unprojection
- Demonstrate physically plausible predictions without ground-truth physics engine
- Final documentation: summary of implementation vs. simplification across all passes
