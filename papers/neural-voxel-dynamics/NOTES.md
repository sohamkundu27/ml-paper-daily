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
