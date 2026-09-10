# Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass

**arXiv**: https://arxiv.org/abs/2501.13928

**Published**: January 2025 (CVPR 2025)

**Authors**: Yushuang Wu, Tianwei Lin, Zilong Dong, Liangzeng Li, Shunbo Zhou, Rui Zhao, Chi Zhang, Zhen Lei, Yu Liu

## Summary

Fast3R extends DUSt3R's pairwise approach to multi-view 3D reconstruction by processing all images in parallel using a Transformer architecture. Instead of the typical sequential pair-wise alignment followed by costly global pose refinement, Fast3R predicts relative camera poses between all image pairs simultaneously in a single forward pass, enabling scalable reconstruction of 1000+ images with improved speed and accuracy. The core insight is that a Transformer can learn to aggregate geometric information across all views and resolve relative pose ambiguities more effectively than iterative refinement procedures.

## Plan: 4 passes

**Pass 1 — Foundational pose embedding and pairwise estimation**
- Implement image feature extraction (DINO ViT backbone)
- Implement pose representation (6D rotation representation + translation)
- Build pairwise relative pose prediction head: given two image embeddings, predict 3D relative pose
- Test on a small synthetic dataset of camera poses
- Simple test: verify pose predictions are reasonable tensors with correct shapes

**Pass 2 — Multi-view Transformer aggregation**
- Build a Transformer encoder that ingests all image pairs and their initial pose estimates
- Implement attention mechanism to learn interactions between pose predictions across image pairs
- Refine pairwise pose estimates through Transformer passes
- Test with 3-5 images to verify multi-view aggregation works

**Pass 3 — Pose refinement and consistency**
- Implement pose consistency loss: enforce transitivity (if A→B and B→C, then A should match A→C)
- Add simple ICP-style pose refinement for edge pairs
- Demo with 10-20 synthetic views

**Pass 4 — End-to-end demo on toy data**
- Build simple data loader for synthetic multi-view scenes (e.g., rotating object around fixed camera)
- Reconstruct 3D points from predicted poses using triangulation
- Evaluate reprojection error and relative pose accuracy
- Honest summary of what was simplified vs. the full paper

## Implemented vs. simplified

**Pass 1 — What's implemented:**
- 6D rotation representation (continuous, avoiding gimbal lock) with Gram-Schmidt orthogonalization to convert to 3x3 rotation matrices
- PoseHead: a simple MLP that predicts 6D relative pose from concatenated image embeddings (3D rotation in 6D form; translation stubbed as zero)
- `PairwisePoseModel`: ingests image embeddings and pair indices, outputs both 6D pose vectors and 3x4 transformation matrices
- Comprehensive tests verifying: rotation matrix orthogonality, pose matrix validity, gradient flow, batched processing

**Pass 1 — What's simplified or stubbed:**
- **No image feature extraction**: we assume embeddings are provided (no DINO ViT backbone yet)
- **No translation learning**: the model predicts 6D rotation but translation is currently set to zero in `pose_6d_to_matrix`; Pass 2 will extend this to predict full 6D+3D poses
- **No multi-view consistency**: each pair is predicted independently; Pass 2 will add a Transformer to aggregate across all pairs
- **No real camera data**: tests use random embeddings and random pose ground truth for validation only
- **No pose loss or refinement**: just verify shapes and mathematical properties
