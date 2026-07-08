# Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model

**arXiv:** https://arxiv.org/abs/2401.09417

**Authors:** Lianghui Zhu, Bencheng Liao, Qian Zhang, Xiaoyi Dong, Wenyu Liu, Yuru Wang

## Summary

Vision Mamba proposes using state space models (specifically, bidirectional Mamba layers) as an alternative to self-attention for visual representation learning. The key insight is that images can be treated as sequences of patches with position embeddings, and then processed through efficient Mamba blocks instead of transformer attention. This achieves competitive performance on ImageNet classification while being more computationally efficient than Vision Transformers.

## Plan: 4 passes

**Pass 1 (foundational):** Implement image patching + position embeddings + a simplified unidirectional SSM block. Create a minimal model that can process an image and output patch representations. This includes:
- Image-to-patches conversion (non-overlapping)
- Sinusoidal position embedding
- A basic state space layer (simplified SSM without selective gating)
- Forward pass on a single image

**Pass 2 (core mechanism):** Implement bidirectional SSM blocks with selective gating. This makes the model closer to the paper's full method by adding:
- Proper selective gating in SSM layers (the "selective" part of Mamba)
- Bidirectional processing (forward + backward passes)
- Stacked SSM blocks

**Pass 3 (real increment):** Add a simple image classification head and classification loss. Enable training on toy CIFAR-10 data and demonstrate that the model can learn.

**Pass 4 (end-to-end demo):** Train on a small subset of CIFAR-10 and evaluate. Add a summary of what works, what was simplified, and honest assessment vs. the full paper.

## Implemented vs. Simplified (to be updated after each pass)

### After Pass 1:
- **Implemented:**
  * `ImagePatcher`: Converts images (batch, C, H, W) to patch embeddings using Conv2d with kernel=patch_size, stride=patch_size
  * `PositionalEmbedding`: Pre-computed sinusoidal positional embeddings added to patches (no learnable PE)
  * `LinearSSMBlock`: Simplified unidirectional SSM with A (state transition), B (input), C (output), and D (skip) parameters. Processes sequence step-by-step maintaining hidden state h_t = A @ h_{t-1} + B @ x_t, y_t = C @ h_t + D @ x_t
  * `VisionMambaBlock`: Wraps SSM with LayerNorm and residual connection
  * `VisionMambaPass1`: Full model with patcher → position embeddings → stacked SSM blocks → final LayerNorm. Outputs patch representations (batch, num_patches, embed_dim)
  * Forward pass tested on random (4, 3, 32, 32) images producing (4, 64, 64) outputs
  * Gradients flow correctly through entire model

- **Simplified/Stubbed:**
  * SSM lacks selective gating (Mamba's core innovation). This is a generic linear SSM, not selective.
  * Only unidirectional processing. The paper uses bidirectional SSMs; this version processes left-to-right only.
  * No learned positional embeddings (sinusoidal fixed encoding only)
  * No classification head; outputs raw patch representations
  * No training loop, loss function, or optimizer
  * Uses full attention-free forward pass but doesn't leverage hardware efficiency benefits yet

**Test:** `test_vision_mamba_pass1.py` includes 7 tests:
  1. Image patcher shape correctness
  2. Positional embedding addition
  3. SSM block output shape
  4. Vision Mamba block with residual
  5. Full model output shape
  6. Gradient flow through entire model
  7. Deterministic/consistent output
  
All tests pass. Model is runnable and gradients flow.
