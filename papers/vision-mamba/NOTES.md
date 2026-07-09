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

### After Pass 2:
- **Implemented:**
  * `SelectiveSSMBlock`: Core innovation—SSM with input-dependent selective gating. Each input is projected to hidden dimension, then multiplied by a learned sigmoid gate computed from the input. This gates which information flows through the state space. Supports bidirectional processing (forward/backward).
  * `BidirectionalSSMBlock`: Combines forward and backward selective SSM passes. Concatenates outputs and projects back to embedding dimension. Captures context from both directions.
  * Updated `VisionMambaBlock`: Now accepts `use_bidirectional` parameter to switch between original LinearSSMBlock (pass 1) and new BidirectionalSSMBlock (pass 2).
  * `VisionMambaPass2`: New model using bidirectional selective SSM blocks. Same architecture as Pass 1 but with selective gating and bidirectionality.
  * All components tested and working. Gradients flow through bidirectional blocks.
  * Pass 1 backward compatibility maintained—both Pass 1 and Pass 2 models run and produce different outputs as expected.

- **Simplified/Stubbed:**
  * Selective gating is input-dependent via learned sigmoid projection, not the full parameter modulation in the real Mamba paper. This is a simplified approximation.
  * Bidirectional pass still concatenates and projects; full Mamba uses separate parameters for fwd/bwd SSMs, not a simple concatenation.
  * Still no classification head; outputs raw patch representations.
  * No training loop or loss function yet.
  * SSM state is still fully exposed (hidden state computed sequentially); real hardware-optimized Mamba uses structured state representations and parallel scan algorithms (not implemented).

**Tests:** `test_vision_mamba_pass2.py` includes 10 tests covering:
  1. Selective SSM forward direction
  2. Selective SSM backward direction
  3. Gating effect (gates actually affect output)
  4. Bidirectional SSM block shape
  5. Bidirectional vs unidirectional differences
  6. Vision Mamba block with bidirectional option
  7. Pass 1 backward compatibility
  8. Pass 2 full model
  9. Gradient flow through Pass 2
  10. Pass 1 vs Pass 2 output differences
  
All tests pass. Gradients flow correctly. Model is ready for Pass 3 (classification head and training).

### After Pass 3:
- **Implemented:**
  * `VisionMambaPass3`: Full model with classification head. Uses bidirectional selective SSM blocks like Pass 2, adds global average pooling over patches, then a linear classification head (embed_dim -> num_classes).
  * Classification head: simple nn.Linear layer outputting logits for 10-way classification.
  * Improved SSM initialization: A matrix initialized near-identity (0.9 * I + small noise) rather than pure noise. This provides numerical stability and better gradient flow.
  * Training script `train_pass3.py`: Demonstrates full training pipeline on CIFAR-10 (small subset for demo). Includes:
    - DataLoader setup for train/test splits
    - Cross-entropy loss and Adam optimizer
    - Per-epoch metrics (loss, accuracy) on both train and test sets
    - Gradient clipping for stability
    - Configurable hyperparameters (batch_size, learning_rate, num_epochs, subset_size)
  * Model trains stably on toy data: loss decreases consistently over training steps (no NaN), gradients flow through entire model including classification head.

- **Simplified/Stubbed:**
  * Classification head is minimal (single linear layer) rather than more complex heads with MLPs or gating mechanisms.
  * No data augmentation in training (just normalize to [-1, 1]).
  * No learning rate scheduling, warmup, or other advanced training techniques.
  * Training is on a small subset of CIFAR-10 (500 samples) for speed. Full dataset training would improve accuracy.
  * Positional embeddings still use fixed sinusoidal encoding, not learned.
  * SSM state is still computed sequentially step-by-step (no parallel scan optimization from real Mamba).

**Tests:** `test_vision_mamba_pass3.py` includes 6 tests covering:
  1. Output shape (batch, num_classes)
  2. Different number of classes
  3. Gradient flow through classification head
  4. Stable training over multiple steps (no NaN, loss decreases)
  5. Model learns on toy data (loss decreases over epochs)
  6. Pass 2 backward compatibility (Pass 2 still works alongside Pass 3)

All tests pass. Model trains stably without NaN loss. Backward compatibility maintained.
