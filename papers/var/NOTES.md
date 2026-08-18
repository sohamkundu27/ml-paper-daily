# Visual AutoRegressive Modeling (VAR)

**Title:** Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction

**arXiv:** https://arxiv.org/abs/2404.02905

**Authors:** Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang

**Venue:** NeurIPS 2024 (Best Paper Award)

## Summary

Visual AutoRegressive Modeling (VAR) proposes a fundamentally different approach to autoregressive image generation compared to standard raster-scan token prediction. Instead of flattening a 2D image and predicting tokens left-to-right, VAR predicts images coarse-to-fine across multiple scales: given all tokens at coarser resolutions, it predicts the next higher-resolution token map. This preserves spatial structure and enables faster training and inference. The approach achieves state-of-the-art FID scores and outperforms diffusion transformers for the first time in visual generation, with clear power-law scaling laws similar to LLMs.

## Plan: 4 passes

**Pass 1 (Foundational):** Multi-scale tokenizer and basic single-layer VAR core. Implement a simple hierarchical tokenizer that downsamples images into coarse token maps at different scales, and a single transformer layer that can encode a lower-scale token map and predict the next-scale tokens. Focus on the data flow and core prediction mechanism with a minimal test on synthetic data.

**Pass 2 (Distinctive mechanism):** Full transformer stack and scale-conditional architecture. Stack multiple transformer layers, add proper scale embeddings and cumulative scale masking so the model predicts scales sequentially (coarse → fine). This is the core innovation that makes VAR work.

**Pass 3 (Real increment):** Training loop and loss computation. Add a simple training pipeline on toy images, compute cross-entropy loss for next-scale token prediction, and verify the model can learn to predict finer scales conditioned on coarser ones.

**Pass 4 (End-to-end demo):** Inference and generation. Implement sampling/generation from the trained model (starting from an empty coarse scale and iteratively sampling finer scales), run on synthetic/toy images, and document what was simplified.

## Implemented vs. simplified

### Pass 1
**Implemented:**
- Hierarchical tokenizer that downsamples an image into a sequence of lower-resolution token maps (e.g., 16→32→64 resolution token grids)
- Single transformer layer with positional embeddings that can encode coarse-scale tokens and output logits for next-scale tokens
- Basic test verifying that tokenizer produces correct hierarchical scales and that the transformer layer produces reasonable-shaped output

**Simplified/stubbed:**
- Tokenizer is extremely simple: just strided convolution / max pooling downsampling, no learned codebook (VAR uses a full VQ-VAE)
- Only a single transformer layer (no stacking yet; full model would have 12-24 layers)
- No scale embeddings or cumulative masking (added in pass 2)
- No training, no loss function, no actual gradient flow yet
- Generated tokens are random logits, not trained predictions

### Pass 2
**Implemented:**
- Full transformer stack: replaced single layer with 6 stacked transformer blocks for deeper modeling
- Scale embeddings: added learnable embeddings for each scale, concatenated to token representations so the model knows which scale each token belongs to
- Cumulative scale masking: implemented causal attention mask that enforces coarse-to-fine prediction order. Each token can attend to:
  - All tokens from coarser scales (earlier scales)
  - All tokens from its own scale (parallel generation within a scale)
  - NOT to tokens from finer scales (future scales being predicted)
- This realizes the core VAR innovation: sequential scale-conditional prediction instead of raster-scan token prediction

**Simplified/stubbed:**
- Tokenizer remains the same simple convolution-based design (no learned codebook)
- No actual training yet (pass 3 will add training loop)
- Inference/generation still stubbed (pass 4 will add sampling)

### Pass 3
**Implemented:**
- VARTrainer class: orchestrates training loop with optimizer (Adam) and learning rate scheduling
- Cross-entropy loss computation: loss is computed over predicted token logits vs. target token indices across all scales
- Target token generation: crude quantization scheme that converts continuous token features to discrete indices (simulating a VQ-VAE codebook)
- Training step: forward pass, loss computation, backward pass, and gradient updates via optimizer
- Validation loop: computes validation loss without gradient updates
- ToyImageDataset: synthetic dataset that generates random images for training verification
- DataLoader integration: supports batched training with PyTorch's DataLoader
- Comprehensive testing: verified that loss decreases during training, gradients flow correctly, and different learning rates affect convergence speed

**Simplified/stubbed:**
- Target tokens use a crude hash-based quantization instead of a learned VQ-VAE codebook. Real VAR uses a pre-trained codebook from a discrete VAE
- Toy dataset is entirely synthetic (random tensors), not real images. In practice, would use ImageNet or similar with a pre-trained tokenizer
- No learning rate scheduling or other optimizations (uses constant learning rate)
- No checkpoint saving or resuming
- Inference/generation still stubbed (pass 4 will add sampling)
