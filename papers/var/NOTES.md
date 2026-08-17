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
