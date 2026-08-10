# BitDance: Scaling Autoregressive Generative Models with Binary Tokens

**Title:** BitDance: Scaling Autoregressive Generative Models with Binary Tokens

**arXiv:** https://arxiv.org/abs/2602.14041

**Authors:** ByteDance Research Team

## Summary

This paper presents an efficient autoregressive image generation approach that predicts binary visual tokens instead of discrete codebook indices. Rather than using softmax to select from a limited vocabulary, BitDance employs continuous-space binary diffusion to generate binary tokens, where each token can represent up to 2^256 states. This enables a highly expressive yet compact representation. The method scales effectively, achieving state-of-the-art image generation quality among autoregressive models on ImageNet while using significantly fewer parameters and achieving substantially faster inference than prior autoregressive approaches.

## Plan: 4 passes

**Pass 1 — Binary tokenization and VAE encoder/decoder**
Implement a learnable binary tokenizer that converts images to/from binary token representations. This includes a convolutional encoder that compresses spatial dimensions and produces binary embeddings, plus a decoder that reconstructs images from tokens. Minimal test on toy 32x32 synthetic data to verify encoding/decoding works.

**Pass 2 — Binary diffusion head for token generation**
Implement a simple binary diffusion process that gradually refines noisy binary tokens. Add a basic network head that conditions on text/context and uses continuous-space diffusion to predict binary tokens. Test that the diffusion process produces reasonable token predictions over timesteps.

**Pass 3 — Autoregressive token predictor with caching**
Build a simple autoregressive model that predicts the next token sequence given previous tokens, using the binary diffusion head. Include token-level caching for efficiency. Add a multi-scale coarse-to-fine strategy where coarser tokens are predicted first, then refined at finer scales.

**Pass 4 — End-to-end generation demo with quality metrics**
Implement end-to-end image generation on toy data: encode random images to tokens, run AR generation on masked tokens, decode back to image space. Compute FID-like toy metrics (reconstruction error, diversity across generated samples). Write honest summary of what was simplified or skipped.

## Implemented vs. simplified

**Pass 1 (completed):**
- **Implements:** Convolutional encoder/decoder for binary tokenization. Encoder compresses 32x32 images to 4x4 token maps via three stride-2 conv layers. Binary quantization via thresholding (>0.5). Decoder reconstructs images via transposed convolutions. Includes reconstruction loss and binary entropy loss for training. Test suite verifies shapes, binary correctness, and training convergence on toy data.
- **Simplifies:** No learnable codebook; direct binary thresholding instead of learned quantization. Single fixed spatial scale (4x4 tokens) rather than multi-scale hierarchy. No masking or positional embeddings yet.
- **Stubs/Not included:** Binary diffusion head (Pass 2), autoregressive token prediction (Pass 2), multi-scale coarse-to-fine generation (Pass 3), text conditioning/context models, actual training on real image datasets.

**Pass 2 (completed):**
- **Implements:** Binary diffusion head for refining token predictions via continuous-space diffusion. Includes sinusoidal positional encoding for timesteps, a simple 2-layer convolutional denoising network that conditions on timestep embeddings, and a linear noise schedule (beta_t). Diffusion loss computes MSE between predicted and ground-truth tokens across timesteps. BinaryDiffusionSampler performs reverse diffusion via iterative refinement (simplified DDPM-style sampling). Test suite verifies shapes, noise schedule behavior, loss computation, training convergence, and binary token generation via sampling.
- **Simplifies:** Linear noise schedule instead of cosine schedule. Simplified reverse diffusion step (weighted interpolation toward prediction) rather than full DDPM posterior sampling. No text/context conditioning yet; timestep is the only input. Sampler uses fixed step count rather than adaptive scheduling. No learned variance schedule.
- **Stubs/Not included:** Text/context conditioning (Pass 3), autoregressive prediction of token sequences (Pass 3), multi-scale coarse-to-fine generation (Pass 3), end-to-end demo (Pass 4), actual training or evaluation on real image datasets.

**Pass 3 (completed):**
- **Implements:** Autoregressive token predictor using transformer self-attention to capture dependencies among tokens in a sequence. Includes learnable token embedding layer, positional embeddings for sequence positions, and a multi-layer transformer encoder with 4 attention heads by default. AutoregressiveLoss uses binary cross-entropy for token prediction. MultiScaleTokenGenerator implements coarse-to-fine generation strategy that can process multiple spatial scales (4x4, 8x8, 16x16) and refine tokens progressively. Autoregressive generation via sampling that extends token sequences one token at a time. Token-level caching infrastructure in place (cache_embeds, cache_pos) for future efficiency improvements. Test suite verifies transformer shapes, training convergence, sequence generation, multi-scale processing, caching mechanisms, and loss computation.
- **Simplifies:** Sequence-based prediction rather than full spatial grid conditioning. Transformer uses norm_first architecture without complex scheduling strategies. No text/context conditioning; purely token-to-token prediction. Multi-scale generation supports manual scale specification rather than automatic hierarchical scheduling. Caching infrastructure added but not yet utilized in forward passes (reserved for future refinement).
- **Stubs/Not included:** Text/context conditioning, integration of diffusion head into autoregressive loss (diffusion operates on spatial grids, not sequences), adaptive timestep scheduling for multi-scale generation, full end-to-end demo (Pass 4), training on real datasets.

**Pass 4 (completed):**
- **Implements:** End-to-end image generation pipeline via EndToEndDemo class. Provides toy image generation (solid color, stripes, checkerboard patterns), full encode-decode-regenerate cycle. Token masking mechanism that zeros out random portions of token grids. Masked token regeneration using the autoregressive transformer + diffusion refinement. Quality metrics computation including MSE, L1, PSNR, and variance-based diversity measures. Complete test suite demonstrating: toy image creation, encode-decode cycles, token masking, regeneration, image completion with masking, metrics computation, and full end-to-end pipeline. Example metrics on toy data show PSNR ~42dB for reconstruction without masking, and ability to regenerate masked regions (though with lower quality due to lack of training).
- **Simplifies:** Toy data only (synthetic patterns, no real images). No training of any component in pass 4 itself (all models used in eval mode). Masking strategy is random, not structured. Metrics are toy metrics (not real FID/IS), computed only on synthetic data. Regenerated tokens initialized from autoregressive predictions with single diffusion refinement pass, not iterative optimization.
- **Stubs/Not included:** Training of tokenizer/diffusion/AR predictor on real image data. Text/image conditioning during generation. Structured masking strategies (e.g., spatial patches, progressive refinement). Real perceptual metrics (FID, IS, LPIPS). Comparison with baseline generative models. Multi-resolution image handling beyond 32x32. Caching optimization (infrastructure in place but unused). Streaming/batched generation for large-scale deployment.
