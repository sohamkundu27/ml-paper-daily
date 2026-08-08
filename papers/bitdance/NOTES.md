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
