# xAR: Beyond Next-Token Prediction for Visual Generation

## Paper Details
- **Title**: Beyond Next-Token: Next-X Prediction for Autoregressive Visual Generation
- **arXiv**: https://arxiv.org/abs/2502.20388
- **Authors**: Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, Liang-Chieh Chen
- **Published**: February 2025

## Summary
Traditional autoregressive image generation models predict one token at a time, which is slow and suffers from exposure bias. This paper proposes xAR, which generalizes the notion of a "token" to an arbitrary entity X—patches, cells (grouped patches), subsamples (distant patch groups), scales (resolution levels), or whole images. By reformulating token prediction as continuous entity regression via flow-matching, xAR flexibly captures different spatial granularities and mitigates exposure bias through noisy context learning. On ImageNet-256, a 172M model outperforms 675M baselines while being 20x faster to generate.

## Plan: 4 Passes

### Pass 1: Entity Definitions and Basic AR Framework
Implement the foundational concept of flexible entities and tokenization:
- Define an `Entity` abstraction that can represent patches, cells, subsamples, or scales.
- Implement image-to-entity conversion: vectorize an image into entities at a given granularity.
- Implement entity-to-image reconstruction: reconstruct an image from predicted entities.
- Build a minimal AR model skeleton that autoregresses over entities (using a simple linear predictor, not flow-matching yet).
- Write a test that converts image → entities → predictions → reconstructed image.

### Pass 2: Flow-Matching and Continuous Regression
Extend Pass 1 with the distinctive mechanism:
- Implement the flow-matching objective for continuous entity values.
- Add the noisy context learning loss (training with corrupted/noisy entities as input).
- Integrate a small transformer encoder to predict the next entity conditioned on past entities.
- Test that the model can learn to denoise entities in a small toy scenario.

### Pass 3: Multi-Granularity and Exposure Bias Mitigation
Build toward a more realistic system:
- Extend the framework to support multiple entity granularities (e.g., 4×4 patches and 8×8 cells in parallel).
- Implement scheduled sampling or similar curriculum to transition from teacher-forced to noisy contexts during training.
- Add a simple backbone (small CNN or transformer) to encode/decode image features.
- Honest simplification: no full multi-scale coarse-to-fine generation; only demonstrate multiple granularities.

### Pass 4: Toy End-to-End Demo
Demonstrate the full pipeline on small synthetic data:
- Train xAR on a toy dataset (e.g., 32×32 images of simple shapes or MNIST variants).
- Generate images end-to-end, showing sample outputs.
- Report training curves (loss, reconstruction error).
- Final summary in NOTES.md: what worked, what was simplified, what insights emerged.

## Implemented vs. Simplified

### Pass 1 Complete
✓ `ImageToEntity` class: converts images ↔ patches with configurable patch size
✓ Entity abstraction: supports patch, cell (grouped patches), and subsample (non-local) entities
✓ Vectorization/devectorization: convert patches to fixed-dim entity vectors and back
✓ `SimpleARModel`: minimal 2-layer linear predictor that autoregressively generates entity sequences
✓ Full pipeline test: image → patches → entities → AR prediction → reconstruction
✓ 4 test functions validating each component

### Still Simplified / Not Yet Implemented
- **No flow-matching** (Pass 2): AR currently uses simple linear prediction, not continuous regression via flow-matching
- **No noisy context learning** (Pass 2): no training with corrupted entities to mitigate exposure bias
- **No actual training loop** (Pass 2-4): only inference/prediction skeleton
- **No backbone encoder/decoder** (Pass 3): entities are raw vectorized pixels
- **No multi-scale coarse-to-fine** (Pass 3): currently only single granularity at a time
- **No end-to-end demo** (Pass 4): no training on real or toy data yet
- **AR predictor architecture**: uses simple dense layers, not transformer (will upgrade in Pass 2-3)

The implementation demonstrates the core conceptual contribution: flexible entity definitions at different spatial granularities and an AR framework that can predict entity sequences. Flow-matching and exposure bias mitigation (the novel technical contributions) come next.
