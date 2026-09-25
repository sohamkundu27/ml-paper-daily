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

### Pass 2 Complete
✓ `TransformerEntityPredictor`: Multi-head transformer encoder predicting entity sequences
✓ `NoisyContextLearner`: End-to-end trainer implementing flow-matching and noisy context learning
✓ Flow-matching objective: Continuous MSE regression loss (model predicts clean entities from noisy input)
✓ Noisy context learning: Gaussian noise added to input during training to mitigate exposure bias
✓ Training loop: Full epoch-based training with loss tracking and inference
✓ 2 new tests validating denoising performance and loss convergence

### Pass 3 Complete
✓ `SimpleFeatureBackbone`: Lightweight CNN encoder/decoder for learned feature representations
✓ `ScheduledNoisyContextLearner`: Curriculum learning with noise schedule (noise increases over epochs)
✓ `MultiGranularityARTrainer`: Supports training on multiple entity granularities in parallel
✓ Gradient accumulation: Multiple granularities optimized jointly with shared optimizer
✓ Backbone integration: Optional learned feature backbone in multi-granularity trainer
✓ 4 new tests validating backbone, curriculum learning, and multi-granularity training

### Still Simplified / Not Yet Implemented
- **No multi-scale coarse-to-fine** (Pass 4): currently trains granularities independently, not hierarchically
- **No end-to-end demo on real data** (Pass 4): tested on synthetic/toy data only
- **Simplified curriculum**: noise schedule is linear; no exposure bias mitigation via teacher forcing
- **No acceleration tricks** (e.g., parallel decoding, skipping empty predictions)
- **Backbone is minimal**: 2-layer CNN instead of deeper architecture

The implementation now demonstrates: (1) flexible entity definitions at multiple granularities, (2) flow-matching + noisy context learning, and (3) multi-granularity training with learned representations. This foundation supports the coarse-to-fine generation strategy described in xAR.
