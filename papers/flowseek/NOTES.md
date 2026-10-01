# FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases

**arXiv:** https://arxiv.org/abs/2509.05297  
**Authors:** Matteo Poggi, Fabio Tosi  
**Submitted:** September 5, 2025

## Summary

This paper proposes a lightweight approach to optical flow estimation that leverages depth foundation models (specifically Depth Anything V2) and classical motion parameterization techniques. Rather than building from scratch, FlowSeek combines existing depth priors with motion basis functions to create an efficient flow network that achieves state-of-the-art cross-dataset generalization while requiring significantly lower computational resources (8x lower GPU memory than competing methods).

## Plan: 4 passes

**Pass 1 (Foundational):** Implement a basic cost volume and correlation pyramid for optical flow estimation. Build the core matching mechanism that compares image patches and computes correlation statistics.

**Pass 2 (Mechanism):** Integrate depth foundation model priors. Load a pretrained depth encoder and use its features to guide optical flow refinement.

**Pass 3 (Incremental):** Implement motion bases parameterization. Use low-dimensional motion models (translation, affine, homography) to constrain and regularize flow estimates.

**Pass 4 (Demo):** End-to-end inference on a toy image pair. Compare estimated flow against ground truth or qualitatively visualize the motion field. Record what was simplified.

## Implemented vs. simplified

*After Pass 1:*

**Implemented:**
- `CorrelationVolume`: Computes correlation between feature maps across a search range
- `CorrelationPyramid`: Builds multi-scale pyramid (1-4 levels) with downsampling
- `FlowEstimator`: Takes two images, builds pyramid, finds best matches via argmax
- Feature extraction: Simple feature normalization (zero-mean, unit variance)
- Upsampling: Bilinear interpolation to original resolution
- Comprehensive test suite: dimension checks, pyramid levels, batch processing, synthetic motion
- All code runs without errors; tests verify correctness

**Simplified/Stubbed:**
- No depth model integration (Depth Anything V2 will be added in Pass 2)
- No motion bases parameterization (classical motion models added in Pass 3)
- Uses simple L2 distance matching, not learned features
- No iterative coarse-to-fine refinement
- No training loop (inference-only)
- No sophisticated warping or deformation
- Flow estimation is coarse (from single coarsest pyramid level)

*After Pass 2:*

**Implemented:**
- `DepthEncoder`: Extracts depth features using Depth Anything V2 (with fallback gradient-based proxy)
  - Attempts to load transformers-based depth model if available
  - Falls back to edge-based depth proxy if transformers unavailable
  - Normalizes depth features to [-1, 1] range for stable correlation
- Depth-guided feature extraction in `CorrelationPyramid`:
  - Concatenates normalized RGB features with depth features (3+1 channels)
  - Passes combined features to correlation volume computation
  - Enables depth-aware optical flow matching
- `use_depth` parameter throughout pipeline:
  - `FlowEstimator` accepts `use_depth` flag
  - `estimate_optical_flow()` passes through `use_depth` parameter
  - Can toggle depth guidance on/off for comparison
- Comprehensive tests for depth integration:
  - Depth encoder shape and value validation
  - Feature channel concatenation verification
  - Flow estimation with/without depth
  - Consistency on structured motion (translation)
- All code runs without errors; depth guidance is optional and backward-compatible

**Simplified/Stubbed:**
- Depth model is lightweight fallback (gradient-based) when transformers unavailable
  - Would use Depth Anything V2 with full transformers integration in production
  - Current fallback demonstrates the mechanism without heavy dependencies
- No learned depth encoder fine-tuning (uses frozen pretrained weights)
- Motion bases parameterization not yet implemented (Pass 3)
- No iterative refinement using depth confidence maps
- No learned combination weights for RGB-depth fusion (simple concatenation)
- Flow estimation still coarse (from coarsest pyramid level)
