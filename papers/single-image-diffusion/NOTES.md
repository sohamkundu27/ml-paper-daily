# Efficient and Training-Free Single-Image Diffusion Models

## Metadata
- **arXiv**: https://arxiv.org/abs/2606.04299
- **Date**: June 3, 2026
- **Authors**: Haojun Qiu, Kiriakos N. Kutulakos, David B. Lindell
- **Affiliation**: University of Toronto, Vector Institute

## Summary

This paper addresses generative modeling of images using a single reference image without any neural network training. The core insight is that image patches at different scales can be organized into a finite dataset, and since patch dimensionality is small, a closed-form denoiser can be computed directly from the patch distribution. By computing the score function for noisy patches tractably, the method eliminates training while enabling high-quality, diverse image generation through diffusion sampling. Applications include unconditional generation, text-guided stylization, symmetrization, and image retargeting.

## Plan: 4 passes

### Pass 1: Patch extraction and basic image statistics
- Extract image patches at multiple scales using sliding windows
- Compute basic statistics (mean, covariance) of patches
- Implement simple patch-based nearest-neighbor lookup
- Runnable test: verify patches can be extracted and queried

### Pass 2: Closed-form denoiser
- Implement optimal closed-form denoiser using patch covariance
- Score function computation from patch distribution
- Handle dimension reduction for computational tractability
- Test: verify denoiser reduces noise on synthetic data

### Pass 3: Diffusion sampling loop
- Implement reverse diffusion process with schedule
- Integrate denoiser into iterative sampling
- Add guidance mechanism for conditional generation
- Test: basic unconditioned image generation from noise

### Pass 4: End-to-end demo on toy data
- Create small reference image
- Generate variations through diffusion
- Demonstrate multiple applications (generation, stylization)
- Final summary and honest assessment of simplifications

## Implemented vs. simplified

### After Pass 1

**Implemented:**
- `PatchExtractor` class that extracts patches from images at multiple scales using sliding windows
- Multi-scale support (default scales [1, 2], customizable)
- Computation of patch statistics: mean and covariance of the patch distribution
- L2-based nearest-neighbor lookup for patches
- Helper functions for creating synthetic test images and adding Gaussian noise
- Comprehensive test suite covering extraction, statistics, multi-scale handling, and noise robustness

**Simplified/Stubbed:**
- No dimensionality reduction yet (full covariance matrix used)
- No actual closed-form denoiser implementation
- No diffusion process or sampling
- Patches treated as grayscale (RGB images converted to grayscale)
- Simple sliding window extraction; no sophisticated patch matching algorithms

**Why these simplifications:** The goal of Pass 1 is to establish the data pipeline for patch extraction and basic statistics computation. The actual denoising and generation mechanisms are deferred to later passes. Grayscale handling keeps the implementation simple while preserving the core idea.
