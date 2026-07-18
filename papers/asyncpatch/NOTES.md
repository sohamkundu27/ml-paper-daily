# AsyncPatch Diffusion

**Title:** AsyncPatch Diffusion: Spatially-Flexible Image Generation

**arXiv:** https://arxiv.org/pdf/2606.07079

**Authors:** (AsyncPatch Diffusion paper, June 2026)

## Summary

AsyncPatch Diffusion introduces a joint-diffusion framework that decouples the noise schedule across spatial dimensions. Instead of applying the same noise level to all pixels or tokens simultaneously, different spatial regions can follow independent denoising trajectories. This enables efficient spatially adaptive generation, inpainting without fine-tuning, and flexible inference strategies. The method maintains generation quality comparable to standard diffusion while unlocking new capabilities like uncertainty-guided acceleration and autoregressive spatial sampling.

## Plan: 4 passes

**Pass 1:** Core noise-level scheduler. Implement a mechanism to assign independent noise levels to spatial patches and verify that forward diffusion with heterogeneous noise follows a valid joint process. Small synthetic test on a toy 8×8 or 16×16 grid.

**Pass 2:** Full joint-diffusion forward process. Extend to realistic shapes (batched images), implement the forward process that corrupts pixels independently based on per-patch noise schedules, and verify theoretical properties (variance, signal/noise ratios).

**Pass 3:** Reverse process (sampling). Implement a basic denoiser that learns to reverse the heterogeneous diffusion trajectories. Start with a small UNet or simple ConvNet, train on toy data.

**Pass 4:** End-to-end demo with inpainting. Show that a trained model can perform inpainting by fixing known regions and sampling only unknown regions. Demonstrate on small synthetic images and provide honest summary of what still works vs. what was simplified.

## Implemented vs. simplified

### Pass 1

**Implemented:**
- Noise-level assignment scheduler for spatially heterogeneous diffusion
- Forward diffusion with per-patch noise schedules
- Verification that joint process maintains valid Gaussian properties
- Small test case on toy 8×8 grids

**Simplified/stubbed:**
- No neural network, no sampling or reverse process yet
- Toy data only (synthetic grid patterns)
- No inpainting logic

### Pass 2

**Implemented:**
- Full joint-diffusion forward process with realistic image sizes (32×32, 64×64)
- Signal-to-noise ratio (SNR) computation for each patch: SNR(t) = log(α̅_t / (1 - α̅_t))
- Comprehensive theoretical property verification:
  - Variance preservation across all patches (α_t^2 + σ_t^2 = 1)
  - SNR monotonicity (SNR decreases as t increases)
  - Patch statistics collection (alpha, sigma, SNR, variance per patch)
- Large-scale batch testing with heterogeneous timestep assignments
- Joint diffusion properties analysis showing valid signal/noise ratio ranges
- Extended test suite:
  - Realistic image size tests (32×32 RGB, 64×64 RGB)
  - SNR monotonicity verification
  - Large batch heterogeneous processing (8 images × 64×64)
  - Forward process structural validation

**Simplified/stubbed:**
- No reverse process (denoising) yet; this is pass 3
- No training or sampling from the model yet
- Inpainting demonstration deferred to pass 4
- No visualization of corruption levels (analysis only)

### Pass 3

**Implemented:**
- SimpleDenoiser: Lightweight ConvNet that conditions on per-patch timesteps
  - Takes corrupted image + normalized timestep map as input
  - Predicts noise for denoising
  - 3-layer convolutional architecture: 16 hidden channels
- DenoisingTrainer: Training framework for heterogeneous denoising
  - Compute MSE loss between predicted and true noise
  - Single train_step: forward corruption, denoiser prediction, backward pass
  - Eval mode for validation loss computation
  - Iterative sampling (reverse process):
    - Starts from corrupted image
    - Iteratively predicts and removes noise over fixed steps
    - Simple linear timestep schedule with 0.1 noise scaling
- Comprehensive test suite for Pass 3:
  - Denoiser forward pass (correctness and shape validation)
  - Trainer loss computation on heterogeneous timesteps
  - Single training step with gradient descent verification
  - Mini training loop (3 steps) on toy 16×16 images
  - Iterative sampling starting from heavily corrupted images

**Simplified/stubbed:**
- Denoiser architecture is minimal (no skip connections, no attention)
- Sampling uses simple linear timestep reduction and fixed noise scaling (not full reverse diffusion)
- No guidance mechanism (e.g., classifier guidance)
- No adaptive denoising schedules per patch
- Inpainting (fixing known regions) deferred to pass 4
- No model persistence/checkpointing
- Single-step sampling improvement rather than full ancestral sampling

### Pass 4 (this session)

**Implemented:**
- Inpainting method on DenoisingTrainer:
  - Takes an image, an inpainting mask (1 = known, 0 = unknown), and timesteps
  - Applies forward diffusion only to unknown regions (high noise levels)
  - Iteratively denoises while keeping known regions fixed (clamped to original values)
  - Blends predicted denoising updates with fixed known regions at each step
- End-to-end inpainting demonstration:
  - Trains denoiser on toy checkerboard patterns (4 patterns, 16×16, grayscale)
  - Tests inpainting on a uniform gray image with edge-only mask (corners known, center unknown)
  - Demonstrates that unknown regions are filled with realistic values while known regions remain unchanged
  - Shows heterogeneous timestep assignment: high (999) for unknown patches, moderate (100) for known
- Comprehensive Pass 4 test suite:
  - Basic inpainting with right-half unknown mask
  - Center-region inpainting with 8×8 center mask
  - Full end-to-end demo with 5 training epochs and 5 inpainting steps

**Simplified/stubbed:**
- Inpainting does not use learned priors specific to the content (random initialization for unknown regions)
- No content-aware guidance or boundary smoothing
- Denoiser architecture remains simple (no advanced architectures like UNet with skip connections)
- No probabilistic sampling from the reverse process (deterministic noise prediction and removal)
- Timestep assignment for inpainting is manual, not learned
- No comparison to ground truth or evaluation metrics beyond shape/value checking
- Limited to small synthetic toy images (16×16)

## Summary

AsyncPatch Diffusion successfully demonstrates spatially-flexible diffusion through heterogeneous per-patch noise schedules. The 4-pass implementation shows:

1. **Pass 1:** Valid heterogeneous noise assignment and forward diffusion
2. **Pass 2:** Realistic image handling with verified SNR properties
3. **Pass 3:** Trainable reverse process with iterative sampling
4. **Pass 4:** Proof-of-concept inpainting where regions with different noise levels are handled independently

The core mechanism is functional: different spatial regions can follow independent denoising trajectories, enabling inpainting by selectively applying high noise to unknown regions and high signal retention to known regions. This decoupling is the key insight that makes AsyncPatch Diffusion distinctive—traditional diffusion requires a global noise schedule, whereas this approach allows per-patch flexibility.
