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

### Pass 1 (this session)

**Implemented:**
- Noise-level assignment scheduler for spatially heterogeneous diffusion
- Forward diffusion with per-patch noise schedules
- Verification that joint process maintains valid Gaussian properties
- Small test case on toy 8×8 grids

**Simplified/stubbed:**
- No neural network, no sampling or reverse process yet
- Toy data only (synthetic grid patterns)
- No inpainting logic
- Theoretical guarantees left to pass 2
