# Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

**arXiv:** [2407.01392](https://arxiv.org/abs/2407.01392)

**Authors:** Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, Vincent Sitzmann

**Published:** July 2024

## Summary

This paper introduces Diffusion Forcing, a training paradigm that combines next-token prediction with full-sequence diffusion. The core idea is to train a model to denoise tokens that have been corrupted with independent, per-token noise levels. Rather than fully diffusing all previous tokens (as in standard diffusion) or conditioning only on immediate history (as in autoregressive models), Diffusion Forcing allows flexible denoising with variable noise per timestep. This enables variable-length generation, improved rollout horizons, and novel sampling schemes that leverage the strengths of both next-token and full-sequence approaches.

## Plan: 4 passes

**Pass 1 — Foundational diffusion mechanics for sequences**
- Implement per-token noise scheduling and forward diffusion process
- Build a simple denoiser network (tiny MLP that predicts clean tokens from noisy tokens)
- Test on toy synthetic sequences (e.g., repeating patterns or simple arithmetic sequences)
- Runnable, tested, no training yet

**Pass 2 — Training loop and loss**
- Implement training loop with basic loss (L2 between predicted clean tokens and ground truth)
- Train the denoiser on simple sequence data (random or synthetic patterns)
- Validate that loss decreases and denoiser learns to denoise

**Pass 3 — Inference and generation**
- Implement sampling/inference: start with fully noisy sequence, iteratively denoise
- Add masking to show "keep past, denoise future" capability
- Generate variable-length sequences beyond training horizon

**Pass 4 — End-to-end demo**
- Demo on a simple task (e.g., predicting next tokens in a numeric sequence or simple text)
- Compare rollout quality against naive autoregressive baseline
- Write honest summary of what is working and what was simplified/stubbed

## Implemented vs. simplified

**Pass 1 Implementation:**
- ✅ Per-token noise scheduling (cosine annealing)
- ✅ Forward diffusion: q(x_t | x_0) with reparameterization
- ✅ Simple 2-layer MLP denoiser
- ✅ Minimal tests on synthetic data
- ⏸️ No training (stubbed for pass 2)
- ⏸️ No inference/sampling (stubbed for pass 3)
- ⏸️ Simplified to 1D token sequences for clarity
