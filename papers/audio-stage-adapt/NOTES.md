# Stage-Adaptive Audio Diffusion Modeling

**Title:** Stage-adaptive audio diffusion modeling

**arXiv:** https://arxiv.org/abs/2605.04547

**Authors:** Xuanhao Zhang, Chang Li

**Date:** May 6, 2026

## Summary

The paper addresses the computational inefficiency of training audio diffusion models by observing that the optimal balance between semantic acquisition and perceptual refinement changes throughout training. Early training should prioritize learning coarse structure aligned with conditioning signals, while later stages should focus on fine-detail temporal consistency and fidelity. The authors propose adapting the training objective weights dynamically based on training progress, reducing computational cost while maintaining or improving audio quality for generation and restoration tasks.

## Plan: 4 passes

**Pass 1** — Foundational diffusion model for audio
- Implement a basic diffusion model backbone for 1D audio waveforms
- Use a simple MLP or small transformer encoder conditioned on noise schedule
- Include forward diffusion (adding noise) and basic reverse process (denoising)
- Test on toy audio data with shape assertions; no conditioning yet

**Pass 2** — Stage-adaptive scheduler (the key innovation)
- Implement loss weighting that adapts across stages (e.g., semantic vs. perceptual)
- Create a scheduler that modulates loss weights based on training step/epoch
- Apply to semantic and detail-refinement objectives separately
- Verify weights change as expected over training progress

**Pass 3** — Conditioning mechanism for audio generation
- Add simple conditioning input (text embedding placeholder or one-hot class label)
- Integrate conditioning into diffusion model via concatenation or cross-attention stub
- Enable basic text-to-audio or class-conditional audio generation
- Honest simplification: use pre-made embeddings, no actual text encoder

**Pass 4** — End-to-end demo on toy data
- Demonstrate generation loop (sampling from noise through reverse diffusion)
- Show that stage-adaptive weighting produces better results than static weights
- Small synthetic dataset (e.g., generated sine waves or simple patterns)
- Summary: document what actually works and what was simplified

## Implemented vs. Simplified

### Pass 1 Implementation

**What works:**
- Core GaussianDiffusion with linear beta schedule and standard forward/reverse process
- AudioDiffusionModel: simple MLP-based denoising network that takes noisy audio + timestep embedding
- Sinusoidal positional embeddings for timestep conditioning
- Forward diffusion (q_sample): correct noise addition with alpha_bar schedule
- Basic reverse sampling: iterative denoising from noise to sample
- Full training loop with MSE loss (denoising objective)

**What was simplified:**
- No variance scheduling in reverse process: uses fixed beta schedule instead of learned or optimal variance
- No text/class conditioning yet (Pass 3 will add this)
- Model is small MLP, not transformer (keeps it fast and testable)
- No audio-specific preprocessing: treats audio as 1D vectors, not spectrograms
- Sampling uses uniform timestep subsampling (every nth step) instead of learned schedule
- No stage-adaptive weighting yet (Pass 2 will add this)

**Test coverage:**
- Forward diffusion correctness
- Model forward pass shapes and NaN/inf checks
- Training step (backward pass works, loss decreases)
- Sampling produces valid outputs
- Model determinism in eval mode

### Pass 2 Implementation

**What works:**
- StageAdaptiveScheduler: computes semantic vs. perceptual loss weights based on training progress
  - Linear strategy: weights transition linearly from semantic→perceptual over training
  - Exponential and cosine strategies also available for non-linear transitions
- get_timestep_mask: partitions timesteps into semantic (high-noise, t > mid) and perceptual (low-noise, t < mid)
- stage_adaptive_loss: applies adaptive weights to separate loss components
  - Early training: emphasizes semantic loss (coarse structure learning)
  - Late training: emphasizes perceptual loss (fine detail refinement)
- Verified weights change correctly: early steps have semantic_w ≈ 1.0, late steps have perceptual_w ≈ 1.0
- Compatible with existing training loop: drop-in replacement for MSE loss

**What was simplified:**
- Timestep partitioning is simple (binary split at midpoint) rather than learned or task-aware
- No ablation on different strategy curves (linear suffices for demo)
- Stage weighting assumes all high-noise timesteps are "semantic" and all low-noise are "perceptual"
  - Real implementation might weight more flexibly based on frequency content
- No integration with conditioning yet (Pass 3 will add conditioning)

### Pass 3 Implementation

**What works:**
- Conditional AudioDiffusionModel: extended to accept optional conditioning vector via cond parameter
  - Model concatenates conditioning to audio+time embeddings before MLP
  - Supports both conditional (cond_dim specified) and unconditional (cond_dim=None) modes
  - Backward compatible: existing unconditional code works unchanged
- Class-based conditioning via make_class_embedding function
  - Converts class indices to one-hot encoding, then projects to cond_dim
  - Deterministic projection ensures same class produces same embedding
  - Supports single class or batch of class indices
- Conditional sampling: diffusion.sample() now accepts cond parameter
  - Passes conditioning through all reverse diffusion steps
  - Enables class-conditional generation during inference
- Full training loop with conditioning:
  - Stage-adaptive loss works seamlessly with conditional models
  - Tested on small synthetic data (batch_size=4, audio_dim=128)
  - Successfully trains class-conditional denoising network

**Test coverage:**
- Class embedding determinism and shape correctness
- Conditional model forward pass with and without conditioning
- Unconditional model backward compatibility
- Conditional sampling from noise to clean audio
- Training loop with class conditioning and stage-adaptive loss
- Class consistency (same class produces deterministic results, different classes differ)

**What was simplified:**
- No actual text encoder: uses simple class label embeddings via one-hot projection
  - Real implementation would embed natural language descriptions of audio
  - Here, we use discrete class indices (0-4 for demo) as conditioning
- No cross-attention: conditioning is concatenated to input
  - Real implementation might use cross-attention for more flexible interaction
  - Concatenation is simpler and works for this scope
- No classifier-free guidance: unconditional paths not implemented
  - Real implementation uses guidance scale to control conditioning strength
  - Omitted to keep scope focused on basic conditioning integration
- No explicit conditioning during training with different probability (CFG training)
  - Would randomly drop conditioning during training to improve unconditional performance
  - Not needed for this demo
