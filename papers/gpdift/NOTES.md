# Generative Pre-trained Autoregressive Diffusion Transformer

**Title:** Generative Pre-trained Autoregressive Diffusion Transformer

**arXiv:** https://arxiv.org/abs/2505.07344

**Authors:** Yuan Zhang, Jiacheng Jiang, Guoqing Ma, Zhiying Lu, Haoyang Huang, Jianlong Yuan, Nan Duan, Daxin Jiang (Microsoft Research)

## Summary

GPDiT unifies autoregressive and diffusion modeling by autoregressively predicting future latent frames using a diffusion loss in continuous space. Rather than predicting discrete tokens or directly predicting next frames, it applies diffusion denoising to each frame position, enabling natural modeling of motion dynamics and semantic consistency. The paper introduces a lightweight causal attention variant and a parameter-free rotation-based time-conditioning mechanism that improve efficiency.

## Plan: 4 passes

**Pass 1 — Foundational diffusion + autoregressive setup**
- Implement basic diffusion scheduler (noise schedule, forward/reverse process)
- Simple VAE-like latent encoder/decoder (toy, not trained)
- Minimal transformer block with standard dot-product attention
- Diffusion loss (MSE on predicted noise)
- Test on synthetic random frames

**Pass 2 — Lightweight causal attention + rotation time-conditioning**
- Replace standard attention with efficient linear causal attention variant
- Implement parameter-free rotation-based time embedding (rotation matrices)
- Verify attention on variable-length sequences

**Pass 3 — Frame sequence autoregressive pipeline**
- Autoregressive frame prediction loop (predict next frame given previous)
- Multi-frame context window handling
- Latent trajectory inference

**Pass 4 — End-to-end toy video generation demo**
- Generate a short synthetic video sequence on toy data
- Visualize frame predictions over timesteps
- Document what was simplified vs. paper

## Implemented vs. simplified

**Pass 1 implemented:**
- Fixed diffusion noise schedule (linear beta schedule)
- Toy latent encoder/decoder using simple random projections
- Single transformer block with causal masking for attention
- Denoising loss (MSE prediction error)
- Minimal test suite with synthetic data (no actual video)

**Pass 1 simplified/skipped:**
- No actual VAE training; using random projection for latents
- No learned positional embeddings (using sinusoidal)
- Single-frame prediction only (no multi-frame sequence yet)
- No pre-training or fine-tuning
- No video data; test only on synthetic tensors

**Pass 2 implemented:**
- **Linear causal attention**: Efficient kernel-based attention using ELU+1 kernel. Computes attention as O = (Q_ker @ K_ker^T @ V) / (Q_ker @ K_ker^T @ 1), with sequential causal masking to reduce quadratic complexity. Works with variable-length sequences.
- **Rotation-based time embedding**: Parameter-free time conditioning using 2D rotation matrices applied element-wise based on timestep. Preserves norm and naturally encodes continuous time information without learnable parameters.
- Updated DenoisingModel to support both linear attention and rotation time embedding via optional flags (use_linear_attn, use_rotation_time).
- Updated TransformerBlock to support linear causal attention.
- Comprehensive test suite (test_pass2.py) verifying: rotation properties (norm preservation), linear attention on variable-length sequences, both components individually and combined, backward compatibility, and training with new components.

**Pass 2 simplified/skipped:**
- Linear attention implemented sequentially (loop over positions) rather than matrix form; not optimized for speed
- Rotation embedding applied post-projection rather than truly RoPE-style (which would rotate during attention computation)
- No actual performance benchmarks; focus on correctness and functionality

**Pass 3 implemented:**
- **SequenceContextDenoisingModel**: Extends denoising to handle frame sequences. Takes a sequence of context frames plus the current noisy frame, projects all to embedding space, concatenates them, applies transformer with causal attention (ensuring current frame can only see previous frames), and predicts noise for the current frame.
- **AutoregressiveFramePredictor**: Orchestrates autoregressive video generation. Implements `predict_next_frame()` which starts from noise and iteratively denoises using the sequence context model; implements `generate_sequence()` which autoregressively generates a full sequence by repeatedly predicting the next frame and maintaining a sliding context window of the last N frames.
- **Multi-frame context window**: Configurable context length (default 4) that limits how many previous frames condition each prediction, making it efficient and bounded.
- **Latent trajectory inference**: Can initialize prediction from single or multiple frames and generate extended sequences, useful for trajectory prediction and video extrapolation.
- Comprehensive test suite (test_pass3.py) verifying: sequence context denoising with standard and optimized attention, autoregressive prediction with both model types, sequence generation from single and multiple initial frames, variable context lengths, and integration with the encoder.

**Pass 3 simplified/skipped:**
- Denoising step uses simplified formula (z_t = (z_t - noise_pred) / alpha) rather than full reverse diffusion process with variance schedule
- No actual video data; all testing on random latent tensors
- Context frames are simply concatenated (no learned fusion or attention weighting between context and current frame)
- Autoregressive generation is single-sample (no parallel batch-level optimizations)
