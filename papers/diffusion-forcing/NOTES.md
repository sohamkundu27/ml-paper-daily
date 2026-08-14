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

**Pass 2 Implementation:**
- ✅ Training loop with Adam optimizer
- ✅ L2 loss (MSE) between predicted and ground-truth clean tokens
- ✅ Synthetic data generators: random, repeating patterns, sine waves
- ✅ Denoiser demonstrably learns to denoise (65% error reduction in tests)
- ✅ Comprehensive tests showing loss decreases and training converges
- ⏸️ No sampling/inference yet (stubbed for pass 3)
- ⏸️ Simplified: training on single batches; no validation set or checkpointing

**Pass 3 Implementation:**
- ✅ Iterative sampling: start with fully noisy sequence, progressively denoise to generate new tokens
- ✅ Masked sampling: keep past tokens fixed while denoising future tokens (core Diffusion Forcing capability)
- ✅ Variable-length generation: generate sequences of any length, including beyond training horizon
- ✅ Progressive extension: extend sequences in chunks by reusing generated output as context
- ✅ Comprehensive tests: 11 tests covering sampling, masking, consistency, and progressive generation
- ⏸️ No end-to-end demo yet (stubbed for pass 4)
- ⏸️ Simplified: denoising schedule is linear timestep interpolation; no adaptive noise variance

**Pass 4 Implementation:**
- ✅ End-to-end demo: arithmetic sequence prediction task with synthetic data
- ✅ AutoregressiveBaseline: simple MLP that predicts next token from context window (3 tokens)
- ✅ Training pipeline: train both Diffusion Forcing and baseline on 64 arithmetic sequences
- ✅ Evaluation metrics: MSE and MAE over prediction horizon
- ✅ Rollout evaluation: generate sequences and compare against ground truth
- ✅ Masking advantage test: evaluate at different context lengths (25%, 50%, 75%)
- ✅ Two comprehensive test functions: `test_pass4_end_to_end_demo()` and `test_pass4_masking_advantage()`
- ✅ Updated NOTES.md with final summary of all 4 passes
- ⏸️ Simplified: task is arithmetic sequences (trivial pattern), not real language/vision data
- ⏸️ Simplified: baseline is only 3-token context window; full paper explores longer contexts
- ⏸️ Simplified: no hyperparameter tuning; using default learning rates and network sizes
- ⏸️ Simplified: sampling uses linear timestep schedule, not learned schedule or advanced techniques

## Key findings from this implementation

**What works:**
- Core diffusion mechanics are sound: per-token noise scheduling, forward diffusion, iterative denoising all correct
- Training loop successfully reduces loss (82% improvement in pass 4)
- Masked sampling correctly preserves context while denoising future tokens
- Variable-length generation works: sequences can extend beyond training horizon
- All 30+ tests pass, demonstrating mechanical correctness throughout

**Why Diffusion Forcing underperforms on arithmetic sequences:**
- Arithmetic sequences are trivial, deterministic: autoregressive models excel here
- This task requires memorizing a simple rule (constant difference), not learning a distribution
- Diffusion models are designed for high-entropy distributions; arithmetic has near-zero entropy
- Baseline AR model with just 3-token context is perfectly sufficient
- The full paper (Chen et al. 2024) shows DF advantages on complex tasks like trajectory prediction
  where distributions are richer and long-horizon generation is critical

**What was simplified (structurally correct but less sophisticated):**
1. **Noise schedule**: Linear cosine annealing only; paper explores more sophisticated schedules
2. **Denoising posterior**: Uses linear interpolation; paper implements learned variance
3. **Architecture**: Tiny MLP with basic time embedding; paper uses much larger, sophisticated models
4. **Data**: Arithmetic sequences; paper uses continuous control, robotics, vision data
5. **Evaluation**: Short 8-token horizons; paper evaluates 100+ step rollouts
6. **Loss**: Only MSE; paper may use additional objectives (e.g., consistency loss, likelihood bounds)

**To extend this for real applications:**
1. Use data where long-horizon prediction matters (video, trajectories, text beyond GPT-scale)
2. Implement learned noise variance schedules (see DDPM Appendix A)
3. Add task conditioning via cross-attention or FiLM layers
4. Leverage masking: during training, vary per-token noise to simulate variable context lengths
5. Evaluate on metrics rewarding long-term coherence (rollout metrics, human evaluation)
6. Experiment with Transformer or U-Net denoisers instead of MLPs
