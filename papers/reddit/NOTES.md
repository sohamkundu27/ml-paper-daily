# ReDDiT: Rehashing Noise for Discrete Visual Generation

**arXiv:** https://arxiv.org/abs/2505.19656

**Authors:** Tianren Ma, Wentian Xu, Kai Wang, Yanzhi Wang, Chao Yang

**Published:** May 26-29, 2025

## Summary

Discrete diffusion models are emerging as an efficient alternative to continuous diffusion for visual generation, offering better computational efficiency and compatibility with discrete representations. However, they typically lag behind their continuous counterparts in generation quality. The paper identifies the root issue: the standard absorbing state (fully noised) design limits the diversity of noise paths during training, and naive sampling heuristics during generation lead to high discrepancy.

The authors propose ReDDiT (Rehashing Noise for Discrete Diffusion Transformer), which enriches the training signal by using randomized multi-index corruption patterns instead of a single absorbing state. During inference, a "rehash sampler" leverages these diverse paths to ensure low-discrepancy generation. The approach dramatically improves discrete diffusion quality (gFID: 6.18 → 1.61) while maintaining computational efficiency advantages over continuous methods.

## Plan: 4 passes

**Pass 1 — Basic discrete diffusion forward process**
- Implement categorical noise diffusion: a sequence of tensors with categorical distributions
- Build a simple forward diffusion process that gradually corrupts discrete tokens
- Categorical transition matrix and sampling
- Minimal test: verify that repeated corruption converges to uniform distribution

**Pass 2 — Rehashing mechanism (multi-index corruption)**
- Replace single absorbing state with randomized corruption patterns
- Implement multi-index noise injection during training
- Corrupt multiple independent positions per step with random indices
- Test that rehashing increases path diversity

**Pass 3 — Reverse process and rehash sampler**
- Implement a simple denoiser network (lightweight MLP or small transformer block)
- Build the reverse sampling loop with the rehash sampler strategy
- Incorporate the multi-index inversion into the sampling procedure
- Test on a small synthetic classification task

**Pass 4 — End-to-end demo and summary**
- Implement simple one-hot encoding for discrete tokens (e.g., on MNIST digits or toy dataset)
- Full forward → reverse pipeline with learned denoiser
- Visualize generated vs. original samples
- Document what was simplified (e.g., no scale, no transformer-scale denoiser, toy data only)

## Implemented vs. simplified

### Pass 1: ✅ Complete
**What works:**
- `CategoricalDiffusion` class with linear alpha schedule (1.0 → 1e-3 over 100 steps)
- Transition matrices Q_t for each timestep, modeling discrete categorical transitions
- `forward(x0, t)` samples x_t from q(x_t | x_0) using transition matrix rows
- `forward_batch(x0, timesteps)` handles multiple timesteps efficiently
- Full test suite: 6 tests verify transition matrix validity, convergence to uniform at late timesteps, schedule properties, and that noise increases over time

**Simplified/stubbed:**
- Linear alpha schedule (paper doesn't specify; could use cosine, sqrt, etc.)
- No rehashing yet (that's pass 2)
- No learned denoiser or reverse process
- No actual image data or training
- Pure numpy implementation (no pytorch)
