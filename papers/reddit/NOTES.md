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

### Pass 2: ✅ Complete
**What works:**
- `forward_with_rehash(x0, t, num_corrupts)` implements randomized multi-index corruption: randomly selects `num_corrupts` positions per sample and corrupts only those, leaving others unchanged
- Returns both noised tokens and a boolean corruption mask
- `forward_batch_with_rehash(x0, timesteps, num_corrupts)` handles batch processing with per-sample timesteps and independent random corruption patterns
- Creates diverse training paths: different samples and different calls produce different corruption patterns even from the same x0
- Path diversity verified via entropy: corrupted positions show non-zero entropy but constrained (not fully uniform)
- Unmasked positions preserved unchanged from x0
- Full test suite: 5 new tests verify basic functionality, corruption count control, batch processing, path diversity (entropy > 0.1), and position preservation

**Simplified/stubbed:**
- Simple uniform random position selection (paper may use more sophisticated scheduling)
- No learned corruption pattern adaptation (fully random selection)
- Single corruption per position per step (not nested/hierarchical)
- Corruption mask returned but not used in training loop (would need reverse process to leverage)
- Still numpy-only, no actual training loop or model integration

### Pass 3: ✅ Complete
**What works:**
- `SimpleDenoiser` class: lightweight MLP (PyTorch) that predicts p(x_0 | x_t, t)
  - One-hot encodes noised tokens x_t
  - Embeds timestep t with learnable embeddings
  - MLP with two hidden layers maps to output logits for each position
  - Handles both scalar and per-sample timesteps gracefully
- `reverse_kernel(x_t, t, x_0_pred)` computes reverse distribution analytically
  - Uses Bayes rule to derive q(x_{t-1} | x_t, x_0)
  - Combines forward transition matrices Q_t and Q_{t-1}
  - Samples x_{t-1} from the posterior, position by position
  - Handles edge cases (zero posterior) with uniform fallback
- `sample_with_rehash_sampler(denoiser, x_T)` runs full reverse sampling loop
  - Iterates from t=num_steps down to t=1
  - At each step, denoise to predict x_0, then reverse-sample x_{t-1}
  - Produces final clean sample x_0 from initial noise x_T
- `_denoise_step(denoiser, x_t, t)` interface for model predictions
  - Converts numpy → torch, runs inference, returns argmax predictions
  - No gradients computed (inference only)
- Full test suite: 7 new tests verify denoiser output shape, reverse kernel validity, end-to-end sampling, and synthetic classification task

**Simplified/stubbed:**
- Untrained denoiser: SimpleDenoiser initialized randomly, so predictions are uninformed
  - Reverse process still runs correctly (mathematically sound)
  - Generated samples are random but valid (shows structure is in place for training)
- No actual training loop (optimizer, loss function, dataset)
  - Code is ready for training but not executed; tests only verify forward/backward pass plumbing
- No learned timestep embedding optimization
- Reverse process does not use corruption masks (full position-by-position reversal)
  - Paper's rehash sampler would selectively update only corrupted positions
  - Current implementation handles all positions uniformly in reverse
- No batch inference optimization (processes one timestep at a time)
- Single sample generation per call (no batch generation in sampling loop)
