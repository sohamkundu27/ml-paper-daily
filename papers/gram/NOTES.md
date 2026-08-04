# GRAM: Generative Recursive Reasoning Models

**Title:** Generative Recursive Reasoning

**arXiv:** https://arxiv.org/abs/2605.19376

**Authors:** Junyeob Baek, Mingyu Jo, Minsu Kim, Mengye Ren, Yoshua Bengio, Sungjin Ahn

**Summary:**

This paper proposes a framework for treating recursive reasoning itself as a probabilistic, stochastic process in latent space. Rather than following a single deterministic reasoning trajectory, GRAM models reasoning as multiple hypothesis exploration through stochastic latent transitions. The key innovation is formulating recursive latent reasoning as a generative process with probabilistic transitions between latent states, enabling both diverse solution discovery and efficient parallel inference through trajectory sampling. Unlike prior deterministic recursive models that converge to a single prediction, GRAM can explore alternative strategies and represent multiple plausible reasoning paths simultaneously.

## Plan: 4 passes

**Pass 1 (Foundation):** Basic recursive latent reasoning framework.
- Implement encoder and decoder for latent representation
- Single-step transition function from one latent state to the next
- Basic forward pass that applies the transition deterministically for N steps
- Simple reconstruction loss (MSE)
- Test on synthetic constraint satisfaction task

**Pass 2 (Stochastic Core):** Add stochastic trajectory generation.
- Introduce learnable Gaussian perturbations on latent states
- Implement amortized variational inference for stochastic transitions
- Add KL divergence regularization on the perturbation distribution
- Support multiple trajectory sampling at inference

**Pass 3 (Scaling):** Inference-time scaling and trajectory management.
- Implement parallel trajectory sampling and merging
- Add trajectory resampling based on likelihood
- Support variable recursion depth per trajectory
- Implement trajectory pruning for efficiency

**Pass 4 (Demo):** End-to-end demonstration on constraint satisfaction.
- Synthetic constraint satisfaction problem (e.g., simple SAT-like constraints)
- Demonstrate multiple solution discovery via trajectory sampling
- Show inference-time scaling benefits
- Final summary and honest reflection on what was simplified

## Implemented vs. simplified

**Pass 1 complete:**
- ✅ Encoder: MLP from input to latent space
- ✅ Decoder: MLP from latent space to output
- ✅ Transition function: Simple linear transformation (no adaptivity)
- ✅ Forward loop: Deterministic N-step recursion
- ✅ Loss: Mean squared error reconstruction
- ✅ Test: Basic toy constraint problem (numeric sequence with constraints)

**Simplified in Pass 1:**
- Transition function is fixed linear, not learned adaptively
- No variational inference yet (added in Pass 2)
- Single trajectory only (no stochastic sampling)
- Toy problem is very simple (just constraint satisfaction on 5D vectors)
- No dual-loop architecture from paper (simplified to single loop)

**Pass 2 complete:**
- ✅ Stochastic transitions: Learned Gaussian perturbations on latent states
- ✅ Amortized variational inference: Transition network outputs (mean, log_var)
- ✅ KL divergence regularization: KL(q(z_t+1|z_t) || N(0,1)) for each step
- ✅ Multiple trajectory sampling: Can sample diverse trajectories at inference
- ✅ Flexible loss: Reconstruction + weighted KL term (configurable weight)
- ✅ Tests: Forward pass, stochasticity verification, KL loss computation, training convergence, constraint satisfaction with stochastic reasoning

**Simplified in Pass 2:**
- No trajectory resampling/selection yet (added in Pass 3)
- No parallel trajectory merging or pruning (added in Pass 3)
- Variational inference is basic VAE-style with fixed prior N(0,1)
- No learned prior or hierarchical latent structure
- Single-sample training (one trajectory per forward pass, though can sample multiple at inference)
