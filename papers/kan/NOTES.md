# Kolmogorov-Arnold Networks (KAN)

**Title:** Kolmogorov-Arnold Networks

**arXiv:** https://arxiv.org/abs/2404.19756

**Authors:** Ziming Liu, Yonatan Belinkov, Nir Shavit, et al.

## Summary

Instead of using fixed activation functions in neural networks, Kolmogorov-Arnold Networks replace the standard weight matrix plus activation function pattern with learnable univariate functions placed on edges. The key insight is inspired by the Kolmogorov-Arnold representation theorem: any multivariate continuous function can be decomposed into a sum of univariate functions. Rather than learning coefficients alone, KAN learns the actual functional form of each edge, using B-splines as the parameterization. This approach can lead to sparser, more interpretable networks that may learn analytical expressions more naturally than traditional MLPs.

## Plan: 4 passes

**Pass 1** — Foundational B-spline KAN layer. Implement a single KAN layer with B-spline basis functions and learnable control points. Build the forward pass for a single univariate function approximation. Include a minimal test showing that a single KAN layer can learn a simple function.

**Pass 2** — Full KAN module with grid refinement. Extend to multi-variable input with learnable B-spline grids per edge. Implement the activation function on each edge and compose them correctly. Add support for changing spline resolution adaptively.

**Pass 3** — KAN-based classifier or regressor. Build a small feedforward network composed of KAN layers and train it on toy data (e.g., MNIST or a synthetic dataset). Demonstrate end-to-end learning with backprop.

**Pass 4** — End-to-end demo and sparsification. Show a trained KAN solving a simple regression task. Add a small pruning routine to highlight sparsity. Provide a final summary of what was implemented and what was simplified (e.g., approximations to the B-spline computation, no adaptive grid refinement beyond pass 2 level).

## Implemented vs. simplified

**Pass 1 - What is implemented:**

- A fully differentiable `KANLayer` module that replaces the standard linear + activation pattern with learnable B-spline univariate functions
- Piecewise linear B-spline basis functions (tent functions) for each edge, fully vectorized using tensor operations to maintain gradient flow
- Learnable control points (parameters) for each input-output feature pair
- A working forward pass that computes weighted sums of basis functions
- End-to-end learning via backpropagation: the layer can successfully learn identity functions, nonlinear functions (e.g., x²), and multi-feature mappings

**Pass 1 - What is simplified/stubbed:**

- Uses piecewise linear (degree 1) B-splines rather than higher-degree splines mentioned in the paper (e.g., cubic)
- Does not implement adaptive grid refinement; grid size is fixed at initialization
- No sparsity enforcement or pruning mechanism
- No activation spline (the paper may use a separate activation component on edges)
- Does not implement the paper's full architecture for composing KAN layers into deep networks
- Grid normalization is fixed to [-1, 1]; no per-feature normalization
- No L1 regularization for inducing sparsity as suggested by the paper

These simplifications allow for a clean, minimal implementation that demonstrates the core idea (learnable univariate functions on edges) while remaining easy to understand and efficient to train on toy problems.
