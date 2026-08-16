# CART: Context-Anchored Recurrent Transformer

**Title:** CART: Context-Anchored Recurrent Transformer -- A Parameter-Efficient Architecture with Learned Stability

**arXiv:** https://arxiv.org/abs/2606.01495

**Authors:** Chad A. Capps, et al.

**Date:** June 2026

## Summary

CART proposes a parameter-efficient language model that reuses a single transformer block across multiple recurrent iterations. Unlike prior looped transformers that recompute all representations at each step, CART separates context encoding from iterative refinement: a multi-layer prelude computes key and value representations once, and a recurrent core reuses these through multi-head latent attention (MLA) cross-attention while applying query-dependent transformations. A learned Linear Time-Invariant (LTI) gate ensures the recurrence remains stable, with spectral radius naturally settling to a narrow band across training. The design reduces per-loop computation and provides consistent attention anchors across recurrent iterations.

## Plan: 4 passes

**Pass 1:** Core recurrent block with multi-head latent attention (MLA) and learned LTI gate for stability. Minimal scaffold: fixed context, simple forward pass, basic test.

**Pass 2:** Multi-layer prelude network that encodes context into reusable K,V representations. Integrate with recurrent core to show the separation of encoding from refinement.

**Pass 3:** Stack multiple recurrent iterations with proper handling of residual connections and layer normalization. Add a simple sequence-level task (e.g., length extrapolation or synthetic reasoning).

**Pass 4:** End-to-end language model with small synthetic dataset, demonstrate parameter efficiency vs. a standard transformer, and honest summary of simplifications.

## Implemented vs. simplified

**Pass 1 implementation:**
- Core MLA block: query input is transformed, K,V are reused from context
- Learned LTI gate: scalar α parameter controlling spectral radius
- Basic forward pass: single recurrent iteration
- Test: verify forward shape and gate stability

**Simplified/stubbed for Pass 1:**
- Prelude network not yet implemented (will be added in Pass 2)
- No multi-head mechanism yet (single projection pass only)
- No layer norm or residual connections (added incrementally in Pass 3)
- No batching or position encoding (will add in later passes)

**Pass 2 implementation:**
- CartPrelude: multi-layer feedforward encoder that processes raw context
- Separate K,V projections: independent linear transformations for expressiveness
- Full Cart model: integrates prelude + core to show separation of encoding from refinement
- Prelude computed once, K,V reused across all recurrent iterations (parameter efficiency)
- Tests: prelude shape/gradients, multiple layer depths, end-to-end integration

**Simplified/stubbed for Pass 2:**
- Prelude is simple feedforward (no attention or normalization yet—added in Pass 3)
- Single context per forward pass (no batched contexts or dynamic resizing)
- No positional encoding for context (will add context position embeddings in Pass 3)
- No explicit parameter count tracking (will compare vs. standard transformer in Pass 4)

**Pass 3 implementation:**
- Layer normalization in CartRecurrentCore: stabilizes stacking across multiple iterations
- Explicit residual connections: x_new = α * residual + y (input added back via LTI gate)
- CartSequenceClassifier: sequence-level classification on top of CART for demonstrating task performance
- Synthetic task: binary classification on sequence length (predict if seq_len >= threshold)
- Dataset creation and training utilities: full pipeline for training and evaluation
- Full test coverage: 8 new tests for Pass 3 components, all passing

**Simplified/stubbed for Pass 3:**
- Task is synthetic length classification (simple proof-of-concept; could extend to language modeling in Pass 4)
- No explicit positional encoding yet (context is treated as unordered bag of features)
- No beam search or decoding strategies (will add in Pass 4 for language generation)
- Model trained on small synthetic dataset (large-scale training deferred to Pass 4)
- Parameter efficiency not yet quantified (will compare vs. baseline transformer in Pass 4)
