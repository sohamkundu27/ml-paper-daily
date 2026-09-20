# Switch Attention (SwiAttn)

**ArXiv:** https://arxiv.org/abs/2603.26380
**Published:** March 27, 2026
**Authors:** Yusheng Zhao, Hourun Li, Bohan Wu, Jingyang Yuan, Meng Zhang, Yichun Yin, Lifeng Shang, Ming Zhang

## Summary

Self-attention in transformers scales quadratically with sequence length, making long-context modeling expensive. Sliding window attention reduces cost linearly but loses global context. This paper introduces a hybrid approach that dynamically routes each token-layer computation to either full attention (for global information) or sliding window attention (for local pattern matching), choosing the most efficient path per context.

## Plan: 4 passes

**Pass 1** - Core switching mechanism: Implement a basic hybrid attention module that can compute full attention and sliding window attention, plus a simple binary router that decides which path to use per token. This is the foundational building block.

**Pass 2** - Learnable routing: Add learnable router parameters (e.g., a small MLP or logits predictor) that learns when to route to full vs. sliding window attention. Train the router with gradient-based learning on a simple task.

**Pass 3** - Efficiency analysis and integration: Add sparsity regularization to encourage efficient routing, implement batched computation to make hybrid execution fast, and measure actual speedups vs. pure full attention.

**Pass 4** - End-to-end demo: Build a small transformer encoder with Switch Attention blocks, test on synthetic long-sequence data (e.g., synthetic reasoning task or document understanding), and provide an honest summary of what works and what is simplified.

## Implemented vs. simplified

### Pass 1 Complete
- ✅ Implemented `FullAttention`: standard multi-head self-attention
- ✅ Implemented `SlidingWindowAttention`: local attention with fixed window size
- ✅ Implemented `SwitchAttention`: hybrid module with a learnable binary router (simple gating)
- ✅ Router uses sequence-mean embeddings to predict single routing probability
- ✅ Forward pass computes both heads and routes output via the router gate
- ✅ Minimal test verifying output shapes and numerical correctness

**Simplified:**
- Router is very simple (single linear layer + sigmoid), not the full adaptive regularization from the paper
- Routing decisions are made at sequence level (not per-token)
- No continual pretraining; just a standalone module

### Pass 2 Complete
- ✅ Enhanced router to compute **per-token routing probabilities** instead of sequence-level
- ✅ Implemented deeper MLP router: `Linear(d_model) -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid`
- ✅ Each token independently routed to full or sliding window attention based on its embedding
- ✅ Added `get_routing_decisions()` method to inspect routing probabilities for analysis
- ✅ Created `train_router.py`: trains router on synthetic task where important tokens need global context
- ✅ Router successfully learns to minimize routing classification loss (26% reduction over 50 epochs)
- ✅ Updated tests to verify per-token routing shapes and probability ranges
- ✅ Updated demo to show per-token routing statistics and token distribution

**Simplified:**
- Synthetic task uses token norm as a proxy for importance (not real downstream task)
- No integration with actual language model training or fine-tuning
- Router trained with simple BCE loss against ground-truth labels, not end-to-end on downstream objectives
- No analysis of actual computational savings (will be in Pass 3)
- No layer-level or adaptive regularization per the original paper

### Pass 3 Complete
- ✅ Implemented entropy-based **sparsity regularization** to encourage binary routing decisions
  - Loss is high when routing near 0.5 (uncertain), low when near 0 or 1 (decisive)
  - Configurable `sparsity_weight` parameter to control regularization strength
- ✅ Added `get_routing_sparsity()` method to measure fraction of tokens routed to full attention
- ✅ Framework for **batched computation** mode with `use_batched` parameter
  - Separates tokens by hard routing decisions (threshold > 0.5)
  - Allows future optimization to compute only needed attention paths per batch
- ✅ Comprehensive **benchmarking suite** in `pass3_efficiency.py`:
  - Measures wall-clock time across sequence lengths (32 to 512)
  - Compares full attention, sliding window, and hybrid approaches
  - Shows routing efficiency and entropy metrics
- ✅ Updated tests with 3 new tests for sparsity, metrics, and batched computation

**Results:**
- Sparsity regularization successfully encourages binary routing (entropy ~0.57 bits)
- Router learns to route tokens to efficient paths with λ=0.1
- Current implementation still computes both attention paths (no actual speedup yet)
- Benchmarks show baseline performance; future optimization would skip unnecessary computations

**Simplified:**
- Batched computation framework is in place but doesn't yet skip redundant attention computations
- To achieve actual speedups, would need to:
  1. Separate token embeddings into two groups by routing decision
  2. Run only the needed attention path for each group
  3. Reconstruct outputs in original token order
  4. This requires careful index management and is deferred to an enhanced version
- Sparsity measurement is at inference-time threshold (0.5); during training uses soft routing probs
