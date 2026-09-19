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
- ✅ Router uses token embeddings to predict routing probability per token
- ✅ Forward pass computes both heads and routes output via the router gate
- ✅ Minimal test verifying output shapes and numerical correctness

**Simplified:**
- Router is very simple (single linear layer + sigmoid), not the full adaptive regularization from the paper
- Routing decisions are made independently per token (no layer-aware or sequence-aware context in routing)
- No continual pretraining; just a standalone module
