# PISA: Piecewise Sparse Attention for Efficient Diffusion Transformers

**Title:** PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers

**ArXiv:** https://arxiv.org/abs/2602.01077

**Authors:** Haopeng Li, Shitong Shao, Wenliang Zhong, Zikai Zhou, Lichen Bai, Hui Xiong, Zeke Xie

**Published:** February 2026

## Summary

Diffusion Transformers face a critical efficiency bottleneck: the quadratic complexity of attention mechanisms, which becomes prohibitive when generating long sequences (video or high-resolution images). Most sparse attention methods either discard attention entirely or use hand-crafted patterns that may miss important interactions.

PISA proposes that attention scores in non-critical blocks follow a stable distribution and can be accurately approximated via Taylor expansion, rather than discarded. The method partitions the attention computation into exact computation for critical blocks and efficient approximation for the rest, achieving sub-quadratic complexity while maintaining generation quality. The key contribution is recognizing that approximate attention is far superior to no attention in these regions.

## Plan: 4 passes

**Pass 1:** Implement block-wise attention pattern detection and basic piecewise sparse attention. This is the foundation: partition a sequence into blocks, identify which blocks are "critical," and apply sparse attention patterns. No Taylor expansion yet—just the block structure and mask generation.

**Pass 2:** Add Taylor expansion approximation for non-critical blocks. Implement the exact-or-approximate strategy using block-wise expansion to compute attention scores efficiently.

**Pass 3:** Integrate with a minimal diffusion timestep pipeline and add efficiency metrics (FLOPs, latency). Show how sparse attention scales better than dense attention.

**Pass 4:** End-to-end demonstration on a toy diffusion task (e.g., conditional generation on small synthetic data) with performance comparison and final summary.

## Implemented vs. simplified

### Pass 1

**Implemented:**
- `BlockwiseSparseAttention` module that partitions a batch of tokens into blocks
- Critical block detection based on attention score variance (identifies which blocks have high-variance, potentially important attention patterns)
- Mask generation for block-wise sparse patterns
- Full attention computation (no approximation yet)
- Minimal unit test verifying mask shapes and sparsity pattern

**Simplified/Stubbed:**
- No Taylor expansion (approximation will come in Pass 2)
- No efficiency metrics; Pass 1 just validates the blocking mechanism and mask generation
- Dummy critical block identification based on variance; a real implementation would learn this
- No integration with actual diffusion models; just the attention component in isolation
