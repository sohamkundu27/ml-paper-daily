# Sparse Forcing: Native Trainable Sparse Attention for Real-time Autoregressive Diffusion Video Generation

**arXiv:** https://arxiv.org/abs/2604.21221  
**Authors:** Contributors from multiple institutions (see arXiv for full author list)  
**Published:** April 2026

## Summary

This paper addresses the challenge of generating long videos efficiently using diffusion-based autoregressive models. The core observation is that in video generation, the attention mechanism naturally concentrates computation on a small set of visually important regions that persist across frames, and these regions follow a block-sparse pattern within sliding temporal windows. Rather than computing full attention, the paper proposes learning which blocks matter and maintaining them across decoding steps. This enables faster video generation while preserving quality through an efficient sparse attention kernel that accelerates both training and inference.

## Plan: 4 Passes

**Pass 1 — Sparse Attention Masking**  
Implement the foundational sparse attention mechanism: block-wise attention masking where attention is computed only within local blocks and a fixed set of persistent blocks. Create a basic attention mask generator and demonstrate that it correctly identifies and masks non-local positions. This is the core building block upon which everything else depends.

**Pass 2 — Learnable Block Selection**  
Add a trainable mechanism to learn which blocks should be marked as persistent/salient. Implement a lightweight scoring network that predicts block importance from the value cache, and demonstrate that learned block selection outperforms fixed patterns on synthetic video data.

**Pass 3 — Persistent Block State Management**  
Implement proper management of persistent blocks across multiple decoding steps: compressing block states, updating them as new frames arrive, and clearing stale information. This adds the long-horizon memory aspect that allows the model to maintain visual consistency across many frames.

**Pass 4 — End-to-End Video Generation Demo**  
Integrate sparse forcing into a minimal diffusion model and demonstrate end-to-end video generation on toy data (small resolution, short sequences). Show timing comparisons and verify that sparse attention produces reasonable output and runs faster than full attention baselines.

## Implemented vs. Simplified

### Pass 1: Sparse Attention Masking

**What is implemented:**
- `SparseAttentionMask` class: generates binary masks for local + persistent block patterns
- Block-wise masking: each position attends to its own block + fixed persistent blocks
- Efficient vectorized mask generation and sparsity statistics
- `MultiHeadSparseAttention` layer: full PyTorch module with Q/K/V projection, masked softmax, multi-head support
- Mask caching for efficiency across different sequence lengths
- Comprehensive test suite: 6 tests covering correctness, sparsity, masking, variable seq lengths
- Demo script showing attention patterns, compression ratios, and computational savings (~3.4x speedup for seq_len=256)

**What is simplified or stubbed:**
- No CUDA kernel; uses standard PyTorch (slower but fully functional)
- Persistent block selection is fixed (hardcoded to first N blocks), not learned
- No persistent block state updates across multiple frames
- No actual video generation; purely demonstrates the attention mechanism
- Single forward pass; no recurrent decoding or KV cache management yet
