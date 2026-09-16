# Log-Linear Sparse Attention for Efficient Diffusion Transformers

**Title:** Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers

**arXiv:** https://arxiv.org/abs/2512.16615

**Authors:** Yifan Zhou, Zeqi Xiao, Tianyi Wei, Shuai Yang, Xingang Pan (S-Lab, Nanyang Technological University)

## Summary

Diffusion Transformers have become state-of-the-art for image generation, but their quadratic self-attention mechanism makes scaling to long token sequences prohibitively expensive. This paper proposes Log-linear Sparse Attention (LLSA), which hierarchically selects key blocks at multiple granularity levels to reduce computation from O(N²) to O(N log N). The method is trainable end-to-end, using learnable log-linear weights to determine block importance, and includes an efficient GPU kernel implementation for practical speedup.

## Plan: 4 passes

**Pass 1 — Hierarchical sparse attention indexing and selection**
Implement the core mechanism: hierarchical Top-K block selection across multiple pyramid levels (coarse to fine). Build the indexing logic and show that we can extract which tokens participate in sparse attention at each level.

**Pass 2 — Trainable log-linear attention mechanism**
Make the selection learnable: replace static Top-K with log-linear attention scores that are trained end-to-end. Compute attention only between selected blocks, not the full matrix.

**Pass 3 — Integration with diffusion backbone**
Connect the sparse attention layer to a small diffusion transformer model. Show that it reduces memory/compute vs. full attention while maintaining output shape and gradients.

**Pass 4 — End-to-end demo on synthetic data**
Run a small synthetic diffusion task (e.g., denoising random noise) demonstrating the speedup, and write a final summary of what was actually implemented vs. simplified.

## Implemented vs. simplified

### Pass 1 (Hierarchical sparse attention indexing)
- ✓ Hierarchical pyramid structure: build coarse-to-fine token groups across multiple levels
- ✓ Top-K block selection: at each level, select the K most important blocks
- ✓ Attention index extraction: return which tokens participate in sparse attention
- ✗ GPU kernel: using dense PyTorch operations for now (not optimized)
- ✗ Integration with actual attention: just the indexing; actual QKV multiplication comes later
- ✗ Learnable weights: selection is fixed Top-K; training comes in Pass 2
