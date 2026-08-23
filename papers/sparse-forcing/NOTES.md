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

### Pass 2: Learnable Block Selection

**What is implemented:**
- `BlockScorer` module: lightweight neural network that learns to score block importance
  - Takes averaged value features per block as input
  - Small 2-layer MLP (dim → 64 → 1) produces importance scores
  - Handles both plain and multi-head value shapes automatically
- Extended `SparseAttentionMask` to accept dynamic persistent block indices
- Extended `MultiHeadSparseAttention` with `use_learned_blocks` flag:
  - When enabled, scorer runs on input, selects top-k blocks via softmax scoring
  - Dynamically updates mask based on learned scores per forward pass
  - Maintains compatibility with fixed block selection (backward compatible)
- Comprehensive test suite for learned blocks:
  - Basic scorer functionality
  - Multi-head input handling
  - Learned attention layer correctness
  - Per-batch different block selections
  - Score sensitivity to input changes
  - Comparison between learned vs fixed blocks
- Demo script showing:
  - How block scorer works and visualizes scores
  - Learned block selection in attention layer
  - Comparison on synthetic "video" data (4 frames × 16 tokens)
  - Block scores changing during optimization
  - Training loop showing learned adaptation

**What is simplified or stubbed:**
- BlockScorer pools values by averaging per block (could use more sophisticated aggregation)
- Block selection uses top-k on average scores per batch (deterministic, not sampling)
- No learned ranking or attention over blocks themselves (each block scored independently)
- No persistent block state carried across multiple decoding steps yet
- No end-to-end video generation or downstream task to validate that learned selection is better
- All block selection at test/demo time is deterministic (could add temperature/sampling)

### Pass 3: Persistent Block State Management

**What is implemented:**
- `PersistentBlockState` class: manages block feature states across decoding steps
  - Stores frame-by-frame block features in a history deque
  - Tracks frame indices and timestamps
  - Provides exponentially-weighted averaging for current state (recent frames weighted more heavily)
  - Supports `update()` to add new frame's block features
  - Supports `compress()` to reduce memory by pooling old frames (~40-55% memory reduction)
  - Supports `clear_stale()` to maintain only recent frames (sliding window)
  - Provides `get_block_importance()` via variance-based scoring or learned scorer
  - Includes `memory_info()` for tracking memory usage

- `PersistentBlockCache` class: high-level cache for multi-step decoding
  - Wraps PersistentBlockState with convenient interface
  - `update_from_attention_output()`: extracts and pools attention output into block features
  - `compress()`, `clear_stale()`, `reset()`, `get_state()`, `get_memory_info()` methods
  - Device-agnostic (handles CPU/GPU storage)

- Integration with `MultiHeadSparseAttention`:
  - Added `use_persistent_cache` parameter to constructor
  - Automatically creates and manages cache on first forward pass
  - Updates cache with attention output after each forward pass
  - Maintains state across multiple calls (autoregressive decoding)

- Comprehensive test suite: 13 tests covering:
  - State creation and updates
  - Current state computation with recency weighting
  - Compression effectiveness and correctness
  - Stale clearing and sliding window behavior
  - Cache creation and integration
  - Updating cache from attention output
  - Multi-step decoding simulation
  - Memory tracking
  - Block importance scoring
  - Full autoregressive decoding with state accumulation

- Demo script showing:
  - Block state accumulation across 5 frames
  - Compression reduces memory ~40-55% while preserving state
  - Stale clearing for sliding window operation
  - Full autoregressive cache through 6 decoding steps
  - Integration with sparse attention layer
  - Long-horizon decoding memory efficiency comparison

**What is simplified or stubbed:**
- Compression uses simple averaging of old frames (could weight by importance)
- Block features extracted via spatial pooling only (averaging over tokens)
- No hierarchical compression (could have multi-level summary)
- No learned compression or importance weighting for what to compress
- State update extracts features via averaging (could use more sophisticated pooling)
- Importance scoring uses variance only (could incorporate other metrics)
- No explicit temporal attention over cached blocks
- Cache state is not used in the actual attention computation (pass 4 will integrate)
