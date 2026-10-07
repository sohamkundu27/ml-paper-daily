# Vision Mamba

**arXiv:** https://arxiv.org/abs/2401.09417

**Authors:** Yue Meng, Rui Xia, et al. (Huawei Noah's Ark Lab, Tsinghua University)

**Venue:** ICML 2024 (Accepted)

## Summary

Vision Mamba proposes using bidirectional state space models (SSMs) for vision tasks as an alternative to attention-based transformers. Rather than operating on patches with quadratic-time attention, Vision Mamba applies efficient linear-time SSM blocks (like Mamba) in both forward and backward directions to capture bidirectional context while maintaining computational efficiency. The approach achieves competitive accuracy on ImageNet, COCO detection, and ADE20k segmentation with lower memory and compute requirements than vision transformers.

## Plan: 4 passes

**Pass 1:** Core Vision Mamba block — implement a bidirectional SSM module that processes flattened image patches. Include positional embeddings and basic block structure. Test on a small synthetic image tensor to verify shape transformations.

**Pass 2:** Vision backbone assembly — stack multiple Mamba blocks into a simple feature extractor with multi-scale output. Add layer normalization and residual connections. Test intermediate feature map extraction.

**Pass 3:** Classification head + ImageNet-scale inference — add a global average pooling and linear classifier. Implement a minimal inference pipeline on a real image (or synthetic test data). Verify end-to-end shape flow.

**Pass 4:** End-to-end demo and honest summary — build a tiny toy classification task (CIFAR-10 or similar synthetic data), train the model for a handful of steps, and report accuracy. Document all simplifications made.

## Implemented vs. simplified

**Pass 1 implementation:**
- ✅ ImagePatcher: converts images to patch embeddings
- ✅ PositionalEmbedding: learnable positional embeddings for patches
- ✅ LinearSSMBlock: simple linear state space model (not selective)
- ✅ VisionMambaBlock: combines SSM with layer norm and residual connections
- ✅ VisionMambaPass1: stacks multiple blocks into an encoder
- ⚠️ **Simplified:** LinearSSMBlock uses basic matrix operations without selective gating. No input-dependent state update modulation.
- ⚠️ **Simplified:** Patches are assumed to be fixed-size; no multi-scale hierarchical feature extraction

**Pass 2 implementation:**
- ✅ SelectiveSSMBlock: SSM with learnable gating (sigmoid gate modulates state updates)
- ✅ BidirectionalSSMBlock: scans forward and backward using SelectiveSSMBlock, combines with gated fusion
- ✅ VisionMambaBlock: now supports both linear (unidirectional) and selective bidirectional SSM via use_bidirectional flag
- ✅ VisionMambaPass2: uses bidirectional selective SSM blocks for improved context aggregation
- ✅ Backward compatibility: Pass 1 continues to work with LinearSSMBlock (unidirectional)
- ⚠️ **Simplified:** Selective gating is simple sigmoid-based; no learned dynamics or adaptive computation. Does not use actual Mamba-style hardware-aware complexity.
- ⚠️ **Simplified:** Bidirectional combination is a simple gated linear fusion, not a learned cross-attention mechanism
- ⚠️ **Simplified:** No multi-scale pyramid or hierarchical downsampling (reserved for Pass 3)

**Pass 3 implementation:**
- ✅ VisionMambaPass3: extends Pass 2 backbone with classification head
- ✅ Global average pooling: reduces (B, num_patches, embed_dim) to (B, embed_dim)
- ✅ Linear classifier: projects embed_dim to num_classes for logit output
- ✅ End-to-end inference: image (B, C, H, W) → logits (B, num_classes)
- ✅ Training support: full backward pass with cross-entropy loss, verified on toy data
- ⚠️ **Simplified:** Classification head is a single linear layer; no dense layers or feature fusion stages
- ⚠️ **Simplified:** Global average pooling directly reduces patch dimension; no learnable aggregation weights
- ⚠️ **Simplified:** No multi-scale hierarchical features or resolution adaptation
