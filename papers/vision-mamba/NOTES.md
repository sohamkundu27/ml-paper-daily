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
- ✅ Bidirectional SSM block using a simplified state space model (not the full Mamba algorithm)
- ✅ Positional embeddings for image patches
- ✅ Forward/backward scanning of patch sequences
- ✅ Minimal test with shape assertions
- ⚠️ **Simplified:** Uses a linear SSM (no selective scanning; no hardware-aware complexity from Mamba v1). Just basic matrix operations to model state transitions.
- ⚠️ **Simplified:** No layer norm or residual connections yet (added in pass 2)
- ⚠️ **Simplified:** Patches are not extracted from real images; Pass 1 works with pre-flattened synthetic inputs
