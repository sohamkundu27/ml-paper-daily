# TC-CLIP: Leveraging Temporal Contextualization for Video Action Recognition

**Title**: Leveraging Temporal Contextualization for Video Action Recognition

**arXiv**: [2404.09490](https://arxiv.org/abs/2404.09490)

**Venue**: ECCV 2024

**Authors**: Minji Kim, Dongyoon Han, Taekyung Kim, Bohyung Han (NAVER AI)

## Summary

This paper addresses the problem that video understanding models often fail to effectively leverage temporal information across frames. TC-CLIP proposes two key mechanisms. First, Temporal Contextualization (TC) extracts key information from video frames and summarizes it into learnable context tokens that capture cross-frame relationships; these context tokens are then injected into each frame's encoding process to enable global temporal awareness. Second, Video-conditional Prompting (VP) customizes text prompts based on action-specific video context to improve text-image alignment in the CLIP framework. Together, these components enable zero-shot and few-shot video understanding by extending the standard CLIP architecture to reason about temporal dynamics.

## Plan: 4 passes

**Pass 1 (Foundational)**: Implement basic frame feature extraction and temporal pooling. Load a video, extract per-frame embeddings using a simple pre-trained encoder (e.g., ResNet or vision transformer), and implement naive temporal aggregation (mean pooling). Verify the pipeline produces output of expected shape.

**Pass 2 (Core Method)**: Implement the Temporal Contextualization mechanism. Add learnable context tokens that summarize key information across frames. Integrate context token generation into the feature encoding pipeline. Show that context tokens attend to diverse temporal positions.

**Pass 3 (Integration)**: Integrate with CLIP's text encoder. Implement Video-conditional Prompting to generate action-specific prompts. Combine video context features with text embeddings for action classification on toy videos.

**Pass 4 (End-to-end Demo)**: Build a complete pipeline that takes synthetic videos, generates class predictions using TC-CLIP, and evaluates on a toy action recognition dataset (e.g., HMDB51 subset or synthetic data). Provide a final summary of what works and what was simplified.

## Implemented vs. simplified

### Pass 1
Uses a simple 3-layer CNN for frame feature extraction (not pre-trained) and global average pooling for temporal aggregation. Learnable context tokens and video-conditional prompting are deferred to later passes. This validates the basic video encoding pipeline.

### Pass 2
Implements the core Temporal Contextualization (TC) mechanism:
- **Learnable context tokens**: A set of learnable parameters that act as "summary" tokens for the video
- **Multi-head cross-attention**: Context tokens attend to frame features to capture cross-frame relationships
- **Context injection**: Each frame is enriched by combining its original feature with attention-weighted context information
- **Residual connections and layer norm**: Standard Transformer components for stable training

The implementation uses 4 context tokens by default and 8 attention heads. Context tokens are initialized randomly and learned during training. The attention mechanism shows that context tokens attend to diverse temporal positions, enabling global temporal awareness. Video-conditional prompting and CLIP integration are deferred to Pass 3.
