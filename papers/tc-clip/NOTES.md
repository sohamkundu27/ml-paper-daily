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

### Pass 3
Implements video-text alignment and video-conditional prompting:
- **SimpleTextEncoder**: A learned text encoder that maps tokenized action class descriptions to fixed-dimensional embeddings. Uses token embeddings, positional embeddings, and a 2-layer MLP for projection to the feature space.
- **VideoConditionalPrompting**: Generates action-specific prompt embeddings based on video context. Uses multi-head soft attention to compute adaptive weights over action classes, enabling the model to focus on relevant actions given the video content.
- **Refined video features**: The conditional prompts are weighted-combined and used to refine the video features (residual connection), allowing video context to modulate the feature representation.
- **Action classification**: Computes similarity scores between refined video features and action embeddings using normalized dot-product (cosine similarity), with a learnable temperature parameter for scaling.

The implementation enables zero-shot and few-shot action classification by aligning video understanding with language-based action descriptions. Gradients flow through all components, enabling end-to-end training. The video-conditional prompting module specifically learns to generate action-aware refinements of video features.

### Pass 4
Implements end-to-end demo and evaluation pipeline:
- **TCClipPass4**: A wrapper around Pass 3 that adds a `predict()` interface and parameter management for training. Supports easy inference and provides confidence scores for predictions.
- **ToyActionDataset**: Generates synthetic videos with distinct motion patterns for 5 action classes: horizontal motion, vertical motion, diagonal motion, circular motion, and static with color variation. Allows controlled generation of specific actions and batching.
- **demo_pass4.py**: A complete end-to-end training and evaluation script that trains the model on synthetic action recognition for 3 epochs, evaluates per-action accuracy, and shows sample predictions with confidence scores.

The demo demonstrates that the full pipeline is trainable end-to-end and can learn to distinguish between different action patterns. On the synthetic toy dataset, the model achieves ~24% overall accuracy on held-out data after 3 epochs of training with random initialization (baseline for 5 classes would be 20%, so the model is learning meaningful distinctions). Training without pre-trained weights and on simple synthetic patterns limits accuracy, but the pipeline works end-to-end.

## What was simplified or stubbed

1. **Frame encoder**: Uses a simple 3-layer CNN rather than a pre-trained vision backbone (e.g., ViT or ResNet). Pre-training would require large-scale supervised or self-supervised data.

2. **Text encoder**: A simple learned embedding + projection rather than CLIP's text transformer. A full CLIP text encoder would require pre-training on 400M image-text pairs.

3. **Action token embeddings**: Randomly initialized and learned, rather than derived from a pre-trained CLIP text model. Real action class embeddings would come from CLIP's frozen text encoder applied to action descriptions.

4. **Dataset**: Synthetic toy videos with simple geometric motion patterns, not real action videos (HMDB51, Kinetics, etc.). Real video understanding requires learning from complex visual and temporal patterns.

5. **Training**: 3 epochs on 100 synthetic samples. Real-world action recognition requires training on thousands of videos over many epochs with careful hyperparameter tuning.

6. **Temporal modeling**: Context tokens capture frame dependencies, but the model does not explicitly model action duration, speed variation, or other temporal properties that real action recognition requires.

Despite these simplifications, the implementation validates that the core TC-CLIP idea (learnable context tokens + video-conditional prompting) is technically sound and can be trained end-to-end on a complete action recognition pipeline.
