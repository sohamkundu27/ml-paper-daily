# GenCeption: Video Generation Models as General-Purpose Vision Learners

**arXiv:** https://arxiv.org/abs/2607.09024  
**Publication Date:** July 10, 2026  
**Authors:** Letian Wang, Chuhan Zhang, Rishabh Kabra, Jasper Uijlings, Steven Waslander, Andrew Zisserman, Joao Carreira, Kaiming He, Misha Andriluka, Eduard Gabriel Bazavan, Andrei Zanfir, Cristian Sminchisescu

## Summary

GenCeption repurposes a pre-trained text-to-video generative diffusion model into a unified vision learner that can tackle diverse tasks (depth estimation, surface normals, segmentation, camera pose, keypoint detection) through text-guided instruction. The key insight is that spatiotemporal priors learned during large-scale video generation transfer effectively to general vision tasks when coupled with task-specific text conditioning.

The model takes an image and a natural-language instruction (e.g., "estimate depth") as input and outputs task-specific predictions in a single forward pass. This challenges the conventional wisdom that tasks require task-specific architectures, demonstrating instead that a single generative backbone can function as a general-purpose visual intelligence system.

## Plan: 4 Passes

**Pass 1 — Foundational architecture and instruction conditioning**
- Build a simplified diffusion-based feature extractor (small UNet with convolutional blocks).
- Add instruction/text embedding module (embed text instructions into a conditioning vector).
- Implement basic forward pass: image → backbone features → instruction-conditioned representation.
- Create a minimal test on synthetic image + instruction pairs.
- Goal: Show that the backbone can accept and condition on instructions; no task-specific heads yet.

**Pass 2 — Multi-task decoding heads**
- Add task-specific decoder heads for 2-3 key tasks (depth prediction, surface normals, segmentation).
- Implement a task-selection mechanism: instructions are parsed or embedded to select which head to use.
- Each head outputs task-appropriate shapes (depth: (H, W), normals: (H, W, 3), segmentation: (H, W, num_classes)).
- Test that different instructions produce outputs from different heads.

**Pass 3 — Lightweight task-agnostic refinement**
- Add a shared refinement module after the backbone that learns task-agnostic feature processing.
- Implement a simplified version of cross-task feature sharing or attention.
- Optionally add a minimal training loop on synthetic data (gradient updates on a toy dataset).

**Pass 4 — End-to-end demo and synthesis**
- Build a small synthetic dataset generator (random images + ground-truth task labels).
- Run inference on multiple tasks with text instructions (e.g., "depth estimation", "surface normals").
- Create a small demo script that visualizes outputs from each task on toy images.
- Update NOTES.md with a final summary of what works, what is simplified, and what assumptions were made.

## Implemented vs. Simplified

**Pass 1 — What is implemented:**
- Text instruction embedding module using learned word embeddings (vocabulary size 1000, embed dim 128) with mean-pooling over token sequence to produce fixed-size instruction vector.
- Simplified UNet-like backbone with 3 downsampling blocks (using convolution + batch norm + ReLU), a bottleneck block, and symmetric 3 upsampling blocks. Architecture preserves spatial dimensions (H, W) throughout.
- Instruction conditioning via feature modulation: instruction vector is projected to feature channel space and multiplied with image features (acts as learned scaling/shifting per image).
- Full forward pass: image (B, 3, H, W) + token IDs (B, seq_len) → instruction-conditioned features (B, base_channels, H, W).
- Comprehensive test suite verifying: text embedding shapes, UNet feature extraction, instruction conditioning effects, end-to-end forward pass, that different instructions produce different outputs, and deterministic behavior in eval mode.

**Simplified/stubbed for Pass 1:**
- Text tokenization: model receives pre-tokenized integer sequences. Real tokenization (BPE, word-level, etc.) is stubbed; test uses random token IDs.
- No actual diffusion backbone: the "UNet" is a toy architecture with only 3 downsampling levels and no skip connections. A real video-generation pre-trained model is not loaded.
- Instruction conditioning is multiplication-based (feature modulation). The paper likely uses more sophisticated conditioning mechanisms (cross-attention, etc.); this is simplified to element-wise scaling.
- No task-specific heads or output decoders. Model outputs only the conditioned feature map; subsequent passes will add task-specific heads for depth, normals, segmentation, etc.
- No training loop. Model is initialized but not trained on any data; gradients are not demonstrated.
- Synthetic test data only. Real images and instructions are not used; all tests use random tensors.

**Key assumptions:**
- Instruction vector has fixed dimensionality regardless of instruction length (mean pooling loses sequential information, but is simple and effective for Pass 1).
- Conditioning is applied uniformly across all spatial locations (multiplication, not selective attention).
- Base architecture (num_blocks=3) is very small; real models would be much deeper and wider.

**Pass 2 — What is implemented:**
- Three task-specific decoder heads:
  - DepthHead: takes conditioned features and outputs (B, 1, H, W) single-channel depth map
  - NormalsHead: outputs (B, 3, H, W) RGB surface normals prediction
  - SegmentationHead: outputs (B, num_classes, H, W) semantic segmentation logits
- TaskSelector module: embeds instruction vector into 3-way task logits, selects task via argmax (depth, normals, or segmentation)
- Modified GenCeption.forward() to support multi-task decoding:
  - return_all_tasks=False (default): returns conditioned features for backward compatibility with Pass 1 tests
  - return_all_tasks=True: returns dictionary with task_id, task_logits, depth, normals, segmentation outputs
- All decoder heads use simple 2-layer ConvBlock architecture (conv → conv → final projection)
- Comprehensive test suite verifying: individual head output shapes, task selection mechanism, multi-task output correctness, task selection variation across instructions, deterministic outputs in eval mode

**Simplified/stubbed for Pass 2:**
- Task selection is purely learned from instruction embeddings via a simple linear projection to 3 logits. Real implementation might use:
  - Semantic understanding of task keywords ("depth", "normal", "segment")
  - Hierarchical task grouping or soft task mixing
  - Learned task routing with mixture-of-experts
- Decoder heads are minimal (only 2 conv blocks). Real implementation would use:
  - Skip connections from intermediate backbone features
  - Progressive upsampling with larger feature maps
  - Task-specific architectural innovations (e.g., disentangled depth decoders, normal estimation via orientation regression)
- No task-specific loss functions or training. All heads trained equally in forward pass; no actual gradient updates demonstrated.
- No cross-task feature sharing or task-agnostic refinement (that's Pass 3).
- Synthetic test data only; no real image, instruction, or task label data.

**Key assumptions for Pass 2:**
- Task is fully determined by instruction (hard routing, not soft mixing of multiple tasks).
- All tasks share the same conditioned feature representation (no task-specific feature extraction).
- Segmentation uses num_classes=10 fixed; real segmentation would vary by dataset.
- Decoder depth and width (64 hidden channels) is fixed and small; larger models would use task-specific depths.

**Pass 3 — What is implemented:**
- TaskAgnosticRefiner module: a shared refinement pipeline applied after instruction conditioning and before task-specific heads. The refiner consists of 2 configurable convolutional blocks (ConvBlock) that perform task-agnostic feature processing.
- Integration into GenCeption: the refiner is applied in the forward pass after conditioning, so all outputs (depth, normals, segmentation) use refined features. This allows the model to learn task-agnostic feature transformations that benefit all downstream tasks.
- SyntheticGenCeptionDataset: a PyTorch Dataset that generates random images and instructions for training experiments.
- train_genception_step function: performs one gradient update step with synthetic targets. Generates task-agnostic synthetic targets for depth (random [0, 10]), normals (random unit vectors), and segmentation (random class labels), then computes and backpropagates multi-task loss (average of depth MSE, normals MSE, and segmentation cross-entropy).
- Comprehensive Pass 3 tests: verifies refiner output shapes, that refinement modifies features, backward compatibility with refined features, dataset generation, parameter updates during training, and a mini training loop that runs multiple batches.

**Simplified/stubbed for Pass 3:**
- Task-agnostic refinement is implemented as simple sequential convolutional blocks (no skip connections, attention, or more sophisticated architectures).
- Training loss is synthetic (generated on-the-fly): real scenarios would use actual ground-truth labels from a dataset (e.g., NYU Depth, Cityscapes).
- No learning rate scheduling, batch normalization momentum tuning, or other training hyperparameter optimization.
- No validation or metrics tracking during training (only loss values are reported).
- Refiner is applied uniformly to all features; the paper may use adaptive refinement that depends on instruction or task.

**Key assumptions for Pass 3:**
- Task-agnostic refinement (shared across all tasks) improves feature quality without task-specific specialization.
- Convolutional blocks are sufficient for refinement; more sophisticated operators (attention, gating, etc.) are not needed for this demonstration.
- Synthetic targets are adequate to demonstrate gradient flow and parameter updates; real training would require proper datasets.
