# YOLOv10: Real-Time End-to-End Object Detection

**arXiv:** https://arxiv.org/abs/2405.14458

**Authors:** Ao Wang, Hui Chen, Lihao Liu, Kunchang Li, Z.X. Lin, Peng Wang, Cheng Cui (Tsinghua University)

**Published:** May 2024

## Summary

YOLOv10 proposes a modern approach to real-time object detection by eliminating Non-Maximum Suppression (NMS) during inference and making end-to-end trainable detection heads that predict objects without post-processing. The core innovation is replacing the traditional pipeline (model output → NMS) with a unified model that learns to output non-overlapping predictions directly. This architectural shift significantly reduces inference latency while maintaining or improving accuracy.

## Plan: 4 passes

**Pass 1:** Implement a basic single-scale object detection head that predicts bounding boxes and class scores without NMS. The foundation will be a simple backbone feature extractor (minimal CNN) and a detection head that processes multi-scale features.

**Pass 2:** Implement the dual-head design (one for regression, one for classification) with decoupled predictions that YOLOv10 uses, plus training loss function (focal loss or similar for handling class imbalance).

**Pass 3:** Add end-to-end trainability optimizations and training procedure with data augmentation on toy COCO or synthetic dataset.

**Pass 4:** End-to-end demo on small image samples showing inference without NMS, speed benchmarks, and honest summary of what was simplified.

## Implemented vs. Simplified (Pass 1)

### Implemented:
- **SimpleBackbone**: 4-layer CNN that downsamples input by 16x (stride 2 at each layer)
- **DetectionHead**: Predicts bounding boxes (4 coords) and class logits (80 classes) for 3 anchors per spatial location
- **Forward pass**: Takes RGB image (any H×W), outputs predictions of shape (B, num_predictions, 4) for boxes and (B, num_predictions, num_classes) for classes
- **End-to-end differentiability**: Full gradient flow for future training implementations

### Simplified/Stubbed:
- **No actual anchor assignment**: Predictions are raw (not tied to actual anchor boxes or targets yet)
- **No loss function**: Pass 1 only covers forward pass; loss will come in Pass 2
- **No NMS removal logic**: That optimization comes later; for now this is just a standard dense prediction head
- **No multi-scale detection**: Single feature map from backbone; real YOLOv10 uses multi-scale FPN
- **Minimal backbone**: 4 Conv blocks instead of real CSPDarknet backbone
- **Fixed architecture**: No scaling variants (N, S, M, etc. as in real YOLOv10)

### Tests:
- Forward pass with correct output shapes for different input sizes (320, 416, 640)
- Gradient flow verification
- No NaN checks for numeric stability

## Implemented vs. Simplified (Pass 2)

### Implemented:
- **DecoupledDetectionHead**: Independent feature processing pathways for bbox regression and classification (YOLOv10's key architectural innovation)
  - Bbox pathway: 2 ConvBlocks (128→96→96) + Conv head
  - Cls pathway: 2 ConvBlocks (128→96→96) + Conv head
  - Both pathways process features independently for better feature specialization
- **FocalLoss**: Handles class imbalance with alpha=0.25, gamma=2.0 (reduces easy negatives, focuses on hard examples)
- **Target matching**: Greedy one-to-one assignment via IoU-based matching (each prediction matched to at most one target)
  - compute_iou(): Computes intersection-over-union between predicted and target boxes
  - match_predictions_to_targets(): Greedy matching algorithm with configurable IoU threshold
- **Loss computation**: YOLOv10DetectorPass2.compute_loss()
  - Smooth L1 loss for bbox regression on matched predictions
  - Focal loss for classification on matched predictions
  - Per-batch aggregation
- **Training-capable model**: YOLOv10DetectorPass2 supports forward pass with optional targets dict for loss computation
- **Comprehensive tests**: 7 test functions covering decoupled head, feature independence, focal loss, matching, and gradient flow

### Simplified/Stubbed:
- **Greedy matching only**: Real YOLOv10 uses more sophisticated Hungarian algorithm or cost-based assignment; we use simple greedy by IoU
- **Fixed IoU threshold**: Matching threshold hardcoded at 0.5; real YOLOv10 may use adaptive thresholds or simpler positive/negative assignment
- **No anchor boxes**: Predictions don't explicitly tie to anchor definitions; just dense spatial predictions
- **No NMS post-processing**: Model outputs raw predictions; NMS removal is a Pass 4 focus (though not yet implemented here)
- **No multi-scale**: Still using single feature map; real YOLOv10 uses FPN for multi-scale detection
- **No data augmentation or actual training**: Loss can be computed but no optimizer loop yet (Pass 3)
- **No end-to-end efficiency tricks**: No knowledge distillation, pruning, or quantization awareness

### Tests:
- Decoupled head forward pass output shapes (2, 4800, 4) and (2, 4800, 80) ✓
- Feature pathway independence verification (distinct gradients per path) ✓
- Focal loss computation and gradient flow ✓
- Target matching with multiple predictions and targets ✓
- Pass 2 model with targets dict producing valid loss ✓
- Inference mode (no loss when targets absent) ✓
- Backward compatibility: Pass 1 tests still pass ✓
