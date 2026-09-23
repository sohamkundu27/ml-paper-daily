"""
Pass 4: End-to-end YOLOv10 inference demo without NMS

This demo shows:
1. Brief training on synthetic data (2 epochs)
2. Inference on test images
3. Speed benchmarking (no NMS post-processing)
4. Visualization of predictions
"""
import torch
import torch.nn as nn
import time
from detection_head import YOLOv10DetectorPass3
from train_pass3 import (
    SyntheticObjectDataset,
    TrainingUtils,
    train_model,
)


class InferenceEngine:
    """Lightweight inference engine for Pass 4 model."""
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_tensor):
        """
        Run inference on a single image or batch.

        Args:
            image_tensor: (B, 3, H, W) or (3, H, W) tensor

        Returns:
            predictions: dict with 'bbox' and 'cls' keys
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        return outputs

    def filter_predictions(self, outputs, conf_threshold=0.5):
        """
        Filter predictions by confidence threshold (no NMS applied).

        Args:
            outputs: model outputs dict
            conf_threshold: minimum class confidence to keep prediction

        Returns:
            filtered_boxes: list of (x1, y1, x2, y2, class, confidence)
        """
        bbox_pred = outputs['bbox']  # (B, N, 4)
        cls_pred = outputs['cls']    # (B, N, 80)

        results = []
        batch_size = bbox_pred.shape[0]

        for b in range(batch_size):
            boxes = bbox_pred[b]  # (N, 4)
            classes = cls_pred[b]  # (N, 80)

            # Get max class score and class index for each prediction
            class_scores, class_indices = classes.max(dim=1)

            # Filter by confidence threshold
            valid_mask = class_scores >= conf_threshold
            valid_boxes = boxes[valid_mask]
            valid_classes = class_indices[valid_mask]
            valid_scores = class_scores[valid_mask]

            for box, cls_idx, score in zip(valid_boxes, valid_classes, valid_scores):
                results.append({
                    'box': box.detach().cpu().numpy(),
                    'class': cls_idx.item(),
                    'score': score.item(),
                })

        return results

    def benchmark_inference(self, image_size=320, num_images=100):
        """
        Benchmark inference speed on random images.

        Args:
            image_size: image height/width
            num_images: number of images to process

        Returns:
            dict with timing statistics
        """
        times = []

        print(f"Benchmarking inference on {num_images} images ({image_size}x{image_size})...")

        with torch.no_grad():
            for _ in range(num_images):
                image = torch.randn(1, 3, image_size, image_size)

                start = time.time()
                _ = self.predict(image)
                elapsed = time.time() - start

                times.append(elapsed)

        times = torch.tensor(times)

        stats = {
            'mean': times.mean().item() * 1000,  # ms
            'std': times.std().item() * 1000,
            'min': times.min().item() * 1000,
            'max': times.max().item() * 1000,
            'total': times.sum().item(),
        }

        return stats


def demo_end_to_end():
    """Run complete end-to-end demo: train, infer, benchmark."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("YOLOv10 Pass 4: End-to-End Inference Demo (No NMS)")
    print("=" * 70)
    print()

    # Step 1: Train model briefly
    print("STEP 1: Training on synthetic data (2 epochs, 32 samples)")
    print("-" * 70)

    model, history = train_model(
        num_epochs=2,
        batch_size=8,
        learning_rate=0.01,
        num_train_samples=32,
        num_val_samples=8,
        img_size=320,
        num_classes=20,
        device=device,
    )

    print()
    print("Training complete!")
    print("Final train loss:", f"{history['train_loss'][-1]:.4f}")
    print("Final val loss:  ", f"{history['val_loss'][-1]:.4f}")
    print()

    # Step 2: Create inference engine
    print("STEP 2: Setting up inference engine")
    print("-" * 70)

    engine = InferenceEngine(model, device=device)
    print(f"Model moved to {device}")
    print("Inference mode: ON (no gradient computation)")
    print()

    # Step 3: Run inference on test batch
    print("STEP 3: Inference on synthetic test images")
    print("-" * 70)

    test_dataset = SyntheticObjectDataset(
        num_samples=5,
        img_size=320,
        num_classes=20,
        max_objects=3,
        augment=False
    )

    for idx in range(min(3, len(test_dataset))):
        item = test_dataset[idx]
        image = item['image'].unsqueeze(0)

        # Run inference
        outputs = engine.predict(image)

        # Filter predictions
        predictions = engine.filter_predictions(outputs, conf_threshold=0.3)

        print(f"  Image {idx+1}:")
        print(f"    Total predictions: {outputs['bbox'].shape[1]}")
        print(f"    High-confidence predictions (>0.3): {len(predictions)}")

        if len(predictions) > 0:
            print(f"    Top predictions:")
            for pred_idx, pred in enumerate(predictions[:3]):
                box = pred['box']
                print(f"      - Class {pred['class']}, "
                      f"score {pred['score']:.3f}, "
                      f"box [{box[0]:.1f}, {box[1]:.1f}, "
                      f"{box[2]:.1f}, {box[3]:.1f}]")
        print()

    # Step 4: Speed benchmark
    print("STEP 4: Inference speed benchmark (no NMS)")
    print("-" * 70)

    for img_size in [320, 416]:
        stats = engine.benchmark_inference(image_size=img_size, num_images=50)
        print(f"  Image size {img_size}x{img_size}:")
        print(f"    Mean latency: {stats['mean']:.2f}ms")
        print(f"    Std dev:      {stats['std']:.2f}ms")
        print(f"    Min:          {stats['min']:.2f}ms")
        print(f"    Max:          {stats['max']:.2f}ms")
        print(f"    FPS (approx): {1000/stats['mean']:.1f}")
        print()

    # Step 5: Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key observations:
1. Model outputs predictions WITHOUT post-processing (no NMS applied)
2. Each prediction includes bbox coordinates and class confidence
3. Raw model predictions are filtered by simple confidence threshold
4. Direct inference from model eliminates NMS latency overhead
5. End-to-end differentiability enables joint training of all components

YOLOv10 Innovation:
- Traditional YOLO: Dense predictions → NMS (post-processing)
- YOLOv10:       Dense predictions → Minimal filtering

This design enables:
✓ Faster inference (no NMS computation)
✓ Simpler deployment (fewer post-processing steps)
✓ End-to-end training optimization
✓ Direct gradient flow through entire pipeline
""")
    print("=" * 70)
    print()


def demo_inference_modes():
    """Show different inference configurations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 70)
    print("YOLOv10 Pass 4: Inference Mode Comparison")
    print("=" * 70)
    print()

    # Quick model train
    print("Quick model training for demo...")
    model, _ = train_model(
        num_epochs=1,
        batch_size=8,
        learning_rate=0.01,
        num_train_samples=16,
        num_val_samples=4,
        img_size=320,
        num_classes=10,
        device=device,
    )
    print()

    engine = InferenceEngine(model, device=device)
    test_image = torch.randn(1, 3, 320, 320)

    # Mode 1: Raw predictions
    print("MODE 1: Raw model predictions (no filtering)")
    print("-" * 70)
    outputs = engine.predict(test_image)
    print(f"  Output shape (bboxes): {outputs['bbox'].shape}")
    print(f"  Output shape (classes): {outputs['cls'].shape}")
    print()

    # Mode 2: Confidence filtering
    print("MODE 2: Confidence filtering (threshold=0.5)")
    print("-" * 70)
    predictions = engine.filter_predictions(outputs, conf_threshold=0.5)
    print(f"  Predictions with score > 0.5: {len(predictions)}")
    print()

    # Mode 3: Lower threshold
    print("MODE 3: Lower confidence threshold (threshold=0.1)")
    print("-" * 70)
    predictions = engine.filter_predictions(outputs, conf_threshold=0.1)
    print(f"  Predictions with score > 0.1: {len(predictions)}")
    print()

    print("=" * 70)
    print()


if __name__ == '__main__':
    # Run main demo
    demo_end_to_end()

    # Show inference modes
    demo_inference_modes()

    print("✓ Pass 4 inference demo complete!")
