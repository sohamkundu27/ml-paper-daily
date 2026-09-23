"""
Tests for Pass 4: Inference engine and end-to-end functionality
"""
import torch
import time
from detection_head import YOLOv10DetectorPass3
from pass_4_inference_demo import InferenceEngine
from train_pass3 import SyntheticObjectDataset, TrainingUtils, train_model


def test_inference_engine_creation():
    """Test InferenceEngine initialization."""
    device = torch.device('cpu')
    model = YOLOv10DetectorPass3(num_classes=80)
    engine = InferenceEngine(model, device=device)

    assert engine.model is not None
    assert engine.device == device
    assert not engine.model.training  # Should be in eval mode

    print("✓ Inference engine creation test passed!")


def test_inference_engine_predict():
    """Test inference engine prediction."""
    device = torch.device('cpu')
    model = YOLOv10DetectorPass3(num_classes=80)
    engine = InferenceEngine(model, device=device)

    # Test single image (3D tensor)
    image_3d = torch.randn(3, 320, 320)
    outputs = engine.predict(image_3d)

    assert 'bbox' in outputs
    assert 'cls' in outputs
    assert outputs['bbox'].shape[0] == 1  # Should add batch dimension

    # Test batch (4D tensor)
    image_4d = torch.randn(2, 3, 320, 320)
    outputs = engine.predict(image_4d)

    assert outputs['bbox'].shape[0] == 2
    assert outputs['cls'].shape[0] == 2

    print("✓ Inference engine predict test passed!")


def test_filter_predictions():
    """Test prediction filtering by confidence threshold."""
    device = torch.device('cpu')
    model = YOLOv10DetectorPass3(num_classes=10)
    engine = InferenceEngine(model, device=device)

    image = torch.randn(1, 3, 320, 320)
    outputs = engine.predict(image)

    # Filter with high threshold
    predictions_high = engine.filter_predictions(outputs, conf_threshold=0.9)

    # Filter with low threshold
    predictions_low = engine.filter_predictions(outputs, conf_threshold=0.1)

    # Lower threshold should give more predictions
    assert len(predictions_low) >= len(predictions_high)

    # All predictions should have required fields
    for pred in predictions_high:
        assert 'box' in pred
        assert 'class' in pred
        assert 'score' in pred
        assert 0 <= pred['score'] <= 1

    print(f"  High threshold (0.9): {len(predictions_high)} predictions")
    print(f"  Low threshold (0.1):  {len(predictions_low)} predictions")
    print("✓ Prediction filtering test passed!")


def test_benchmark_inference():
    """Test inference benchmarking."""
    device = torch.device('cpu')
    model = YOLOv10DetectorPass3(num_classes=20)
    engine = InferenceEngine(model, device=device)

    stats = engine.benchmark_inference(image_size=320, num_images=10)

    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'total' in stats

    assert stats['mean'] > 0
    assert stats['std'] >= 0
    assert stats['min'] > 0
    assert stats['max'] > 0
    assert stats['min'] <= stats['mean'] <= stats['max']

    print(f"  Mean latency: {stats['mean']:.2f}ms")
    print(f"  Std dev: {stats['std']:.2f}ms")
    print(f"  FPS: {1000/stats['mean']:.1f}")
    print("✓ Inference benchmarking test passed!")


def test_inference_no_nan():
    """Test that inference doesn't produce NaN values."""
    device = torch.device('cpu')
    model = YOLOv10DetectorPass3(num_classes=80)
    engine = InferenceEngine(model, device=device)

    for _ in range(5):
        image = torch.randn(1, 3, 320, 320)
        outputs = engine.predict(image)

        # Check for NaN in outputs
        assert not torch.isnan(outputs['bbox']).any(), "NaN found in bbox predictions"
        assert not torch.isnan(outputs['cls']).any(), "NaN found in class predictions"

    print("✓ No NaN inference test passed!")


def test_inference_with_trained_model():
    """Test inference on a briefly trained model."""
    device = torch.device('cpu')

    # Train a small model
    print("  Training small model...")
    model, history = train_model(
        num_epochs=1,
        batch_size=4,
        learning_rate=0.01,
        num_train_samples=8,
        num_val_samples=4,
        img_size=320,
        num_classes=10,
        device=device,
    )

    engine = InferenceEngine(model, device=device)

    # Create test batch
    test_dataset = SyntheticObjectDataset(
        num_samples=2,
        img_size=320,
        num_classes=10,
        augment=False
    )

    # Inference on test samples
    for idx in range(len(test_dataset)):
        item = test_dataset[idx]
        image = item['image'].unsqueeze(0)

        outputs = engine.predict(image)
        predictions = engine.filter_predictions(outputs, conf_threshold=0.3)

        # Just verify we get predictions back
        assert isinstance(predictions, list)

    print("✓ Inference with trained model test passed!")


def test_batch_inference():
    """Test inference on batched images."""
    device = torch.device('cpu')
    model = YOLOv10DetectorPass3(num_classes=20)
    engine = InferenceEngine(model, device=device)

    batch_size = 4
    images = torch.randn(batch_size, 3, 320, 320)

    outputs = engine.predict(images)

    # Check output shapes
    assert outputs['bbox'].shape[0] == batch_size
    assert outputs['cls'].shape[0] == batch_size

    # Filter predictions from batch
    all_predictions = engine.filter_predictions(outputs, conf_threshold=0.5)

    # Should have some predictions (though might be empty with random untrained model)
    assert isinstance(all_predictions, list)

    print(f"  Batch size: {batch_size}")
    print(f"  Total filtered predictions: {len(all_predictions)}")
    print("✓ Batch inference test passed!")


def test_different_image_sizes():
    """Test inference with different image sizes."""
    device = torch.device('cpu')
    model = YOLOv10DetectorPass3(num_classes=80)
    engine = InferenceEngine(model, device=device)

    sizes = [256, 320, 416, 512]

    for size in sizes:
        image = torch.randn(1, 3, size, size)
        outputs = engine.predict(image)

        # Predictions should exist
        assert outputs['bbox'].shape[0] == 1
        assert outputs['cls'].shape[0] == 1

        # Number of spatial predictions scales with input size
        expected_num_predictions = (size // 16) * (size // 16) * 3
        assert outputs['bbox'].shape[1] == expected_num_predictions

    print("✓ Different image sizes test passed!")


def test_inference_speed_comparison():
    """Compare inference speed across different batch sizes."""
    device = torch.device('cpu')
    model = YOLOv10DetectorPass3(num_classes=20)
    engine = InferenceEngine(model, device=device)

    print("  Inference speed by batch size:")

    for batch_size in [1, 2, 4]:
        times = []

        for _ in range(5):
            image = torch.randn(batch_size, 3, 320, 320)

            start = time.time()
            _ = engine.predict(image)
            elapsed = time.time() - start

            times.append(elapsed)

        avg_time = sum(times) / len(times)
        time_per_image = (avg_time * 1000) / batch_size

        print(f"    Batch {batch_size}: {time_per_image:.2f}ms per image")

    print("✓ Inference speed comparison test passed!")


if __name__ == '__main__':
    print("Running Pass 4 tests...\n")

    test_inference_engine_creation()
    print()

    test_inference_engine_predict()
    print()

    test_filter_predictions()
    print()

    test_inference_no_nan()
    print()

    test_benchmark_inference()
    print()

    test_batch_inference()
    print()

    test_different_image_sizes()
    print()

    test_inference_speed_comparison()
    print()

    test_inference_with_trained_model()
    print()

    print("✓ All Pass 4 tests passed!")
