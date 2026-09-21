import torch
from detection_head import YOLOv10Detector


def test_model_forward_pass():
    """Test that the model produces correct output shapes."""
    batch_size = 2
    num_classes = 80
    input_height, input_width = 640, 640

    model = YOLOv10Detector(num_classes=num_classes)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, input_height, input_width)

    with torch.no_grad():
        outputs = model(dummy_input)

    bbox_pred = outputs['bbox']
    cls_pred = outputs['cls']

    print(f"Input shape: {dummy_input.shape}")
    print(f"Bbox predictions shape: {bbox_pred.shape}")
    print(f"Class predictions shape: {cls_pred.shape}")

    expected_num_predictions = (input_height // 16) * (input_width // 16) * 3
    assert bbox_pred.shape == (batch_size, expected_num_predictions, 4), f"Expected bbox shape (2, {expected_num_predictions}, 4), got {bbox_pred.shape}"
    assert cls_pred.shape == (batch_size, expected_num_predictions, num_classes), f"Expected cls shape (2, {expected_num_predictions}, 80), got {cls_pred.shape}"

    assert bbox_pred.dtype == torch.float32
    assert cls_pred.dtype == torch.float32

    assert not torch.isnan(bbox_pred).any(), "NaN values in bbox predictions"
    assert not torch.isnan(cls_pred).any(), "NaN values in cls predictions"

    print("✓ Forward pass test passed!")


def test_model_trainable():
    """Test that gradients flow through the model."""
    batch_size = 1
    num_classes = 80

    model = YOLOv10Detector(num_classes=num_classes)
    model.train()

    dummy_input = torch.randn(batch_size, 3, 640, 640, requires_grad=True)
    outputs = model(dummy_input)

    loss = outputs['bbox'].sum() + outputs['cls'].sum()
    loss.backward()

    assert dummy_input.grad is not None, "Gradients not flowing to input"
    assert any(p.grad is not None for p in model.parameters()), "Gradients not flowing to model parameters"

    print("✓ Gradient flow test passed!")


def test_different_input_sizes():
    """Test model with different input sizes."""
    model = YOLOv10Detector(num_classes=80)
    model.eval()

    sizes = [(320, 320), (416, 416), (640, 640)]

    with torch.no_grad():
        for h, w in sizes:
            dummy_input = torch.randn(1, 3, h, w)
            outputs = model(dummy_input)

            assert outputs['bbox'].dim() == 3, f"Bbox should be 3D, got {outputs['bbox'].dim()}D"
            assert outputs['cls'].dim() == 3, f"Class should be 3D, got {outputs['cls'].dim()}D"

            print(f"✓ Input size {h}x{w} works - bbox: {outputs['bbox'].shape}, cls: {outputs['cls'].shape}")


if __name__ == '__main__':
    test_model_forward_pass()
    test_model_trainable()
    test_different_input_sizes()
    print("\n✓ All tests passed!")
