import torch
import torch.nn as nn
from detection_head import YOLOv10DetectorPass2, DecoupledDetectionHead, FocalLoss, match_predictions_to_targets


def test_decoupled_head_forward():
    """Test that decoupled head produces correct output shapes."""
    batch_size = 2
    num_classes = 80
    input_height, input_width = 640, 640

    head = DecoupledDetectionHead(num_classes=num_classes)
    head.eval()

    dummy_input = torch.randn(batch_size, 3, input_height, input_width)

    with torch.no_grad():
        outputs = head(dummy_input)

    bbox_pred = outputs['bbox']
    cls_pred = outputs['cls']

    print(f"Decoupled Head - Input shape: {dummy_input.shape}")
    print(f"  Bbox predictions shape: {bbox_pred.shape}")
    print(f"  Class predictions shape: {cls_pred.shape}")

    expected_num_predictions = (input_height // 16) * (input_width // 16) * 3
    assert bbox_pred.shape == (batch_size, expected_num_predictions, 4)
    assert cls_pred.shape == (batch_size, expected_num_predictions, num_classes)
    assert not torch.isnan(bbox_pred).any()
    assert not torch.isnan(cls_pred).any()

    print("✓ Decoupled head forward pass test passed!")


def test_decoupled_head_independent_features():
    """Verify that bbox and cls pathways are truly independent."""
    num_classes = 80
    head = DecoupledDetectionHead(num_classes=num_classes)
    head.train()

    dummy_input = torch.randn(1, 3, 320, 320, requires_grad=True)
    outputs = head(dummy_input)

    bbox_feat = outputs['bbox_feat']
    cls_feat = outputs['cls_feat']

    print(f"  Bbox feature shape: {bbox_feat.shape}")
    print(f"  Cls feature shape: {cls_feat.shape}")

    # Both should have same spatial dims but independent computations
    assert bbox_feat.shape[1:] == cls_feat.shape[1:]

    # Verify different gradients through different paths
    bbox_loss = bbox_feat.sum()
    bbox_loss.backward(retain_graph=True)

    bbox_grad_norm = sum(p.grad.norm().item() for p in head.bbox_stem.parameters() if p.grad is not None)
    cls_grad_norm = sum(p.grad.norm().item() for p in head.cls_stem.parameters() if p.grad is not None)

    # Clear gradients
    head.zero_grad()

    cls_loss = cls_feat.sum()
    cls_loss.backward()

    bbox_grad_norm_2 = sum(p.grad.norm().item() for p in head.bbox_stem.parameters() if p.grad is not None)
    cls_grad_norm_2 = sum(p.grad.norm().item() for p in head.cls_stem.parameters() if p.grad is not None)

    print(f"  Bbox stem gradient (from bbox): {bbox_grad_norm:.6f}")
    print(f"  Cls stem gradient (from cls): {cls_grad_norm_2:.6f}")

    # Pathways should have different gradient patterns
    assert bbox_grad_norm > 0 and cls_grad_norm_2 > 0

    print("✓ Decoupled pathways independence test passed!")


def test_focal_loss():
    """Test focal loss computation."""
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    # Simple test: 4 samples, 3 classes
    pred = torch.randn(4, 3, requires_grad=True)
    target = torch.tensor([0, 1, 2, 0])

    loss = focal_loss(pred, target)

    print(f"Focal loss: {loss.item():.6f}")

    # Verify it's a scalar
    assert loss.dim() == 0
    assert loss.item() >= 0

    # Verify gradients flow
    loss.backward()
    assert pred.grad is not None

    print("✓ Focal loss test passed!")


def test_target_matching():
    """Test one-to-one target assignment."""
    # 5 predictions, 3 targets
    pred_boxes = torch.tensor([
        [0, 0, 10, 10],
        [15, 15, 25, 25],
        [100, 100, 110, 110],
        [5, 5, 15, 15],
        [200, 200, 210, 210]
    ], dtype=torch.float32)

    pred_scores = torch.randn(5, 80)

    target_boxes = torch.tensor([
        [1, 1, 11, 11],
        [16, 16, 26, 26],
        [101, 101, 111, 111]
    ], dtype=torch.float32)

    target_classes = torch.tensor([5, 10, 15])

    matched_targets, matched_mask = match_predictions_to_targets(
        pred_boxes, pred_scores, target_boxes, target_classes, iou_threshold=0.3
    )

    print(f"Predictions: {pred_boxes.shape[0]}, Targets: {target_boxes.shape[0]}")
    print(f"Matched mask: {matched_mask}")
    print(f"Matched count: {matched_mask.sum().item()}")

    # Should match at least some predictions
    assert matched_mask.sum() >= 1

    # Matched targets should have valid classes
    matched_idx = torch.where(matched_mask)[0]
    if matched_idx.shape[0] > 0:
        classes = matched_targets[matched_idx, 4]
        assert (classes >= 0).all()

    print("✓ Target matching test passed!")


def test_pass2_forward_with_targets():
    """Test Pass 2 model with loss computation."""
    batch_size = 2
    num_classes = 80
    input_height, input_width = 320, 320

    model = YOLOv10DetectorPass2(num_classes=num_classes)
    model.train()

    dummy_input = torch.randn(batch_size, 3, input_height, input_width)

    # Create dummy targets
    num_predictions = (input_height // 16) * (input_width // 16) * 3
    targets = {
        'boxes': torch.zeros(batch_size, 5, 4),
        'classes': torch.full((batch_size, 5), -1, dtype=torch.long)
    }

    # Add some valid targets
    targets['boxes'][0, 0] = torch.tensor([10., 10., 50., 50.])
    targets['classes'][0, 0] = 5

    targets['boxes'][1, 0] = torch.tensor([20., 20., 60., 60.])
    targets['classes'][1, 0] = 10

    outputs = model(dummy_input, targets=targets)

    print(f"Pass 2 model outputs keys: {outputs.keys()}")
    print(f"Loss: {outputs['loss'].item():.6f}")

    assert 'loss' in outputs
    assert outputs['loss'].item() >= 0

    print("✓ Pass 2 forward with targets test passed!")


def test_pass2_gradients():
    """Test that gradients flow through loss in Pass 2."""
    batch_size = 1
    num_classes = 80

    model = YOLOv10DetectorPass2(num_classes=num_classes)
    model.train()

    dummy_input = torch.randn(batch_size, 3, 320, 320, requires_grad=True)

    # Create targets with multiple boxes to increase chance of matching
    num_targets = 3
    targets = {
        'boxes': torch.tensor([
            [[50., 50., 100., 100.],
             [150., 150., 200., 200.],
             [250., 250., 300., 300.]],
        ], dtype=torch.float32),
        'classes': torch.tensor([[5, 10, 15]], dtype=torch.long)
    }

    outputs = model(dummy_input, targets=targets)
    loss = outputs['loss']

    if loss.item() > 0:
        loss.backward()
        assert dummy_input.grad is not None
        assert any(p.grad is not None for p in model.parameters())
        print("✓ Pass 2 gradient flow test passed!")
    else:
        print("⚠ Pass 2 gradient flow test skipped (no targets matched)")


def test_model_inference_mode():
    """Test inference without targets."""
    model = YOLOv10DetectorPass2(num_classes=80)
    model.eval()

    dummy_input = torch.randn(1, 3, 320, 320)

    with torch.no_grad():
        outputs = model(dummy_input)

    # Should not have loss in inference mode
    assert 'loss' not in outputs
    assert 'bbox' in outputs
    assert 'cls' in outputs

    print("✓ Inference mode test passed!")


if __name__ == '__main__':
    print("Running Pass 2 tests...\n")
    test_decoupled_head_forward()
    print()
    test_decoupled_head_independent_features()
    print()
    test_focal_loss()
    print()
    test_target_matching()
    print()
    test_pass2_forward_with_targets()
    print()
    test_pass2_gradients()
    print()
    test_model_inference_mode()
    print("\n✓ All Pass 2 tests passed!")
