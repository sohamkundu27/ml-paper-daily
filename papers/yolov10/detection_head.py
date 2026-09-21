import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SimpleBackbone(nn.Module):
    """Minimal CNN backbone for feature extraction."""
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBlock(3, 32, stride=2)
        self.stage2 = ConvBlock(32, 64, stride=2)
        self.stage3 = ConvBlock(64, 128, stride=2)
        self.stage4 = ConvBlock(128, 256, stride=2)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class DetectionHead(nn.Module):
    """
    Simple object detection head that outputs bounding boxes and class scores.
    Predicts: (batch_size, num_anchors, 4 + num_classes) for bboxes and class logits.
    """
    def __init__(self, feature_dim=256, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.backbone = SimpleBackbone()

        self.reduction = ConvBlock(256, 128)

        self.bbox_conv = nn.Sequential(
            ConvBlock(128, 64),
            nn.Conv2d(64, num_anchors * 4, 1)
        )

        self.cls_conv = nn.Sequential(
            ConvBlock(128, 64),
            nn.Conv2d(64, num_anchors * num_classes, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.reduction(features)

        bbox_pred = self.bbox_conv(features)
        cls_pred = self.cls_conv(features)

        batch_size, _, height, width = bbox_pred.shape

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.view(batch_size, height * width * self.num_anchors, 4)

        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
        cls_pred = cls_pred.view(batch_size, height * width * self.num_anchors, self.num_classes)

        return {
            'bbox': bbox_pred,
            'cls': cls_pred,
            'features': features
        }


class YOLOv10Detector(nn.Module):
    """End-to-end object detection model (Pass 1: basic version)."""
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.detection_head = DetectionHead(num_classes=num_classes)

    def forward(self, x):
        outputs = self.detection_head(x)
        return outputs


class DecoupledDetectionHead(nn.Module):
    """
    Pass 2: Decoupled detection heads with separate feature processing for
    regression (bbox) and classification.

    YOLOv10 key innovation: Instead of sharing features between bbox and class
    predictions, use independent pathways. This allows better feature specialization.
    """
    def __init__(self, feature_dim=256, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.backbone = SimpleBackbone()

        self.reduction = ConvBlock(256, 128)

        # Regression (bbox) head - independent pathway
        self.bbox_stem = nn.Sequential(
            ConvBlock(128, 96),
            ConvBlock(96, 96),
        )
        self.bbox_head = nn.Conv2d(96, num_anchors * 4, 1)

        # Classification head - independent pathway
        self.cls_stem = nn.Sequential(
            ConvBlock(128, 96),
            ConvBlock(96, 96),
        )
        self.cls_head = nn.Conv2d(96, num_anchors * num_classes, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.reduction(features)

        # Decoupled regression pathway
        bbox_feat = self.bbox_stem(features)
        bbox_pred = self.bbox_head(bbox_feat)

        # Decoupled classification pathway
        cls_feat = self.cls_stem(features)
        cls_pred = self.cls_head(cls_feat)

        batch_size, _, height, width = bbox_pred.shape

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()
        bbox_pred = bbox_pred.view(batch_size, height * width * self.num_anchors, 4)

        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
        cls_pred = cls_pred.view(batch_size, height * width * self.num_anchors, self.num_classes)

        return {
            'bbox': bbox_pred,
            'cls': cls_pred,
            'features': features,
            'bbox_feat': bbox_feat,
            'cls_feat': cls_feat,
        }


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance (used in detection heads)."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C) unnormalized class predictions
            target: (N,) class indices, or (N, C) one-hot labels
        """
        p = F.softmax(pred, dim=-1)
        ce = F.cross_entropy(pred, target, reduction='none')
        p_t = p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes (cx, cy, w, h format).
    Args:
        box1, box2: (x1, y1, x2, y2) in absolute coordinates
    Returns:
        iou value [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def match_predictions_to_targets(pred_boxes, pred_scores, target_boxes, target_classes, iou_threshold=0.5):
    """
    One-to-one assignment: match each prediction to at most one target.
    Simple greedy matching by IoU.

    Args:
        pred_boxes: (N, 4) predicted boxes
        pred_scores: (N, C) predicted class scores
        target_boxes: (M, 4) target boxes
        target_classes: (M,) target class indices
        iou_threshold: minimum IoU to match

    Returns:
        matched_targets: (N, 4+1) matched bbox and class (or -1 for background)
        matched_mask: (N,) boolean mask of matched predictions
    """
    N = pred_boxes.shape[0]
    M = target_boxes.shape[0]

    matched_targets = torch.zeros(N, 5, dtype=pred_boxes.dtype, device=pred_boxes.device)
    matched_targets[:, 4] = -1
    matched_mask = torch.zeros(N, dtype=torch.bool, device=pred_boxes.device)

    if M == 0:
        return matched_targets, matched_mask

    used_targets = set()

    for i in range(N):
        best_iou = -1
        best_j = -1

        for j in range(M):
            if j in used_targets:
                continue

            iou = compute_iou(pred_boxes[i].detach().cpu().numpy(),
                            target_boxes[j].detach().cpu().numpy())

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_j = j

        if best_j >= 0:
            matched_targets[i, :4] = target_boxes[best_j]
            matched_targets[i, 4] = target_classes[best_j]
            matched_mask[i] = True
            used_targets.add(best_j)

    return matched_targets, matched_mask


class YOLOv10DetectorPass2(nn.Module):
    """
    Pass 2: YOLOv10 with decoupled heads and loss computation.

    Key improvements:
    - Decoupled regression and classification heads (independent feature paths)
    - Focal loss for class imbalance handling
    - One-to-one target assignment
    """
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.detection_head = DecoupledDetectionHead(num_classes=num_classes)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, x, targets=None):
        outputs = self.detection_head(x)

        if targets is not None:
            loss = self.compute_loss(outputs, targets)
            outputs['loss'] = loss

        return outputs

    def compute_loss(self, outputs, targets):
        """
        Compute total loss (bbox + classification).

        Args:
            outputs: dict with 'bbox' and 'cls' predictions
            targets: dict with 'boxes' and 'classes' tensors
                - boxes: (B, M, 4) target boxes
                - classes: (B, M) target class indices (-1 for padding)

        Returns:
            total_loss: scalar loss
        """
        pred_boxes = outputs['bbox']
        pred_scores = outputs['cls']
        target_boxes = targets['boxes']
        target_classes = targets['classes']

        batch_size = pred_boxes.shape[0]
        device = pred_boxes.device

        losses = []

        for b in range(batch_size):
            valid_targets = target_classes[b] >= 0
            valid_target_boxes = target_boxes[b][valid_targets]
            valid_target_classes = target_classes[b][valid_targets]

            if valid_target_boxes.shape[0] > 0:
                matched_targets, matched_mask = match_predictions_to_targets(
                    pred_boxes[b],
                    pred_scores[b],
                    valid_target_boxes,
                    valid_target_classes,
                    iou_threshold=0.5
                )

                matched_idx = torch.where(matched_mask)[0]

                if matched_idx.shape[0] > 0:
                    pred_boxes_matched = pred_boxes[b][matched_idx]
                    target_boxes_matched = matched_targets[matched_idx, :4]

                    bbox_loss = F.smooth_l1_loss(pred_boxes_matched, target_boxes_matched)

                    pred_scores_matched = pred_scores[b][matched_idx]
                    target_classes_matched = matched_targets[matched_idx, 4].long()

                    cls_loss = self.focal_loss(pred_scores_matched, target_classes_matched)

                    losses.append(bbox_loss + cls_loss)

        if len(losses) > 0:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=device, dtype=pred_boxes.dtype, requires_grad=True)

        return total_loss


def make_model(num_classes=80, use_pass2=False):
    """Factory function to create YOLOv10 model."""
    if use_pass2:
        return YOLOv10DetectorPass2(num_classes=num_classes)
    return YOLOv10Detector(num_classes=num_classes)
