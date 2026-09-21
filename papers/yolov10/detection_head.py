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


def make_model(num_classes=80):
    """Factory function to create YOLOv10 model."""
    return YOLOv10Detector(num_classes=num_classes)
