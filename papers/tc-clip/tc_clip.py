"""TC-CLIP: Temporal Contextualization for Video Action Recognition.

Pass 1: Basic frame feature extraction and temporal pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvEncoder(nn.Module):
    """Simple convolutional encoder for frame feature extraction."""

    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        # Simple 3-layer CNN to extract spatial features
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256, feature_dim)

    def forward(self, frames):
        """Extract features from a batch of frames.

        Args:
            frames: Tensor of shape (B, T, C, H, W) where B is batch size,
                   T is number of frames, C is channels (3), H, W are spatial dims.

        Returns:
            Tensor of shape (B, T, feature_dim) containing per-frame embeddings.
        """
        B, T, C, H, W = frames.shape
        # Flatten batch and time: (B*T, C, H, W)
        frames_flat = frames.view(B * T, C, H, W)

        # Forward through conv layers
        x = self.conv1(frames_flat)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(B * T, 256)

        # Project to feature_dim
        features = self.fc(x)

        # Reshape back to (B, T, feature_dim)
        features = features.view(B, T, self.feature_dim)
        return features


class FrameEncoder(nn.Module):
    """Extracts per-frame visual features."""

    def __init__(self, feature_dim=512, freeze_backbone=True):
        super().__init__()
        self.encoder = SimpleConvEncoder(feature_dim=feature_dim)
        self.feature_dim = feature_dim

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, frames):
        """Extract features from a batch of frames.

        Args:
            frames: Tensor of shape (B, T, C, H, W)

        Returns:
            Tensor of shape (B, T, feature_dim)
        """
        return self.encoder(frames)


class TemporalAggregator(nn.Module):
    """Aggregates per-frame features across the temporal dimension."""

    def __init__(self, aggregation_type="mean"):
        super().__init__()
        self.aggregation_type = aggregation_type

    def forward(self, frame_features):
        """Aggregate per-frame features.

        Args:
            frame_features: Tensor of shape (B, T, feature_dim)

        Returns:
            Tensor of shape (B, feature_dim) with aggregated features.
        """
        if self.aggregation_type == "mean":
            return frame_features.mean(dim=1)
        elif self.aggregation_type == "max":
            return frame_features.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")


class TCClipPass1(nn.Module):
    """TC-CLIP Pass 1: Basic frame encoding and temporal pooling."""

    def __init__(self, feature_dim=512, freeze_backbone=True, aggregation="mean"):
        super().__init__()
        self.frame_encoder = FrameEncoder(
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone
        )
        self.temporal_aggregator = TemporalAggregator(aggregation_type=aggregation)

    def forward(self, video_frames):
        """Process a batch of videos.

        Args:
            video_frames: Tensor of shape (B, T, C, H, W)

        Returns:
            Dictionary with 'video_features' (B, feature_dim) and
            'frame_features' (B, T, feature_dim).
        """
        frame_features = self.frame_encoder(video_frames)
        video_features = self.temporal_aggregator(frame_features)

        return {
            "video_features": video_features,
            "frame_features": frame_features
        }


def get_video_transforms(image_size=224):
    """Get standard transforms for video frame preprocessing.

    Returns a simple normalization for synthetic data.
    """
    def normalize(video):
        # Normalize from [0, 1] to [-1, 1] range
        return video * 2.0 - 1.0

    return normalize
