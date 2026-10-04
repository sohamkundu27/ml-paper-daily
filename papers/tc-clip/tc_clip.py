"""TC-CLIP: Temporal Contextualization for Video Action Recognition.

Pass 1: Basic frame feature extraction and temporal pooling.
Pass 2: Temporal contextualization with learnable context tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class ContextTokenGenerator(nn.Module):
    """Generates learnable context tokens that summarize temporal relationships."""

    def __init__(self, feature_dim, num_context_tokens=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_context_tokens = num_context_tokens
        self.context_tokens = nn.Parameter(torch.randn(1, num_context_tokens, feature_dim))
        nn.init.normal_(self.context_tokens, std=0.02)

    def forward(self, batch_size):
        """Generate context tokens for a batch.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            Tensor of shape (batch_size, num_context_tokens, feature_dim)
        """
        return self.context_tokens.expand(batch_size, -1, -1)


class TemporalContextualizer(nn.Module):
    """Applies temporal contextualization via cross-attention."""

    def __init__(self, feature_dim, num_context_tokens=4, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_context_tokens = num_context_tokens
        self.num_heads = num_heads

        self.context_generator = ContextTokenGenerator(
            feature_dim, num_context_tokens
        )

        head_dim = feature_dim // num_heads
        self.scale = math.sqrt(head_dim)

        self.to_q = nn.Linear(feature_dim, feature_dim)
        self.to_k = nn.Linear(feature_dim, feature_dim)
        self.to_v = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.context_fusion = nn.Linear(feature_dim * 2, feature_dim)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.ff = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )

    def forward(self, frame_features):
        """Apply temporal contextualization to frame features.

        Args:
            frame_features: Tensor of shape (B, T, feature_dim)

        Returns:
            Dictionary with:
              - 'contextualized_features': (B, T, feature_dim)
              - 'context_tokens': (B, num_context_tokens, feature_dim)
              - 'attention_weights': (B*num_heads, num_context_tokens, T)
        """
        B, T, D = frame_features.shape

        context_tokens = self.context_generator(B)

        q = self.to_q(context_tokens)
        k = self.to_k(frame_features)
        v = self.to_v(frame_features)

        q = q.view(B, self.num_context_tokens, self.num_heads, D // self.num_heads)
        k = k.view(B, T, self.num_heads, D // self.num_heads)
        v = v.view(B, T, self.num_heads, D // self.num_heads)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = torch.softmax(scores, dim=-1)

        context_out = torch.matmul(attention, v)
        context_out = context_out.permute(0, 2, 1, 3).contiguous()
        context_out = context_out.view(B, self.num_context_tokens, D)

        context_out = self.out_proj(context_out)

        frame_context_attn = torch.matmul(
            torch.softmax(
                torch.matmul(frame_features, context_out.transpose(-2, -1)) / self.scale,
                dim=-1
            ),
            context_out
        )

        contextualized = self.norm1(frame_features + frame_context_attn)
        contextualized = contextualized + self.ff(self.norm2(contextualized))

        attention_flat = attention.view(B * self.num_heads, self.num_context_tokens, T)

        return {
            "contextualized_features": contextualized,
            "context_tokens": context_out,
            "attention_weights": attention_flat,
        }


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


class TCClipPass2(nn.Module):
    """TC-CLIP Pass 2: Adds temporal contextualization with learnable context tokens."""

    def __init__(
        self,
        feature_dim=512,
        freeze_backbone=True,
        aggregation="mean",
        num_context_tokens=4,
        num_heads=8,
    ):
        super().__init__()
        self.frame_encoder = FrameEncoder(
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone
        )
        self.temporal_contextualizer = TemporalContextualizer(
            feature_dim=feature_dim,
            num_context_tokens=num_context_tokens,
            num_heads=num_heads,
        )
        self.temporal_aggregator = TemporalAggregator(aggregation_type=aggregation)

    def forward(self, video_frames):
        """Process a batch of videos with temporal contextualization.

        Args:
            video_frames: Tensor of shape (B, T, C, H, W)

        Returns:
            Dictionary with:
              - 'video_features': (B, feature_dim) aggregated from contextualized frames
              - 'frame_features': (B, T, feature_dim) original per-frame embeddings
              - 'contextualized_features': (B, T, feature_dim) with temporal context
              - 'context_tokens': (B, num_context_tokens, feature_dim)
              - 'attention_weights': (B*num_heads, num_context_tokens, T)
        """
        frame_features = self.frame_encoder(video_frames)

        tc_output = self.temporal_contextualizer(frame_features)
        contextualized_features = tc_output["contextualized_features"]

        video_features = self.temporal_aggregator(contextualized_features)

        return {
            "video_features": video_features,
            "frame_features": frame_features,
            "contextualized_features": contextualized_features,
            "context_tokens": tc_output["context_tokens"],
            "attention_weights": tc_output["attention_weights"],
        }


def get_video_transforms(image_size=224):
    """Get standard transforms for video frame preprocessing.

    Returns a simple normalization for synthetic data.
    """
    def normalize(video):
        # Normalize from [0, 1] to [-1, 1] range
        return video * 2.0 - 1.0

    return normalize
