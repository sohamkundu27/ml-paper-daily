"""TC-CLIP: Temporal Contextualization for Video Action Recognition.

Pass 1: Basic frame feature extraction and temporal pooling.
Pass 2: Temporal contextualization with learnable context tokens.
Pass 3: CLIP text integration and video-conditional prompting.
Pass 4: End-to-end demo with synthetic action recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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


class SimpleTextEncoder(nn.Module):
    """Simple text encoder for action class prompts."""

    def __init__(self, vocab_size=10000, embedding_dim=512, output_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(77, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, output_dim)
        )

    def forward(self, token_ids, num_tokens=None):
        """Encode token IDs to embeddings.

        Args:
            token_ids: Tensor of shape (B, seq_len) with token indices
            num_tokens: Optional actual length of tokens (for variable-length sequences)

        Returns:
            Tensor of shape (B, output_dim) with text embeddings
        """
        B, seq_len = token_ids.shape
        pos_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(B, -1)

        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(pos_ids)

        x = token_emb + pos_emb
        if num_tokens is not None:
            mask = torch.arange(seq_len, device=token_ids.device).unsqueeze(0) < num_tokens.unsqueeze(1)
            x = x * mask.unsqueeze(-1).float()
            text_embedding = x.sum(dim=1) / num_tokens.unsqueeze(1).float().clamp(min=1)
        else:
            text_embedding = x.mean(dim=1)

        output = self.encoder(text_embedding)
        return F.normalize(output, p=2, dim=1)


class VideoConditionalPrompting(nn.Module):
    """Generates context-aware text prompt embeddings based on video features."""

    def __init__(self, feature_dim=512, prompt_dim=512, num_prompts=10):
        super().__init__()
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts

        self.prompt_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_prompts)
        )

        self.prompt_projection = nn.Linear(feature_dim, prompt_dim)

    def forward(self, video_features, base_prompts=None):
        """Generate video-conditional prompt embeddings.

        Args:
            video_features: Tensor of shape (B, feature_dim) from video encoder
            base_prompts: Optional tensor of shape (num_prompts, prompt_dim) with base prompt embeddings

        Returns:
            Dictionary with:
              - 'conditional_prompts': (B, num_prompts, prompt_dim)
              - 'prompt_weights': (B, num_prompts) - attention weights for prompts
        """
        B = video_features.shape[0]

        prompt_weights = self.prompt_attention(video_features)
        prompt_weights = F.softmax(prompt_weights, dim=1)

        if base_prompts is not None:
            conditional_prompts = torch.matmul(
                prompt_weights.unsqueeze(1),
                base_prompts.unsqueeze(0)
            ).squeeze(1)
            conditional_prompts = conditional_prompts.unsqueeze(1).expand(-1, self.num_prompts, -1)
        else:
            projection = self.prompt_projection(video_features)
            conditional_prompts = projection.unsqueeze(1).expand(-1, self.num_prompts, -1)

        return {
            "conditional_prompts": conditional_prompts,
            "prompt_weights": prompt_weights,
        }


class TCClipPass3(nn.Module):
    """TC-CLIP Pass 3: Video-text alignment with video-conditional prompting."""

    def __init__(
        self,
        feature_dim=512,
        freeze_backbone=True,
        aggregation="mean",
        num_context_tokens=4,
        num_heads=8,
        vocab_size=10000,
        num_action_classes=10,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_action_classes = num_action_classes

        self.video_encoder = TCClipPass2(
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone,
            aggregation=aggregation,
            num_context_tokens=num_context_tokens,
            num_heads=num_heads,
        )

        self.text_encoder = SimpleTextEncoder(
            vocab_size=vocab_size,
            embedding_dim=feature_dim,
            output_dim=feature_dim
        )

        self.video_conditional_prompting = VideoConditionalPrompting(
            feature_dim=feature_dim,
            prompt_dim=feature_dim,
            num_prompts=num_action_classes
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, video_frames, action_token_ids=None):
        """Process video and compute action classification scores.

        Args:
            video_frames: Tensor of shape (B, T, C, H, W)
            action_token_ids: Optional tensor of shape (num_action_classes, seq_len) with action tokens

        Returns:
            Dictionary with:
              - 'video_features': (B, feature_dim)
              - 'action_logits': (B, num_action_classes) - raw alignment scores
              - 'action_probs': (B, num_action_classes) - softmax probabilities
              - 'conditional_prompts': (B, num_action_classes, feature_dim)
              - Additional keys from video encoder
        """
        video_output = self.video_encoder(video_frames)
        video_features = video_output["video_features"]

        vcp_output = self.video_conditional_prompting(video_features)
        conditional_prompts = vcp_output["conditional_prompts"]
        prompt_weights = vcp_output["prompt_weights"]

        weighted_context = (prompt_weights.unsqueeze(-1) * conditional_prompts).sum(dim=1)
        refined_video_features = F.normalize(
            video_features + 0.3 * weighted_context,
            p=2, dim=1
        )

        if action_token_ids is not None:
            action_embeddings = self.text_encoder(action_token_ids)
        else:
            action_embeddings = F.normalize(
                torch.randn(self.num_action_classes, self.feature_dim, device=video_features.device),
                p=2, dim=1
            )

        action_logits = torch.matmul(refined_video_features, action_embeddings.t()) * torch.exp(self.temperature)
        action_probs = F.softmax(action_logits, dim=1)

        return {
            "video_features": video_features,
            "refined_video_features": refined_video_features,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "action_embeddings": action_embeddings,
            "conditional_prompts": conditional_prompts,
            "prompt_weights": prompt_weights,
            "frame_features": video_output["frame_features"],
            "contextualized_features": video_output["contextualized_features"],
            "context_tokens": video_output["context_tokens"],
            "attention_weights": video_output["attention_weights"],
        }


def get_video_transforms(image_size=224):
    """Get standard transforms for video frame preprocessing.

    Returns a simple normalization for synthetic data.
    """
    def normalize(video):
        # Normalize from [0, 1] to [-1, 1] range
        return video * 2.0 - 1.0

    return normalize


class TCClipPass4(nn.Module):
    """TC-CLIP Pass 4: End-to-end demo with synthetic video evaluation."""

    def __init__(
        self,
        feature_dim=256,
        freeze_backbone=True,
        aggregation="mean",
        num_context_tokens=4,
        num_heads=8,
        vocab_size=5000,
        num_action_classes=5,
    ):
        super().__init__()
        self.model = TCClipPass3(
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone,
            aggregation=aggregation,
            num_context_tokens=num_context_tokens,
            num_heads=num_heads,
            vocab_size=vocab_size,
            num_action_classes=num_action_classes,
        )
        self.num_action_classes = num_action_classes
        self.feature_dim = feature_dim

    def forward(self, video_frames, action_token_ids=None):
        """Process video and compute action predictions.

        Args:
            video_frames: Tensor of shape (B, T, C, H, W)
            action_token_ids: Optional tensor of shape (num_action_classes, seq_len)

        Returns:
            Dictionary with predictions and confidence scores
        """
        output = self.model(video_frames, action_token_ids=action_token_ids)
        return output

    def predict(self, video_frames, action_token_ids=None):
        """Get class predictions from videos.

        Args:
            video_frames: Tensor of shape (B, T, C, H, W)
            action_token_ids: Optional action token embeddings

        Returns:
            Tuple of (predicted_classes, confidence_scores) both shape (B,)
        """
        with torch.no_grad():
            output = self.forward(video_frames, action_token_ids=action_token_ids)
            action_probs = output["action_probs"]
            predicted_classes = action_probs.argmax(dim=1)
            confidence_scores = action_probs.max(dim=1)[0]
        return predicted_classes, confidence_scores

    def get_learnable_parameters(self):
        """Get parameters that are trainable."""
        return [p for p in self.parameters() if p.requires_grad]


class ToyActionDataset:
    """Generates synthetic videos with simple motion patterns as action classes."""

    def __init__(
        self,
        num_samples=100,
        num_actions=5,
        num_frames=8,
        height=64,
        width=64,
        channels=3,
        seed=42,
    ):
        self.num_samples = num_samples
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.channels = channels
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_action_video(self, action_id):
        """Generate a synthetic video for a specific action.

        Actions are encoded as different motion patterns:
        - Action 0: Horizontal motion (left-right)
        - Action 1: Vertical motion (up-down)
        - Action 2: Diagonal motion
        - Action 3: Circular motion
        - Action 4: Static with color variation
        """
        video = torch.zeros(self.num_frames, self.channels, self.height, self.width)

        for t in range(self.num_frames):
            frame = torch.ones(self.channels, self.height, self.width) * 0.5
            x_center = self.width // 2
            y_center = self.height // 2

            if action_id == 0:
                x = int(x_center + 10 * np.sin(2 * np.pi * t / self.num_frames))
                if 5 <= x < self.width - 5:
                    frame[:, y_center - 5:y_center + 5, x - 5:x + 5] = 1.0

            elif action_id == 1:
                y = int(y_center + 10 * np.sin(2 * np.pi * t / self.num_frames))
                if 5 <= y < self.height - 5:
                    frame[:, y - 5:y + 5, x_center - 5:x_center + 5] = 1.0

            elif action_id == 2:
                x = int(x_center + 8 * np.sin(2 * np.pi * t / self.num_frames))
                y = int(y_center + 8 * np.cos(2 * np.pi * t / self.num_frames))
                if 5 <= x < self.width - 5 and 5 <= y < self.height - 5:
                    frame[:, y - 5:y + 5, x - 5:x + 5] = 1.0

            elif action_id == 3:
                theta = 2 * np.pi * t / self.num_frames
                x = int(x_center + 12 * np.cos(theta))
                y = int(y_center + 12 * np.sin(theta))
                if 5 <= x < self.width - 5 and 5 <= y < self.height - 5:
                    frame[:, y - 4:y + 4, x - 4:x + 4] = 1.0

            else:
                color_factor = 0.5 + 0.3 * t / self.num_frames
                frame[:, 10:self.height - 10, 10:self.width - 10] = color_factor

            video[t] = frame

        video = torch.clamp(video, 0, 1)
        return video

    def __iter__(self):
        """Generate dataset by yielding (video, action_label) tuples."""
        samples_per_action = self.num_samples // self.num_actions
        for action_id in range(self.num_actions):
            for _ in range(samples_per_action):
                video = self.generate_action_video(action_id)
                yield video, action_id

    def get_batch(self, batch_size=4, action_id=None):
        """Get a batch of videos.

        Args:
            batch_size: Number of videos per batch
            action_id: If specified, only generate videos for this action

        Returns:
            Tuple of (videos, labels) both batched
        """
        if action_id is not None:
            videos = []
            labels = []
            for _ in range(batch_size):
                video = self.generate_action_video(action_id)
                videos.append(video)
                labels.append(action_id)
            return torch.stack(videos), torch.tensor(labels)
        else:
            videos = []
            labels = []
            for _ in range(batch_size):
                aid = np.random.randint(0, self.num_actions)
                video = self.generate_action_video(aid)
                videos.append(video)
                labels.append(aid)
            return torch.stack(videos), torch.tensor(labels)
