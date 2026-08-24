"""Minimal diffusion-based video generation model using sparse attention (Pass 4)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from sparse_attention import MultiHeadSparseAttention


class SimpleVideoUNet(nn.Module):
    """Simplified U-Net-like architecture for video generation."""

    def __init__(self, in_channels: int, out_channels: int, latent_dim: int,
                 num_blocks: int = 4, use_sparse_attention: bool = True):
        """
        Initialize video U-Net.

        Args:
            in_channels: Input channels (e.g., 3 for RGB)
            out_channels: Output channels
            latent_dim: Dimension of latent features
            num_blocks: Number of residual blocks
            use_sparse_attention: Whether to use sparse attention
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.use_sparse_attention = use_sparse_attention

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, latent_dim, kernel_size=3, padding=1)

        # Residual blocks with attention
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        for i in range(num_blocks):
            # Residual block
            block = nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim * 2, kernel_size=3, padding=1),
                nn.GroupNorm(8, latent_dim * 2),
                nn.GELU(),
                nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
            )
            self.blocks.append(block)

            # Attention block - reshape spatially to sequence
            if use_sparse_attention:
                attn = MultiHeadSparseAttention(
                    dim=latent_dim,
                    num_heads=4,
                    block_size=8,
                    num_persistent_blocks=2,
                    use_learned_blocks=False,
                    use_persistent_cache=False
                )
            else:
                # Standard multi-head attention for comparison
                attn = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)

            self.attention_blocks.append(attn)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_dim * 2),
            nn.GELU(),
            nn.Conv2d(latent_dim * 2, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through video U-Net.

        Args:
            x: Input of shape (batch, channels, height, width)

        Returns:
            Output of shape (batch, out_channels, height, width)
        """
        # Project input
        features = self.input_proj(x)  # (batch, latent_dim, h, w)
        batch_size, latent_dim, h, w = features.shape

        # Process through blocks
        for block, attn_block in zip(self.blocks, self.attention_blocks):
            # Residual connection
            residual = features

            # Apply convolutional block
            features = block(features)

            # Apply attention
            # Reshape to sequence for attention: (batch, seq_len, dim)
            features_seq = features.view(batch_size, latent_dim, -1).transpose(1, 2)

            if self.use_sparse_attention:
                # Use sparse attention layer
                features_seq = attn_block(features_seq)
            else:
                # Use standard attention
                features_seq, _ = attn_block(features_seq, features_seq, features_seq)

            # Reshape back to spatial: (batch, latent_dim, h, w)
            features = features_seq.transpose(1, 2).view(batch_size, latent_dim, h, w)

            # Add residual
            features = features + residual

        # Output projection
        output = self.output_proj(features)

        return output


class SimpleDiffusionModel(nn.Module):
    """Minimal diffusion model for video generation."""

    def __init__(self, channels: int = 3, latent_dim: int = 64,
                 num_blocks: int = 2, use_sparse_attention: bool = True,
                 num_timesteps: int = 50):
        """
        Initialize diffusion model.

        Args:
            channels: Number of image channels
            latent_dim: Dimension of latent features
            num_blocks: Number of U-Net blocks
            use_sparse_attention: Use sparse attention if True
            num_timesteps: Number of diffusion timesteps
        """
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.use_sparse_attention = use_sparse_attention

        # Timestep embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Main U-Net model
        self.model = SimpleVideoUNet(
            in_channels=channels,
            out_channels=channels,
            latent_dim=latent_dim,
            num_blocks=num_blocks,
            use_sparse_attention=use_sparse_attention
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting noise.

        Args:
            x: Noisy image of shape (batch, channels, height, width)
            t: Timestep of shape (batch,) with values in [0, 1]

        Returns:
            Predicted noise of shape (batch, channels, height, width)
        """
        # Embed timestep
        t_emb = self.time_embedding(t.unsqueeze(-1))  # (batch, latent_dim)

        # Main model prediction
        noise_pred = self.model(x)

        return noise_pred

    @torch.no_grad()
    def generate(self, shape: tuple, num_steps: int = 25, device: str = 'cpu') -> torch.Tensor:
        """
        Generate video frames via reverse diffusion.

        Args:
            shape: Output shape (batch, channels, height, width)
            num_steps: Number of diffusion steps for generation
            device: Device to run on

        Returns:
            Generated video of shape specified by shape
        """
        batch_size, channels, height, width = shape
        x = torch.randn(shape, device=device)

        # Simple DDIM sampling
        for step in range(num_steps):
            t = torch.full((batch_size,), step / num_steps, device=device)

            noise_pred = self.forward(x, t)

            # Simple denoising step
            alpha = 1.0 - (step / num_steps)
            x = alpha * x + (1 - alpha) * noise_pred

        # Clamp to valid range
        x = torch.clamp(x, -1, 1)

        return x


def get_timing(model: nn.Module, x: torch.Tensor, t: torch.Tensor,
               num_iterations: int = 10) -> dict:
    """
    Measure inference time for a model.

    Args:
        model: Model to time
        x: Input tensor
        t: Timestep tensor
        num_iterations: Number of forward passes to average

    Returns:
        Dictionary with timing statistics
    """
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, t)

    # Time forward passes
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x, t)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    total_time = end - start
    avg_time = total_time / num_iterations

    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'time_per_iteration': avg_time,
    }
