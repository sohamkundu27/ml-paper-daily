"""
Pass 1: Multi-scale tokenizer and basic single-layer VAR core.

This implements the foundational pieces of Visual AutoRegressive Modeling:
- A hierarchical tokenizer that downsamples images into token maps at different scales
- A single transformer layer that encodes coarse-scale tokens and predicts next-scale tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalTokenizer(nn.Module):
    """
    Simple hierarchical tokenizer that downsamples an image into token maps at different scales.

    Takes an image of shape (B, C, H, W) and produces a list of token maps where each scale
    is half the resolution of the previous one. Tokens are raw feature vectors.
    """

    def __init__(self, in_channels=3, token_dim=256, num_scales=3):
        """
        Args:
            in_channels: Number of input image channels (default: 3 for RGB)
            token_dim: Dimension of each token representation
            num_scales: Number of hierarchical scales to produce
        """
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.num_scales = num_scales

        # Convolutional layers to progressively downsample and extract tokens at each scale
        self.conv_layers = nn.ModuleList()
        for i in range(num_scales):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = token_dim

            # Each layer: downsample by 2x (stride=2) and map to token_dim
            layer = nn.Sequential(
                nn.Conv2d(in_dim, token_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.conv_layers.append(layer)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            token_maps: List of length num_scales, where each element is shape (B, token_dim, h_i, w_i)
                       and h_i, w_i are progressively halved at each scale
        """
        token_maps = []
        current = x

        for conv_layer in self.conv_layers:
            current = conv_layer(current)
            token_maps.append(current)

        return token_maps


def tokenize_to_sequence(token_maps):
    """
    Flatten hierarchical token maps into a single sequence for transformer processing.

    Args:
        token_maps: List of token maps, each of shape (B, D, H, W)

    Returns:
        token_sequence: Tensor of shape (B, N, D) where N = sum of (h_i * w_i) for all scales
        scale_indices: Tensor indicating which scale each token belongs to, shape (N,)
    """
    sequences = []
    scale_indices = []

    for scale_idx, token_map in enumerate(token_maps):
        B, D, H, W = token_map.shape
        # Flatten spatial dimensions: (B, D, H, W) -> (B, H*W, D)
        tokens = token_map.permute(0, 2, 3, 1).reshape(B, H * W, D)
        sequences.append(tokens)
        scale_indices.extend([scale_idx] * (H * W))

    # Concatenate all scales into a single sequence
    token_sequence = torch.cat(sequences, dim=1)  # (B, N, D)
    scale_indices = torch.tensor(scale_indices, dtype=torch.long)

    return token_sequence, scale_indices


class VARTransformerLayer(nn.Module):
    """
    Single transformer layer for VAR next-scale prediction.

    Takes an input token sequence (typically from coarser scales) and produces
    logits for predicting tokens at the next finer scale.
    """

    def __init__(self, token_dim=256, num_heads=8, ff_dim=1024, vocab_size=4096):
        """
        Args:
            token_dim: Dimension of token embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of feedforward hidden layer
            vocab_size: Size of token vocabulary (for output logits)
        """
        super().__init__()
        self.token_dim = token_dim
        self.vocab_size = vocab_size

        # Self-attention
        self.norm1 = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)

        # Feed-forward
        self.norm2 = nn.LayerNorm(token_dim)
        self.ff = nn.Sequential(
            nn.Linear(token_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, token_dim),
        )

        # Output projection to vocabulary logits
        self.output_proj = nn.Linear(token_dim, vocab_size)

    def forward(self, x, mask=None):
        """
        Args:
            x: Token sequence of shape (B, N, D)
            mask: Optional attention mask

        Returns:
            logits: Tensor of shape (B, N, vocab_size) with logits for next-scale tokens
            hidden: Hidden representation from the layer, shape (B, N, D)
        """
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out

        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        # Project to logits
        logits = self.output_proj(x)

        return logits, x


class VARPass1(nn.Module):
    """
    Pass 1 of Visual AutoRegressive Modeling: tokenizer + single transformer layer.

    This is the foundational architecture. It:
    1. Tokenizes an image into hierarchical token maps
    2. Encodes all tokens from all scales using a single transformer layer
    3. Produces logits for predicting next-scale tokens
    """

    def __init__(self, in_channels=3, token_dim=256, num_scales=3,
                 num_heads=8, ff_dim=1024, vocab_size=4096):
        super().__init__()
        self.tokenizer = HierarchicalTokenizer(in_channels, token_dim, num_scales)
        self.transformer = VARTransformerLayer(token_dim, num_heads, ff_dim, vocab_size)
        self.num_scales = num_scales

    def forward(self, x):
        """
        Args:
            x: Image tensor of shape (B, C, H, W)

        Returns:
            logits: Logits for all tokens across all scales, shape (B, N, vocab_size)
            token_maps: The hierarchical token maps produced by tokenizer
        """
        # Tokenize image into hierarchical scales
        token_maps = self.tokenizer(x)

        # Convert to sequence
        token_sequence, scale_indices = tokenize_to_sequence(token_maps)

        # Pass through transformer layer
        logits, _ = self.transformer(token_sequence)

        return logits, token_maps
