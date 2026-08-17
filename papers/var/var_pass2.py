"""
Pass 2: Full transformer stack and scale-conditional architecture.

This implements the core innovation of VAR:
- Multiple stacked transformer layers for deeper modeling
- Scale embeddings to condition predictions on scale
- Cumulative scale masking to enforce coarse-to-fine sequential prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from var_pass1 import (
    HierarchicalTokenizer,
    tokenize_to_sequence,
)


def create_cumulative_scale_mask(scale_indices, device):
    """
    Create a causal attention mask for scale-conditional prediction.

    When predicting tokens for scale i, the model should only attend to tokens
    from scales 0 to i (coarser scales and current scale). This implements the
    coarse-to-fine dependency structure that makes VAR distinctive.

    Args:
        scale_indices: Tensor of shape (N,) where each element is the scale index
                      of the corresponding token
        device: Device to create mask on

    Returns:
        mask: Boolean attention mask of shape (N, N). mask[i, j] = True means
              position i can attend to position j. False means no attention.
    """
    N = len(scale_indices)
    mask = torch.ones(N, N, dtype=torch.bool, device=device)

    # For each token position i
    for i in range(N):
        scale_i = scale_indices[i]
        # Token i can attend to tokens j from coarser or equal scales (scale_j <= scale_i)
        for j in range(N):
            scale_j = scale_indices[j]
            if scale_j > scale_i:
                mask[i, j] = False

    return mask


class ScaleEmbedding(nn.Module):
    """
    Adds learnable scale embeddings to token representations.

    Each scale gets a unique embedding that is added to all tokens from that scale.
    This allows the model to condition its predictions on which scale is being predicted.
    """

    def __init__(self, token_dim, num_scales):
        """
        Args:
            token_dim: Dimension of token embeddings
            num_scales: Number of hierarchical scales
        """
        super().__init__()
        self.scale_embeddings = nn.Embedding(num_scales, token_dim)

    def forward(self, token_sequence, scale_indices):
        """
        Add scale embeddings to the token sequence.

        Args:
            token_sequence: Tensor of shape (B, N, D)
            scale_indices: Tensor of shape (N,) with scale index for each token

        Returns:
            token_sequence + scale_emb: Tensor of shape (B, N, D) with scale info added
        """
        scale_emb = self.scale_embeddings(scale_indices)  # (N, D)
        return token_sequence + scale_emb.unsqueeze(0)  # Broadcast to batch


class VARTransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""

    def __init__(self, token_dim=256, num_heads=8, ff_dim=1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            token_dim, num_heads, batch_first=True, dropout=0.1
        )

        self.norm2 = nn.LayerNorm(token_dim)
        self.ff = nn.Sequential(
            nn.Linear(token_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, token_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: Token sequence of shape (B, N, D)
            attn_mask: Optional attention mask of shape (N, N)

        Returns:
            x: Transformed token sequence of shape (B, N, D)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False
        )
        x = x + attn_out

        # Feed-forward with residual
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        return x


class VARPass2(nn.Module):
    """
    Pass 2: Full VAR with transformer stack and scale-conditional architecture.

    This implements the distinctive mechanism of VAR:
    - Multiple stacked transformer layers process tokens at all scales
    - Scale embeddings condition predictions on which scale is being predicted
    - Cumulative scale masking enforces coarse-to-fine sequential structure
    - For each scale, the model can see all coarser scales but not finer ones
    """

    def __init__(
        self,
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    ):
        """
        Args:
            in_channels: Number of input image channels
            token_dim: Dimension of token embeddings
            num_scales: Number of hierarchical scales
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Dimension of feedforward hidden layer
            vocab_size: Size of token vocabulary (for output logits)
        """
        super().__init__()
        self.tokenizer = HierarchicalTokenizer(in_channels, token_dim, num_scales)
        self.scale_embedding = ScaleEmbedding(token_dim, num_scales)
        self.num_scales = num_scales
        self.token_dim = token_dim

        # Stack of transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                VARTransformerBlock(token_dim, num_heads, ff_dim)
                for _ in range(num_layers)
            ]
        )

        # Output projection to vocabulary logits
        self.norm_final = nn.LayerNorm(token_dim)
        self.output_proj = nn.Linear(token_dim, vocab_size)

    def forward(self, x):
        """
        Forward pass implementing scale-conditional next-scale prediction.

        Args:
            x: Image tensor of shape (B, C, H, W)

        Returns:
            logits: Logits for all tokens across all scales, shape (B, N, vocab_size)
            token_maps: The hierarchical token maps from the tokenizer
        """
        # Tokenize image into hierarchical scales
        token_maps = self.tokenizer(x)

        # Convert to sequence and get scale indices
        token_sequence, scale_indices = tokenize_to_sequence(token_maps)
        device = token_sequence.device
        scale_indices = scale_indices.to(device)

        # Add scale embeddings: each token gets an embedding for its scale
        token_sequence = self.scale_embedding(token_sequence, scale_indices)

        # Create cumulative scale mask for attention
        # This enforces that when predicting scale i, we can only see scales < i
        attn_mask = create_cumulative_scale_mask(scale_indices, device)
        # MultiheadAttention expects True for positions to mask OUT (not attend to)
        attn_mask = ~attn_mask  # Invert: True = mask out, False = attend

        # Pass through transformer stack
        hidden = token_sequence
        for layer in self.transformer_layers:
            hidden = layer(hidden, attn_mask=attn_mask)

        # Final layer norm and output projection
        hidden = self.norm_final(hidden)
        logits = self.output_proj(hidden)

        return logits, token_maps
