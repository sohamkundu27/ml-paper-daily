"""
Full transformer block with sparse attention support.
Includes attention, feed-forward network, layer norm, and residual connections.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from transformer_attention import SparseMultiHeadAttention


class TransformerBlock(nn.Module):
    """
    A single transformer block: attention + feed-forward with residual connections
    and layer normalization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        pos_tags: Optional[List[str]] = None,
        mask_type: str = "hard",
    ):
        """
        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            ffn_dim: Feed-forward network hidden dimension.
            dropout: Dropout probability.
            pos_tags: POS tags for sparse attention. If None, uses dense attention.
            mask_type: "hard" or "soft" for sparse masking.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.pos_tags = pos_tags

        # Multi-head attention
        self.attn = SparseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            pos_tags=pos_tags,
            mask_type=mask_type,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            key_padding_mask: (batch_size, seq_len) with True for positions to mask.

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        # Attention with residual connection
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


class TransformerStack(nn.Module):
    """
    Stack of transformer blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        pos_tags: Optional[List[str]] = None,
        mask_type: str = "hard",
    ):
        """
        Args:
            num_blocks: Number of transformer blocks to stack.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            ffn_dim: Feed-forward network hidden dimension.
            dropout: Dropout probability.
            pos_tags: POS tags for sparse attention.
            mask_type: "hard" or "soft" for sparse masking.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                pos_tags=pos_tags,
                mask_type=mask_type,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            key_padding_mask: (batch_size, seq_len) with True for positions to mask.

        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x
