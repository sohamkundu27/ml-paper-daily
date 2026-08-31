"""
Transformer attention layer with grammatically-guided sparse masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from sparse_attention import create_hard_mask, create_soft_mask


class SparseMultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional grammatical sparse masking.
    Applies the same sparse mask to all heads for consistency.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pos_tags: Optional[list] = None,
        mask_type: str = "hard",
        allow_self_attention: bool = True
    ):
        """
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads).
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            pos_tags: List of POS tags for sparse masking. If None, uses dense attention.
            mask_type: "hard" for binary masks, "soft" for weighted masks.
            allow_self_attention: Whether to allow self-attention.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Sparse masking
        self.pos_tags = pos_tags
        self.mask_type = mask_type
        self.allow_self_attention = allow_self_attention
        self.sparse_mask = None

        if pos_tags is not None:
            self._generate_sparse_mask(pos_tags, mask_type, allow_self_attention)

    def _generate_sparse_mask(
        self,
        pos_tags: list,
        mask_type: str = "hard",
        allow_self_attention: bool = True
    ):
        """Generate sparse mask based on POS tags."""
        if mask_type == "hard":
            mask_np = create_hard_mask(pos_tags, allow_self_attention=allow_self_attention)
        elif mask_type == "soft":
            mask_np = create_soft_mask(pos_tags, allow_self_attention=allow_self_attention)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        # Convert numpy mask to torch tensor
        # For hard masks, convert 0 -> -inf for masking in softmax
        if mask_type == "hard":
            self.sparse_mask = torch.from_numpy(mask_np).float()
            # Create attention mask: 0 -> -inf, 1 -> 0
            self.sparse_mask = torch.where(
                self.sparse_mask == 0,
                torch.tensor(float('-inf')),
                torch.tensor(0.0)
            )
        else:
            # For soft masks, use as multiplicative weights
            self.sparse_mask = torch.from_numpy(mask_np).float()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch_size, seq_len, embed_dim)
            key: (batch_size, seq_len, embed_dim)
            value: (batch_size, seq_len, embed_dim)
            key_padding_mask: (batch_size, seq_len) with True for positions to mask.
            need_weights: If True, return attention weights.

        Returns:
            output: (batch_size, seq_len, embed_dim)
            attn_weights: (batch_size, num_heads, seq_len, seq_len) if need_weights=True
        """
        batch_size, seq_len, embed_dim = query.shape

        # Project to multi-head space
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        # (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply sparse mask if available
        if self.sparse_mask is not None:
            sparse_mask_device = self.sparse_mask.to(scores.device)
            if self.mask_type == "hard":
                scores = scores + sparse_mask_device
            else:
                # For soft masks, multiply by the weights
                scores = scores * sparse_mask_device

        # Apply padding mask if provided
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)

        output = self.out_proj(attn_output)

        if need_weights:
            return output, attn_weights
        return output, None

    def get_sparse_mask(self) -> Optional[torch.Tensor]:
        """Return the sparse mask (for inspection/testing)."""
        if self.sparse_mask is None:
            return None
        if self.mask_type == "hard":
            # Convert back from attention mask form for inspection
            return torch.where(
                self.sparse_mask == float('-inf'),
                torch.tensor(0.0),
                torch.tensor(1.0)
            )
        return self.sparse_mask


def count_sparse_flops(
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    sparsity: float
) -> Tuple[int, int]:
    """
    Estimate FLOPs for attention computation.

    Args:
        seq_len: Sequence length.
        embed_dim: Embedding dimension.
        num_heads: Number of heads.
        sparsity: Sparsity of attention (fraction of masked positions).

    Returns:
        (dense_flops, sparse_flops) - approximate FLOPs for dense vs sparse attention.
    """
    head_dim = embed_dim // num_heads

    # Dense attention: Q*K^T + softmax + softmax*V
    # Q*K^T: (seq_len, head_dim) x (head_dim, seq_len) = seq_len^2 * head_dim ops
    # softmax*V: seq_len^2 * head_dim ops
    dense_flops = 2 * seq_len * seq_len * head_dim

    # Sparse attention: only (1-sparsity) * seq_len^2 positions computed
    sparse_flops = 2 * (1 - sparsity) * seq_len * seq_len * head_dim

    return dense_flops, int(sparse_flops)
