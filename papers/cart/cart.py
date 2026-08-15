"""
CART: Context-Anchored Recurrent Transformer
Pass 1: Core MLA block with learned LTI gate for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CartMLABlock(nn.Module):
    """
    Multi-head latent attention block with context reuse.

    In pass 1, this is simplified: no multi-head projection, single query transform.
    Context (K,V) is provided as input and reused throughout recurrence.
    """

    def __init__(self, dim: int, head_dim: int = 64, dropout: float = 0.0):
        """
        Args:
            dim: Feature dimension
            head_dim: Dimension per attention head
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Query projection: transform input for cross-attention over fixed context
        self.q_proj = nn.Linear(dim, head_dim)

        # Context is provided externally (K, V), no projection needed in this block
        # In Pass 2, prelude will generate K,V

        # Output projection
        self.out_proj = nn.Linear(head_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context_k: torch.Tensor, context_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Query input, shape (batch, seq_len, dim)
            context_k: Context keys, shape (batch, ctx_len, head_dim)
            context_v: Context values, shape (batch, ctx_len, head_dim)

        Returns:
            Output, shape (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape
        ctx_len = context_k.size(1)

        # Project query
        q = self.q_proj(x)  # (batch, seq_len, head_dim)

        # Attention scores: q @ k^T / sqrt(d)
        scores = torch.matmul(q, context_k.transpose(-2, -1)) * self.scale  # (batch, seq_len, ctx_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, context_v)  # (batch, seq_len, head_dim)

        # Output projection
        output = self.out_proj(attn_output)  # (batch, seq_len, dim)

        return output


class LearnedLTIGate(nn.Module):
    """
    Learned Linear Time-Invariant (LTI) gate for recurrent stability.

    Maintains spectral radius in a stable range via learnable scalar parameter.
    The recurrence x_{t+1} = α * x_t + y_t, where α controls the spectral radius.
    """

    def __init__(self, init_alpha: float = 0.8):
        """
        Args:
            init_alpha: Initial value for the gate parameter (spectral radius).
                       Should be in [0.7, 0.9] for stable recurrence.
        """
        super().__init__()
        # Store alpha as parameter, clipped to [0, 1) to ensure stability
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Recurrent update: x_new = clipped_alpha * x + y

        Args:
            x: Previous state, shape (batch, seq_len, dim)
            y: New information, shape (batch, seq_len, dim)

        Returns:
            Updated state, shape (batch, seq_len, dim)
        """
        # Clip alpha to [0, 0.99) to ensure spectral radius < 1 (stability)
        alpha_stable = torch.clamp(self.alpha, 0.0, 0.99)

        # Recurrent update
        x_new = alpha_stable * x + y

        return x_new

    def get_spectral_radius(self) -> torch.Tensor:
        """Return the current spectral radius (clamped alpha)."""
        return torch.clamp(self.alpha, 0.0, 0.99)


class CartRecurrentCore(nn.Module):
    """
    Recurrent core of CART: iteratively refines representation via MLA.

    Pass 1 simplified: single recurrent iteration over a fixed context.
    """

    def __init__(self, dim: int, head_dim: int = 64, num_iterations: int = 1, dropout: float = 0.0):
        """
        Args:
            dim: Feature dimension
            head_dim: Dimension per attention head
            num_iterations: Number of recurrent iterations (pass 1 uses 1)
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations

        # MLA block
        self.mla = CartMLABlock(dim, head_dim, dropout)

        # LTI gate for stability
        self.lti_gate = LearnedLTIGate(init_alpha=0.8)

    def forward(
        self,
        x_init: torch.Tensor,
        context_k: torch.Tensor,
        context_v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_init: Initial state, shape (batch, seq_len, dim)
            context_k: Context keys, shape (batch, ctx_len, head_dim)
            context_v: Context values, shape (batch, ctx_len, head_dim)

        Returns:
            Final state after recurrent iterations, shape (batch, seq_len, dim)
        """
        x = x_init

        for _ in range(self.num_iterations):
            # Apply MLA block
            y = self.mla(x, context_k, context_v)

            # Recurrent update via LTI gate
            x = self.lti_gate(x, y)

        return x


def create_dummy_context(batch: int, seq_len: int, ctx_len: int, head_dim: int, device: str = 'cpu'):
    """
    Create dummy context K,V for testing.

    Args:
        batch: Batch size
        seq_len: Sequence length
        ctx_len: Context length
        head_dim: Dimension per head
        device: Device to place tensors on

    Returns:
        Tuple of (K, V)
    """
    context_k = torch.randn(batch, ctx_len, head_dim, device=device)
    context_v = torch.randn(batch, ctx_len, head_dim, device=device)
    return context_k, context_v
