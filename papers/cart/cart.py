"""
CART: Context-Anchored Recurrent Transformer
Pass 1: Core MLA block with learned LTI gate for stability
Pass 2: Multi-layer prelude network for context encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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


class CartPrelude(nn.Module):
    """
    Multi-layer prelude network that encodes context into reusable K,V representations.

    The prelude computes K and V once from raw context, which are then reused
    across multiple recurrent iterations in the core. This separation of context
    encoding from iterative refinement is key to CART's parameter efficiency.

    Pass 2: multi-layer feedforward encoder with separate K,V projections.
    """

    def __init__(self, dim: int, head_dim: int, num_layers: int = 2, dropout: float = 0.0):
        """
        Args:
            dim: Input context dimension
            head_dim: Dimension per attention head (output dimension)
            num_layers: Number of layers in the encoder (default 2)
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim

        # Build multi-layer feedforward encoder
        layers = []
        for i in range(num_layers):
            in_d = dim if i == 0 else head_dim
            out_d = head_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

        # Separate learnable projections for K and V
        self.k_proj = nn.Linear(head_dim, head_dim)
        self.v_proj = nn.Linear(head_dim, head_dim)

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode context into K,V representations.

        Args:
            context: Raw context, shape (batch, ctx_len, dim)

        Returns:
            Tuple of (K, V), each with shape (batch, ctx_len, head_dim)
        """
        # Encode context through multi-layer feedforward
        encoded = self.encoder(context)  # (batch, ctx_len, head_dim)

        # Project to K and V (independent projections for expressiveness)
        k = self.k_proj(encoded)  # (batch, ctx_len, head_dim)
        v = self.v_proj(encoded)  # (batch, ctx_len, head_dim)

        return k, v


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


class Cart(nn.Module):
    """
    Full CART model: Prelude + RecurrentCore

    Demonstrates the separation of context encoding from iterative refinement.
    The prelude encodes raw context into K,V once; the core reuses them across
    multiple recurrent iterations, reducing per-iteration computation.

    Pass 2: integrates CartPrelude with CartRecurrentCore.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        prelude_layers: int = 2,
        num_iterations: int = 1,
        dropout: float = 0.0
    ):
        """
        Args:
            dim: Feature dimension (input and output)
            head_dim: Dimension per attention head
            prelude_layers: Number of layers in prelude encoder
            num_iterations: Number of recurrent iterations
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim

        # Prelude: encodes raw context into K,V
        self.prelude = CartPrelude(dim, head_dim, prelude_layers, dropout)

        # Recurrent core: refines input using encoded K,V
        self.core = CartRecurrentCore(dim, head_dim, num_iterations, dropout)

    def forward(self, x_init: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: prelude encodes context, core refines input.

        Args:
            x_init: Initial representation, shape (batch, seq_len, dim)
            context: Raw context, shape (batch, ctx_len, dim)

        Returns:
            Refined representation, shape (batch, seq_len, dim)
        """
        # Prelude: encode context into K,V (computed once, reused across iterations)
        context_k, context_v = self.prelude(context)

        # Core: refine x_init using fixed K,V across recurrent iterations
        output = self.core(x_init, context_k, context_v)

        return output


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


def create_dummy_raw_context(batch: int, ctx_len: int, dim: int, device: str = 'cpu'):
    """
    Create dummy raw context for prelude processing.

    Args:
        batch: Batch size
        ctx_len: Context length
        dim: Feature dimension
        device: Device to place tensors on

    Returns:
        Context tensor of shape (batch, ctx_len, dim)
    """
    return torch.randn(batch, ctx_len, dim, device=device)
