import torch
import numpy as np
from typing import Tuple, List, Optional


class BlockScorer(torch.nn.Module):
    """Learnable network that scores block importance from value cache."""

    def __init__(self, dim: int, num_blocks: int, hidden_dim: int = 64):
        """
        Initialize block scorer.

        Args:
            dim: Value dimension (typically same as embedding dimension)
            num_blocks: Number of blocks to score
            hidden_dim: Hidden dimension of the scoring network
        """
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks

        # Small MLP to score block importance
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, value_cache: torch.Tensor) -> torch.Tensor:
        """
        Score importance of each block.

        Args:
            value_cache: Cached values of shape (batch, seq_len, dim)
                        or (batch, num_heads, seq_len, head_dim) - will be reshaped

        Returns:
            scores: Block importance scores of shape (batch, num_blocks)
        """
        batch_size = value_cache.shape[0]
        seq_len = value_cache.shape[-2]

        # Ensure value_cache is (batch, seq_len, dim)
        if value_cache.ndim == 4:
            # Reshape from (batch, num_heads, seq_len, head_dim) to (batch, seq_len, -1)
            batch_size, num_heads, seq_len, head_dim = value_cache.shape
            value_cache = value_cache.transpose(1, 2).contiguous()
            value_cache = value_cache.view(batch_size, seq_len, -1)

        # Pool values per block by averaging
        block_size = (seq_len + self.num_blocks - 1) // self.num_blocks

        block_features = []
        for block_idx in range(self.num_blocks):
            start = block_idx * block_size
            end = min((block_idx + 1) * block_size, seq_len)

            if start < seq_len:
                block_val = value_cache[:, start:end, :].mean(dim=1)  # (batch, dim)
            else:
                block_val = torch.zeros(batch_size, value_cache.shape[-1], device=value_cache.device)

            block_features.append(block_val)

        block_features = torch.stack(block_features, dim=1)  # (batch, num_blocks, dim)

        # Score each block
        scores = self.mlp(block_features).squeeze(-1)  # (batch, num_blocks)

        return scores


class SparseAttentionMask:
    """Generate sparse attention masks for video generation with persistent blocks."""

    def __init__(self, seq_len: int, block_size: int, num_persistent_blocks: int = 2,
                 persistent_block_indices: Optional[List[int]] = None):
        """
        Initialize sparse attention mask generator.

        Args:
            seq_len: Total sequence length (e.g., total frames * spatial tokens)
            block_size: Size of each attention block
            num_persistent_blocks: Number of blocks to mark as persistent/salient
            persistent_block_indices: Optional list of block indices to mark as persistent.
                                     If None, uses fixed blocks [0, 1, ..., num_persistent_blocks-1]
        """
        self.seq_len = seq_len
        self.block_size = block_size
        self.num_persistent_blocks = num_persistent_blocks
        self.num_blocks = (seq_len + block_size - 1) // block_size

        # Use provided indices or default to first N blocks
        if persistent_block_indices is not None:
            self.persistent_block_indices = persistent_block_indices
        else:
            self.persistent_block_indices = list(range(min(num_persistent_blocks, self.num_blocks)))

    def get_block_range(self, block_idx: int) -> Tuple[int, int]:
        """Get the start and end indices for a block."""
        start = block_idx * self.block_size
        end = min((block_idx + 1) * self.block_size, self.seq_len)
        return start, end

    def create_mask(self) -> torch.Tensor:
        """
        Create sparse attention mask.

        Returns:
            mask: Binary mask of shape (seq_len, seq_len) where mask[i, j] = 1
                  means position i can attend to position j
        """
        mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)

        for i in range(self.seq_len):
            block_i = i // self.block_size
            block_start_i, block_end_i = self.get_block_range(block_i)

            # Local block attention: attend to all positions in same block
            mask[i, block_start_i:block_end_i] = True

            # Persistent block attention: attend to all persistent blocks
            for persist_block_idx in self.persistent_block_indices:
                block_start_p, block_end_p = self.get_block_range(persist_block_idx)
                mask[i, block_start_p:block_end_p] = True

        return mask

    def get_sparsity_ratio(self, mask: torch.Tensor) -> float:
        """Calculate the sparsity ratio (fraction of zeros in mask)."""
        total_elements = mask.numel()
        non_zero = mask.sum().item()
        return 1.0 - (non_zero / total_elements)

    def get_mask_statistics(self, mask: torch.Tensor) -> dict:
        """Get statistics about the sparse attention mask."""
        total = mask.numel()
        non_zero = mask.sum().item()
        return {
            'total_positions': total,
            'allowed_positions': non_zero,
            'sparsity_ratio': 1.0 - (non_zero / total),
            'compression_ratio': total / non_zero if non_zero > 0 else float('inf')
        }


def apply_sparse_mask_to_attention(attn_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply sparse attention mask to attention scores.

    Args:
        attn_scores: Attention scores of shape (..., seq_len, seq_len)
        mask: Sparse attention mask of shape (seq_len, seq_len)

    Returns:
        Masked attention scores where masked positions are set to -inf
    """
    masked_scores = attn_scores.clone()
    masked_scores[..., ~mask] = float('-inf')
    return masked_scores


class MultiHeadSparseAttention(torch.nn.Module):
    """Multi-head attention with sparse attention patterns and learnable block selection."""

    def __init__(self, dim: int, num_heads: int, block_size: int, num_persistent_blocks: int,
                 use_learned_blocks: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)

        self.block_size = block_size
        self.num_persistent_blocks = num_persistent_blocks
        self.use_learned_blocks = use_learned_blocks
        self.sparse_mask_gen = None
        self._mask_cache = {}

        # Optional learnable block scorer
        if use_learned_blocks:
            num_blocks = 1  # Will be updated dynamically
            self.block_scorer = BlockScorer(dim, num_blocks, hidden_dim=64)
        else:
            self.block_scorer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head sparse attention with optional learned block selection.

        Args:
            x: Input of shape (batch_size, seq_len, dim)

        Returns:
            Output of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Compute number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Update block scorer if needed
        if self.use_learned_blocks and self.block_scorer.num_blocks != num_blocks:
            self.block_scorer = BlockScorer(self.dim, num_blocks, hidden_dim=64).to(x.device)

        # Determine persistent block indices
        if self.use_learned_blocks:
            # Compute block importance scores from input
            block_scores = self.block_scorer(x)  # (batch, num_blocks)

            # Select top-k blocks per batch
            _, top_block_indices = torch.topk(block_scores, k=self.num_persistent_blocks, dim=1)
            # Use the first batch's selection for mask (could be made batch-specific if needed)
            persistent_block_indices = top_block_indices[0].tolist()
        else:
            persistent_block_indices = None

        # Initialize sparse mask generator if needed
        if self.sparse_mask_gen is None or self.sparse_mask_gen.seq_len != seq_len:
            self.sparse_mask_gen = SparseAttentionMask(
                seq_len=seq_len,
                block_size=self.block_size,
                num_persistent_blocks=self.num_persistent_blocks,
                persistent_block_indices=persistent_block_indices
            )
            self._mask_cache.clear()
        elif persistent_block_indices is not None:
            # Update mask generator with new persistent blocks
            self.sparse_mask_gen.persistent_block_indices = persistent_block_indices

        # Get or create sparse mask
        if seq_len not in self._mask_cache:
            self._mask_cache[seq_len] = self.sparse_mask_gen.create_mask().to(x.device)
        mask = self._mask_cache[seq_len]

        # Project input
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for head dimension
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply sparse mask
        scores = apply_sparse_mask_to_attention(scores, mask)

        # Softmax and apply to values
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)

        # Reshape and project
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)

        return out
