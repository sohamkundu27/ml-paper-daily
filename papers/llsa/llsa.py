import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class HierarchicalSparseAttention(nn.Module):
    """
    Hierarchical sparse attention with Top-K block selection.

    Builds a pyramid of token groups at different granularities and selects
    the top-k blocks at each level to determine which tokens participate in attention.
    """

    def __init__(self, num_levels: int = 3, top_k_per_level: List[int] = None):
        """
        Args:
            num_levels: Number of pyramid levels (coarse to fine)
            top_k_per_level: Number of top blocks to keep at each level
                           If None, uses [seq_len//(4**i) for each level]
        """
        super().__init__()
        self.num_levels = num_levels
        self.top_k_per_level = top_k_per_level

    def _compute_block_importance(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        """
        Compute importance scores for each block.
        Uses L2 norm of embeddings as importance measure.

        Args:
            x: (seq_len, d) token embeddings
            block_size: size of each block

        Returns:
            (num_blocks,) importance scores
        """
        seq_len = x.shape[0]
        num_blocks = (seq_len + block_size - 1) // block_size

        block_importance = []
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, seq_len)
            block_emb = x[start:end]
            # Importance = mean L2 norm across tokens in block
            importance = block_emb.norm(dim=-1).mean()
            block_importance.append(importance)

        return torch.stack(block_importance)

    def _compute_learnable_block_importance(
        self,
        block_size: int,
        num_blocks: int,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance scores using learnable log-linear weights.

        Args:
            block_size: size of each block
            num_blocks: number of blocks at this level
            weights: learnable log-linear weights for this level

        Returns:
            (num_blocks,) log-linear importance scores
        """
        # Use log-linear (softmax-like) scoring with learnable weights
        # importance = exp(w_i) for block i
        actual_weights = weights[:num_blocks]
        return torch.exp(actual_weights)

    def _select_top_k_blocks(
        self,
        x: torch.Tensor,
        block_size: int,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k blocks from the sequence.

        Args:
            x: (seq_len, d) token embeddings
            block_size: size of each block
            k: number of blocks to select

        Returns:
            selected_block_ids: (k,) indices of selected blocks
            selected_token_mask: (seq_len,) boolean mask of selected tokens
        """
        seq_len = x.shape[0]

        # Compute importance for each block
        block_importance = self._compute_block_importance(x, block_size)
        num_blocks = block_importance.shape[0]

        # Select top-k blocks
        k = min(k, num_blocks)
        _, top_block_ids = torch.topk(block_importance, k)

        # Convert block ids to token mask
        token_mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
        for block_id in top_block_ids:
            start = block_id * block_size
            end = min((block_id + 1) * block_size, seq_len)
            token_mask[start:end] = True

        return top_block_ids, token_mask

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Hierarchical sparse attention selection.

        Args:
            x: (seq_len, d) token embeddings

        Returns:
            selected_masks: list of (seq_len,) boolean masks for each level
            final_mask: (seq_len,) union of all selected tokens
        """
        seq_len = x.shape[0]

        # Determine top-k per level if not provided
        if self.top_k_per_level is None:
            top_k_per_level = []
            for level in range(self.num_levels):
                # Hierarchical sparsity: coarser levels are more sparse
                # Level 0 (coarsest): select ~12% of blocks
                # Level 1: select ~25% of blocks
                # Level 2 (finest): select ~50% of blocks
                block_size = max(1, seq_len // (2 ** (self.num_levels - level)))
                num_blocks = (seq_len + block_size - 1) // block_size
                sparsity_ratio = [0.125, 0.25, 0.5][min(level, 2)]
                k_at_level = max(1, int(num_blocks * sparsity_ratio))
                top_k_per_level.append(k_at_level)
        else:
            top_k_per_level = self.top_k_per_level

        # Hierarchical selection
        selected_masks = []
        all_selected = torch.zeros(seq_len, dtype=torch.bool, device=x.device)

        for level in range(self.num_levels):
            # Block size decreases (coarser at higher levels, finer at lower)
            block_size = max(1, seq_len // (2 ** (self.num_levels - level)))
            k = top_k_per_level[level]

            # Select top-k blocks at this level
            _, level_mask = self._select_top_k_blocks(x, block_size, k)
            selected_masks.append(level_mask)
            all_selected = all_selected | level_mask

        return selected_masks, all_selected

    def get_attention_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the indices of tokens that should attend to each other (sparse indices).

        For now, returns the union of all selected tokens.
        In full implementation, would build the actual sparse Q,K,V indices.

        Args:
            x: (seq_len, d) token embeddings

        Returns:
            (num_selected,) indices of tokens participating in sparse attention
        """
        _, all_selected = self.forward(x)
        indices = torch.where(all_selected)[0]
        return indices


class TrainableSparseAttention(nn.Module):
    """
    Trainable log-linear sparse attention mechanism.

    Replaces static Top-K block selection with learnable log-linear weights.
    Computes multi-head attention only between selected token blocks.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_levels: int = 3,
        max_seq_len: int = 512,
        dropout: float = 0.0
    ):
        """
        Args:
            d_model: embedding dimension
            num_heads: number of attention heads
            num_levels: number of hierarchical levels
            max_seq_len: maximum sequence length for initializing learnable weights
            dropout: dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_levels = num_levels
        self.max_seq_len = max_seq_len

        # Learnable log-linear block importance weights for each level
        # Each level has blocks of different sizes
        self.block_weights = nn.ParameterList()
        for level in range(num_levels):
            block_size = max(1, max_seq_len // (2 ** (num_levels - level)))
            max_blocks_at_level = (max_seq_len + block_size - 1) // block_size
            # Initialize log-linear weights (log probabilities)
            self.block_weights.append(
                nn.Parameter(torch.zeros(max_blocks_at_level))
            )

        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _select_important_blocks(
        self,
        seq_len: int,
        top_k_per_level: Optional[List[int]] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Select important blocks hierarchically using learnable log-linear weights.

        Args:
            seq_len: sequence length
            top_k_per_level: number of blocks to select at each level.
                            If None, uses adaptive sparsity.

        Returns:
            selected_masks: list of boolean masks for each level
            all_selected: union of all selected token indices
        """
        if top_k_per_level is None:
            top_k_per_level = []
            for level in range(self.num_levels):
                block_size = max(1, seq_len // (2 ** (self.num_levels - level)))
                num_blocks = (seq_len + block_size - 1) // block_size
                # Adaptive sparsity: coarser levels are more sparse
                sparsity_ratio = [0.125, 0.25, 0.5][min(level, 2)]
                k_at_level = max(1, int(num_blocks * sparsity_ratio))
                top_k_per_level.append(k_at_level)

        selected_masks = []
        all_selected = torch.zeros(seq_len, dtype=torch.bool, device=self.q_proj.weight.device)

        for level in range(self.num_levels):
            block_size = max(1, seq_len // (2 ** (self.num_levels - level)))
            num_blocks = (seq_len + block_size - 1) // block_size
            k = top_k_per_level[level]

            # Get learnable log-linear weights for this level
            level_weights = self.block_weights[level][:num_blocks]

            # Compute importance using log-linear scores: exp(w_i)
            importance_scores = torch.exp(level_weights)

            # Select top-k blocks
            k = min(k, num_blocks)
            _, top_block_ids = torch.topk(importance_scores, k)

            # Convert block IDs to token mask
            level_mask = torch.zeros(seq_len, dtype=torch.bool, device=all_selected.device)
            for block_id in top_block_ids:
                start = block_id.item() * block_size
                end = min((block_id.item() + 1) * block_size, seq_len)
                level_mask[start:end] = True

            selected_masks.append(level_mask)
            all_selected = all_selected | level_mask

        return selected_masks, all_selected

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with trainable sparse attention.

        Args:
            x: input tensor of shape (batch_size, seq_len, d_model) or (seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: same shape as input
        """
        # Handle 2D input (no batch dimension)
        squeeze_output = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        batch_size, seq_len, d = x.shape

        # Select important tokens using learnable log-linear weights
        _, selected_mask = self._select_important_blocks(seq_len)
        selected_indices = torch.where(selected_mask)[0]
        num_selected = len(selected_indices)

        # Project Q, K, V
        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Extract selected K, V for sparse attention
        # Q shape: (batch_size, num_heads, seq_len, head_dim)
        # K_sparse, V_sparse shape: (batch_size, num_heads, num_selected, head_dim)
        k_sparse = k[:, :, selected_indices, :]
        v_sparse = v[:, :, selected_indices, :]

        # Compute sparse attention scores
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, num_selected)
        scores = torch.matmul(q, k_sparse.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores shape: (batch_size, num_heads, seq_len, num_selected)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch_size, num_heads, seq_len, num_selected) @ (batch_size, num_heads, num_selected, head_dim)
        attn_output = torch.matmul(attn_weights, v_sparse)
        # attn_output shape: (batch_size, num_heads, seq_len, head_dim)

        # Reshape back to (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d)

        # Final output projection
        output = self.out_proj(attn_output)

        if squeeze_output:
            output = output.squeeze(0)

        return output


class DiffusionTransformerBlock(nn.Module):
    """
    A single transformer block for diffusion models using sparse attention.

    Combines sparse multi-head attention with feed-forward network and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_levels: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: embedding dimension
            num_heads: number of attention heads
            ff_dim: dimension of feed-forward inner layer
            num_levels: number of hierarchical levels for sparse attention
            dropout: dropout probability
        """
        super().__init__()
        self.d_model = d_model

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Sparse attention
        self.attention = TrainableSparseAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_levels=num_levels,
            dropout=dropout
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input tensor (batch_size, seq_len, d_model) or (seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: same shape as input
        """
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output

        # Feed-forward with residual connection
        ff_output = self.ff(self.norm2(x))
        x = x + ff_output

        return x


class SimpleDiffusionModel(nn.Module):
    """
    A simple diffusion transformer backbone using hierarchical sparse attention.

    Stacks multiple transformer blocks with sparse attention for efficient
    processing of long token sequences.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 2048,
        num_levels: int = 3,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: embedding dimension
            num_heads: number of attention heads
            num_layers: number of transformer layers
            ff_dim: dimension of feed-forward inner layer
            num_levels: number of hierarchical levels for sparse attention
            max_seq_len: maximum sequence length
            dropout: dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Transformer layers with sparse attention
        self.layers = nn.ModuleList([
            DiffusionTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_levels=num_levels,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input tensor (batch_size, seq_len, d_model) or (seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: same shape as input
        """
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x

    def compute_attention_complexity(self, seq_len: int) -> dict:
        """
        Compare theoretical complexity: sparse vs. full attention.

        Args:
            seq_len: sequence length

        Returns:
            dict with complexity metrics
        """
        # Full attention: O(N^2)
        full_attn_ops = seq_len ** 2

        # Sparse attention with hierarchical selection
        # Each level selects a fraction of tokens
        sparse_selected_tokens = 0
        for level in range(self.layers[0].attention.num_levels):
            block_size = max(1, seq_len // (2 ** (self.layers[0].attention.num_levels - level)))
            num_blocks = (seq_len + block_size - 1) // block_size
            sparsity_ratio = [0.125, 0.25, 0.5][min(level, 2)]
            selected_at_level = int(num_blocks * sparsity_ratio) * block_size
            sparse_selected_tokens = max(sparse_selected_tokens, selected_at_level)

        # Sparse attention: O(N * k) where k is number of selected tokens
        sparse_attn_ops = seq_len * sparse_selected_tokens

        return {
            "seq_len": seq_len,
            "full_attention_ops": full_attn_ops,
            "sparse_attention_ops": sparse_attn_ops,
            "tokens_selected": sparse_selected_tokens,
            "reduction_ratio": full_attn_ops / max(1, sparse_attn_ops),
        }
