import torch
import torch.nn as nn
from typing import Tuple, List


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
