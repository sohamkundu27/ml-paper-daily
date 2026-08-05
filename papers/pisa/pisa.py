import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BlockwiseSparseAttention(nn.Module):
    """
    Piecewise sparse attention for diffusion transformers.
    Partitions sequence into blocks and applies sparse attention patterns.
    Pass 1: Basic block identification and mask generation (no approximation yet).
    """

    def __init__(self, dim, num_heads=8, block_size=32, sparsity_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _compute_block_importance(self, attn_scores):
        """
        Identify critical blocks based on attention score statistics.
        attn_scores: [batch, heads, seq_len, seq_len]
        Returns: [batch, num_blocks] boolean tensor
        """
        batch_size, num_heads, seq_len, _ = attn_scores.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        block_importance = []
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min((block_idx + 1) * self.block_size, seq_len)
            block = attn_scores[:, :, start:end, :]
            importance = block.abs().max(dim=-1)[0].mean(dim=(1, 2))
            block_importance.append(importance)

        importance_scores = torch.stack(block_importance, dim=1)

        num_critical = max(1, int(num_blocks * (1 - self.sparsity_ratio)))
        _, critical_block_indices = torch.topk(
            importance_scores, num_critical, dim=1, largest=True
        )

        is_critical = torch.zeros(
            batch_size, num_blocks, dtype=torch.bool, device=attn_scores.device
        )
        for b in range(batch_size):
            is_critical[b, critical_block_indices[b]] = True

        return is_critical, num_blocks

    def _create_sparse_mask(self, seq_len, is_critical, num_blocks, batch_size):
        """
        Create attention mask for sparse computation.
        Allows attention within critical blocks and within local neighborhoods.
        is_critical: [batch_size, num_blocks]
        Returns: [batch_size, seq_len, seq_len]
        """
        masks = []
        for b in range(batch_size):
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=is_critical.device)
            for block_idx in range(num_blocks):
                start = block_idx * self.block_size
                end = min((block_idx + 1) * self.block_size, seq_len)

                if is_critical[b, block_idx]:
                    mask[start:end, :] = True
                    mask[:, start:end] = True

                mask[start:end, start:end] = True

            masks.append(mask)

        return torch.stack(masks, dim=0)

    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        Returns: [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        is_critical, num_blocks = self._compute_block_importance(scores)

        sparse_mask = self._create_sparse_mask(seq_len, is_critical, num_blocks, batch_size)
        sparse_mask = sparse_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        scores_masked = scores.clone()
        scores_masked[~sparse_mask] = float("-inf")

        attn_weights = F.softmax(scores_masked, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.out_proj(out)

        return out

    def get_sparsity_ratio(self, seq_len):
        """Return the actual sparsity of attention."""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        num_critical = max(1, int(num_blocks * (1 - self.sparsity_ratio)))
        allowed_positions = num_critical * self.block_size * seq_len
        allowed_positions += seq_len * self.block_size
        allowed_positions += seq_len * self.block_size
        total_positions = seq_len * seq_len
        return 1.0 - (allowed_positions / total_positions)
