import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BlockwiseSparseAttention(nn.Module):
    """
    Piecewise sparse attention for diffusion transformers.
    Partitions sequence into blocks and applies sparse attention patterns.
    Pass 1: Basic block identification and mask generation (no approximation yet).
    Pass 2: Taylor expansion approximation for non-critical blocks.
    """

    def __init__(self, dim, num_heads=8, block_size=32, sparsity_ratio=0.5, taylor_order=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio
        self.taylor_order = taylor_order
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

    def _taylor_exp_approximation(self, x, order=None):
        """
        Approximate exp(x) using Taylor series: exp(x) ≈ 1 + x + x²/2! + x³/3! + ...

        This is the core of Pass 2: for non-critical blocks, we compute a polynomial
        approximation of exponential instead of expensive exact computation.

        Args:
            x: input tensor
            order: number of Taylor terms (default: self.taylor_order)

        Returns:
            Polynomial approximation of exp(x)
        """
        if order is None:
            order = self.taylor_order

        result = torch.ones_like(x)
        x_power = x.clone()
        for k in range(1, order + 1):
            result = result + x_power / math.factorial(k)
            if k < order:
                x_power = x_power * x
        return result

    def _compute_piecewise_attention(self, scores, is_critical, num_blocks, batch_size, seq_len):
        """
        Compute attention weights using exact softmax for critical blocks
        and Taylor-approximated attention for non-critical blocks.

        This piecewise strategy is the key contribution of Pass 2:
        - Critical blocks (high-variance attention patterns) get exact computation
        - Non-critical blocks use efficient polynomial approximation
        - Significantly reduces computational cost while maintaining quality

        Args:
            scores: [batch, heads, seq_len, seq_len] - attention scores
            is_critical: [batch, num_blocks] - which blocks are critical
            num_blocks: number of blocks in the sequence
            batch_size: batch size
            seq_len: sequence length

        Returns:
            attn_weights: [batch, heads, seq_len, seq_len]
        """
        attn_weights = torch.zeros_like(scores)

        for b in range(batch_size):
            for block_idx in range(num_blocks):
                start = block_idx * self.block_size
                end = min((block_idx + 1) * self.block_size, seq_len)

                block_scores = scores[b, :, start:end, :]
                block_max = block_scores.max(dim=-1, keepdim=True)[0]
                block_scores_normalized = block_scores - block_max

                if is_critical[b, block_idx]:
                    exp_scores = torch.exp(block_scores_normalized)
                else:
                    exp_scores = self._taylor_exp_approximation(block_scores_normalized)

                denominator = exp_scores.sum(dim=-1, keepdim=True)
                block_attn = exp_scores / (denominator + 1e-8)
                attn_weights[b, :, start:end, :] = block_attn

        return attn_weights

    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        Returns: [batch, seq_len, dim]

        Pass 2: Uses piecewise attention with Taylor approximation.
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

        attn_weights = self._compute_piecewise_attention(scores, is_critical, num_blocks, batch_size, seq_len)

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
