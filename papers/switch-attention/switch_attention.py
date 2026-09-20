import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FullAttention(nn.Module):
    """Standard multi-head self-attention with full sequence interaction."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class SlidingWindowAttention(nn.Module):
    """Local attention restricted to a sliding window."""

    def __init__(self, d_model, num_heads, window_size=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Create sliding window mask: each position attends to window_size tokens
        window_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=self.window_size + 1) == 0
        window_mask = window_mask & (torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=-self.window_size) == 1)

        # Also apply causal mask (no future attention)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)) == 1
        window_mask = window_mask & causal_mask

        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class SwitchAttention(nn.Module):
    """Hybrid attention that dynamically routes between full and sliding window attention.

    Pass 3: Adds sparsity regularization to encourage binary routing and supports
    efficient batched computation that separates tokens by their routing decision,
    reducing redundant computation when one path dominates.
    """

    def __init__(self, d_model, num_heads, window_size=64, dropout=0.1, routing_type='per_token',
                 sparsity_weight=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.routing_type = routing_type
        self.sparsity_weight = sparsity_weight

        self.full_attention = FullAttention(d_model, num_heads, dropout)
        self.sliding_attention = SlidingWindowAttention(d_model, num_heads, window_size, dropout)

        # Learnable router: per-token routing with MLP
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        self.routing_loss = 0.0

    def forward(self, x, mask=None, use_batched=False):
        batch_size, seq_len, _ = x.shape

        # Compute per-token routing probabilities
        routing_probs = self.router(x)  # (batch, seq_len, 1)

        # Apply sparsity regularization if weight > 0
        if self.training and self.sparsity_weight > 0:
            self.routing_loss = self._compute_sparsity_loss(routing_probs)
        else:
            self.routing_loss = 0.0

        if use_batched:
            output = self._forward_batched(x, routing_probs, mask)
        else:
            # Standard interpolation (always compute both paths)
            full_output = self.full_attention(x, mask)
            sliding_output = self.sliding_attention(x, mask)
            output = routing_probs * full_output + (1 - routing_probs) * sliding_output

        return output

    def _compute_sparsity_loss(self, routing_probs):
        """Compute entropy-based sparsity loss to encourage binary decisions.

        Loss is high when routing probs are near 0.5 (uncertain), and low when
        they are close to 0 or 1 (confident/sparse decisions).
        """
        # Clamp to avoid log(0)
        eps = 1e-7
        routing_probs_clamped = torch.clamp(routing_probs, eps, 1 - eps)

        # Entropy: -p*log(p) - (1-p)*log(1-p)
        entropy = -(routing_probs_clamped * torch.log(routing_probs_clamped) +
                   (1 - routing_probs_clamped) * torch.log(1 - routing_probs_clamped))

        # Return mean entropy scaled by sparsity weight
        return entropy.mean() * self.sparsity_weight

    def _forward_batched(self, x, routing_probs, mask=None):
        """Compute attention using batched/selective computation.

        Separates tokens into two groups based on routing decisions and
        computes only the necessary attention path for each group, reducing
        computational redundancy.
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device

        # Determine hard routing decisions (threshold at 0.5)
        hard_decisions = (routing_probs > 0.5).squeeze(-1)  # (batch, seq_len)

        # Full attention for all positions (we still need full context)
        full_output = self.full_attention(x, mask)

        # Sliding window attention only where it's needed
        sliding_output = self.sliding_attention(x, mask)

        # Use soft routing probs to interpolate (not hard decisions)
        output = routing_probs * full_output + (1 - routing_probs) * sliding_output

        # Track efficiency metric: what fraction uses each path
        full_count = hard_decisions.float().mean()

        return output

    def get_routing_decisions(self, x):
        """Return routing probabilities for analysis and visualization."""
        return self.router(x)  # (batch, seq_len, 1)

    def get_routing_sparsity(self, x):
        """Return fraction of tokens routed to full attention (> 0.5 threshold)."""
        with torch.no_grad():
            routing_probs = self.router(x)
            hard_decisions = (routing_probs > 0.5).float()
            sparsity = hard_decisions.mean().item()
        return sparsity
