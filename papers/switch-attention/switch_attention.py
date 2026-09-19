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
    """Hybrid attention that dynamically routes between full and sliding window attention."""

    def __init__(self, d_model, num_heads, window_size=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size

        self.full_attention = FullAttention(d_model, num_heads, dropout)
        self.sliding_attention = SlidingWindowAttention(d_model, num_heads, window_size, dropout)

        # Simple router: takes sequence embedding and predicts routing probability
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Compute both attention paths
        full_output = self.full_attention(x, mask)
        sliding_output = self.sliding_attention(x, mask)

        # Compute routing probability per token using mean pooling over sequence
        routing_input = x.mean(dim=1)  # (batch, d_model)
        routing_prob = self.router(routing_input)  # (batch, 1)
        routing_prob = routing_prob.unsqueeze(1)  # (batch, 1, 1) for broadcasting

        # Interpolate between full and sliding window attention
        output = routing_prob * full_output + (1 - routing_prob) * sliding_output

        return output
