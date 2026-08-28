import torch
import torch.nn as nn
import math


class RotaryPositionEmbedding(nn.Module):
    """Simple sinusoidal positional embeddings."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, seq_len, device):
        """Generate sinusoidal position embeddings."""
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        dim = torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
        angle_rates = 1 / (10000 ** (dim / self.d_model))

        angle_rads = position.unsqueeze(1) * angle_rates
        pos_encoding = torch.zeros(seq_len, self.d_model, device=device)
        pos_encoding[:, 0::2] = torch.sin(angle_rads)
        pos_encoding[:, 1::2] = torch.cos(angle_rads)

        return pos_encoding


class CausalSelfAttention(nn.Module):
    """Self-attention with causal masking (autoregressive)."""

    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal_mask=True):
        batch_size, seq_len, d_model = x.shape

        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if causal_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, d_model)

        # Output projection
        out = self.out_proj(context)
        return out


class TransformerBlock(nn.Module):
    """Single transformer block with causal attention and FFN."""

    def __init__(self, d_model=256, num_heads=8, mlp_dim=1024):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x):
        # Causal attention with residual
        x = x + self.attn(self.norm1(x), causal_mask=True)
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class LatentEncoder(nn.Module):
    """Simple toy encoder: projects images to latent space."""

    def __init__(self, in_channels=3, latent_dim=256, height=32, width=32):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.spatial_dim = height * width

        # Simple projection: flatten and project
        self.proj = nn.Linear(in_channels * height * width, latent_dim)

    def forward(self, x):
        """x: (batch, channels, height, width) -> (batch, latent_dim)"""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.proj(x_flat)


class LatentDecoder(nn.Module):
    """Simple toy decoder: projects latent back to image space."""

    def __init__(self, latent_dim=256, out_channels=3, height=32, width=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.height = height
        self.width = width

        # Simple projection: project and reshape
        self.proj = nn.Linear(latent_dim, out_channels * height * width)

    def forward(self, z):
        """z: (batch, latent_dim) -> (batch, out_channels, height, width)"""
        x = self.proj(z)
        x = x.view(-1, self.out_channels, self.height, self.width)
        return x


class DenoisingModel(nn.Module):
    """Denoising model: predicts noise from noisy latent + timestep."""

    def __init__(self, latent_dim=256, d_model=256, num_heads=8, num_blocks=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        # Project latent to embedding space
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Time embedding (simple)
        self.time_emb = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_blocks)])

        # Output projection
        self.out_proj = nn.Linear(d_model, latent_dim)

    def forward(self, z_t, t):
        """
        Predict noise.
        z_t: (batch, latent_dim) - noisy latent
        t: (batch,) - timestep
        """
        # Project latent
        x = self.latent_proj(z_t)
        batch_size = x.shape[0]
        x = x.unsqueeze(1)  # (batch, 1, d_model)

        # Time embedding
        t_norm = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_emb(t_norm).unsqueeze(1)  # (batch, 1, d_model)

        # Add time embedding
        x = x + t_emb

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = x.squeeze(1)  # (batch, d_model)
        noise_pred = self.out_proj(x)  # (batch, latent_dim)
        return noise_pred
