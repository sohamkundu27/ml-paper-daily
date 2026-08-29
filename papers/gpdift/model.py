import torch
import torch.nn as nn
import math


class RotaryPositionEmbedding(nn.Module):
    """Sinusoidal positional embeddings."""

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


class RotationTimeEmbedding(nn.Module):
    """Parameter-free rotation-based time embedding using rotation matrices."""

    def __init__(self, d_model, base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base

        # Precompute inverse dimensions for rotation
        self.inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("_inv_freq", self.inv_freq, persistent=False)

    def apply_rotation(self, x, t):
        """Apply rotation to embedding based on timestep."""
        # x: (batch, d_model) or (batch, seq_len, d_model)
        # t: (batch,) - timestep values

        # Compute rotation angle: theta = t * inv_freq
        # Shape: (batch, d_model // 2)
        theta = torch.outer(t.float(), self._inv_freq)

        # Expand dimensions for broadcasting
        # For 2D input: (batch, d_model) -> reshape and apply
        # For 3D input: (batch, seq_len, d_model) -> apply to last dim
        x_shape = x.shape

        if len(x_shape) == 2:
            batch_size, d_model = x_shape
            # Reshape x to (batch, d_model // 2, 2) for rotation pairs
            x_pairs = x.view(batch_size, -1, 2)  # (batch, d_model // 2, 2)

            # Apply 2D rotation: [x_i, x_{i+1}] -> [x_i*cos - x_{i+1}*sin, x_i*sin + x_{i+1}*cos]
            cos_theta = torch.cos(theta).unsqueeze(-1)  # (batch, d_model // 2, 1)
            sin_theta = torch.sin(theta).unsqueeze(-1)  # (batch, d_model // 2, 1)

            x_rot = torch.cat([
                x_pairs[..., 0:1] * cos_theta - x_pairs[..., 1:2] * sin_theta,
                x_pairs[..., 0:1] * sin_theta + x_pairs[..., 1:2] * cos_theta,
            ], dim=-1)

            # Reshape back to (batch, d_model)
            x_rot = x_rot.view(batch_size, d_model)

        else:  # len(x_shape) == 3
            batch_size, seq_len, d_model = x_shape
            # For sequence: apply same rotation to all positions
            x_pairs = x.view(batch_size, seq_len, -1, 2)  # (batch, seq_len, d_model // 2, 2)

            cos_theta = torch.cos(theta).unsqueeze(1).unsqueeze(-1)  # (batch, 1, d_model // 2, 1)
            sin_theta = torch.sin(theta).unsqueeze(1).unsqueeze(-1)  # (batch, 1, d_model // 2, 1)

            x_rot = torch.cat([
                x_pairs[..., 0:1] * cos_theta - x_pairs[..., 1:2] * sin_theta,
                x_pairs[..., 0:1] * sin_theta + x_pairs[..., 1:2] * cos_theta,
            ], dim=-1)

            x_rot = x_rot.view(batch_size, seq_len, d_model)

        return x_rot


class LinearCausalAttention(nn.Module):
    """Efficient linear causal attention using kernel methods."""

    def __init__(self, d_model, num_heads=8, kernel="elu"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kernel = kernel

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def kernel_fn(self, x):
        """Apply kernel function to transform attention scores."""
        if self.kernel == "elu":
            return torch.nn.functional.elu(x) + 1.0
        elif self.kernel == "relu":
            return torch.relu(x)
        else:
            return x

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

        # Apply kernel function
        Q_ker = self.kernel_fn(Q)
        K_ker = self.kernel_fn(K)

        # Linear attention: O = (Q_ker @ K_ker^T @ V) / (Q_ker @ K_ker^T @ 1)
        # For causal, accumulate from left to right
        if causal_mask:
            # Compute cumulative attention weights from left to right
            attn_num = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim, device=x.device)
            attn_denom = torch.zeros(batch_size, self.num_heads, seq_len, device=x.device)

            for i in range(seq_len):
                # For position i, only attend to positions 0..i
                K_causal = K_ker[:, :, :i+1, :]  # (batch, heads, i+1, head_dim)
                V_causal = V[:, :, :i+1, :]  # (batch, heads, i+1, head_dim)
                Q_i = Q_ker[:, :, i:i+1, :]  # (batch, heads, 1, head_dim)

                # Numerator: Q @ K^T @ V
                scores = torch.matmul(Q_i, K_causal.transpose(-2, -1))  # (batch, heads, 1, i+1)
                weighted_values = torch.matmul(scores, V_causal)  # (batch, heads, 1, head_dim)
                attn_num[:, :, i, :] = weighted_values.squeeze(2)

                # Denominator: Q @ K^T @ 1
                sum_scores = scores.sum(dim=-1)  # (batch, heads, 1)
                attn_denom[:, :, i] = sum_scores.squeeze(2) + 1e-8

            # Normalize
            context = attn_num / attn_denom.unsqueeze(-1)
        else:
            # Non-causal linear attention
            K_T_V = torch.matmul(K_ker.transpose(-2, -1), V)  # (batch, heads, head_dim, head_dim)
            K_T_1 = K_ker.sum(dim=2, keepdim=True)  # (batch, heads, 1, head_dim)

            numerator = torch.matmul(Q_ker, K_T_V)  # (batch, heads, seq_len, head_dim)
            denominator = torch.matmul(Q_ker, K_T_1.transpose(-2, -1))  # (batch, heads, seq_len, 1)

            context = numerator / (denominator + 1e-8)

        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, d_model)

        # Output projection
        out = self.out_proj(context)
        return out


class CausalSelfAttention(nn.Module):
    """Self-attention with causal masking (autoregressive) - standard softmax version."""

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
    """Transformer block with causal attention and FFN."""

    def __init__(self, d_model=256, num_heads=8, mlp_dim=1024, use_linear_attn=False):
        super().__init__()
        if use_linear_attn:
            self.attn = LinearCausalAttention(d_model, num_heads)
        else:
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

    def __init__(self, latent_dim=256, d_model=256, num_heads=8, num_blocks=1, use_linear_attn=False, use_rotation_time=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.use_rotation_time = use_rotation_time

        # Project latent to embedding space
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Time embedding
        if use_rotation_time:
            self.time_emb = RotationTimeEmbedding(d_model)
        else:
            self.time_emb = nn.Sequential(
                nn.Linear(1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        # Transformer blocks with optional linear attention
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_linear_attn:
                attn = LinearCausalAttention(d_model, num_heads)
            else:
                attn = CausalSelfAttention(d_model, num_heads)

            norm1 = nn.LayerNorm(d_model)
            norm2 = nn.LayerNorm(d_model)
            mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Linear(4 * d_model, d_model),
            )

            block = nn.ModuleDict({
                'attn': attn,
                'norm1': norm1,
                'norm2': norm2,
                'mlp': mlp,
            })
            self.blocks.append(block)

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
        if self.use_rotation_time:
            t_emb = self.time_emb.apply_rotation(x.squeeze(1), t)  # (batch, d_model)
            t_emb = t_emb.unsqueeze(1)  # (batch, 1, d_model)
        else:
            t_norm = t.float().unsqueeze(-1) / 1000.0
            t_emb = self.time_emb(t_norm).unsqueeze(1)  # (batch, 1, d_model)

        # Add time embedding
        x = x + t_emb

        # Process through transformer blocks
        for block in self.blocks:
            # Causal attention with residual
            x = x + block['attn'](block['norm1'](x), causal_mask=True)
            # MLP with residual
            x = x + block['mlp'](block['norm2'](x))

        # Output
        x = x.squeeze(1)  # (batch, d_model)
        noise_pred = self.out_proj(x)  # (batch, latent_dim)
        return noise_pred
