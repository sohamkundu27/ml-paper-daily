import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional encoding for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: [B] timestep indices."""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SimpleViTBlock(nn.Module):
    """Single transformer block with self-attention and MLP."""
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: [B, N, D]."""
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionTransformer(nn.Module):
    """Minimalist Vision Transformer for diffusion on point maps."""
    def __init__(self, point_dim=3, patch_size=16, num_layers=4, hidden_dim=64, num_heads=4):
        super().__init__()
        self.point_dim = point_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Embed point patches to hidden_dim
        self.patch_embed = nn.Linear(patch_size * point_dim, hidden_dim)

        # Positional embedding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_dim))  # max 256 patches
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep embedding
        self.time_embed = SinusoidalPositionEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Transformer layers
        self.transformer = nn.ModuleList([
            SimpleViTBlock(hidden_dim, num_heads, hidden_dim * 4, dropout=0.1)
            for _ in range(num_layers)
        ])

        # Output head: predict noise in point space
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_head = nn.Linear(hidden_dim, patch_size * point_dim)

    def forward(self, x, timesteps):
        """
        x: [B, N, patch_size*point_dim] - flattened point patches
        timesteps: [B] - diffusion timestep indices
        Returns: [B, N, patch_size*point_dim] - predicted noise
        """
        B, N, _ = x.shape

        # Embed patches
        x = self.patch_embed(x)  # [B, N, hidden_dim]

        # Add positional embedding
        x = x + self.pos_embed[:, :N, :]

        # Embed and project timesteps
        t_emb = self.time_embed(timesteps)  # [B, hidden_dim]
        t_proj = self.time_mlp(t_emb)  # [B, hidden_dim]

        # Broadcast time embedding across patches
        t_proj = t_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        x = x + t_proj

        # Apply transformer blocks
        for block in self.transformer:
            x = block(x)

        # Output head
        x = self.norm_out(x)
        x = self.out_head(x)  # [B, N, patch_size*point_dim]

        return x


class GaussianDiffusion:
    """Linear Gaussian diffusion schedule and process."""
    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps

        # Linear schedule for betas
        betas = torch.linspace(beta_start, beta_end, num_steps)

        # Pre-compute alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def register_buffer(self, name, tensor):
        """Store buffers in a dict for portability."""
        if not hasattr(self, "_buffers"):
            self._buffers = {}
        self._buffers[name] = tensor

    def get_buffer(self, name):
        """Retrieve buffer."""
        return self._buffers.get(name)

    def forward_diffusion(self, x0, t):
        """
        Add noise to x0 at timestep t.
        x0: [B, ...] - original point maps
        t: [B] - timestep indices (0 to num_steps-1)
        Returns: xt [B, ...], noise [B, ...]
        """
        noise = torch.randn_like(x0)
        sqrt_alpha = self.get_buffer("sqrt_alphas_cumprod")[t]
        sqrt_one_minus_alpha = self.get_buffer("sqrt_one_minus_alphas_cumprod")[t]

        # Reshape for broadcasting
        while len(sqrt_alpha.shape) < len(x0.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        while len(sqrt_one_minus_alpha.shape) < len(x0.shape):
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return xt, noise

    def reverse_diffusion(self, model, xt, t, device):
        """
        Single reverse diffusion step: predict noise and step back.
        xt: [B, ...] - noisy point maps
        t: [B] - timestep indices
        Returns: x_{t-1} [B, ...]
        """
        with torch.no_grad():
            predicted_noise = model(xt, t)

            betas_t = self.get_buffer("betas")[t]
            sqrt_alpha = self.get_buffer("sqrt_alphas_cumprod")[t]
            sqrt_one_minus_alpha = self.get_buffer("sqrt_one_minus_alphas_cumprod")[t]

            # Reshape for broadcasting
            while len(betas_t.shape) < len(xt.shape):
                betas_t = betas_t.unsqueeze(-1)
            while len(sqrt_alpha.shape) < len(xt.shape):
                sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            while len(sqrt_one_minus_alpha.shape) < len(xt.shape):
                sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

            # Reverse step: x_{t-1} ~ N(mean, var)
            mean = (xt - betas_t * predicted_noise / sqrt_one_minus_alpha) / torch.sqrt(1.0 - betas_t)
            variance = betas_t
            z = torch.randn_like(xt) if t[0] > 0 else torch.zeros_like(xt)
            x_prev = mean + torch.sqrt(variance) * z

            return x_prev
