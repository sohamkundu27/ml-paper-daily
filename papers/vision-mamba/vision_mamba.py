import torch
import torch.nn as nn
import math


class ImagePatcher(nn.Module):
    """Convert images to patch embeddings."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.proj(x)  # (batch, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Pre-compute sinusoidal embeddings
        pe = torch.zeros(num_patches, embed_dim)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, num_patches, embed_dim)
        return x + self.pe[:, :x.size(1), :]


class LinearSSMBlock(nn.Module):
    """Simplified unidirectional linear state space model block."""

    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # State transition: A matrix (hidden_dim, hidden_dim)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

        # Input projection: B matrix (embed_dim, hidden_dim)
        self.B = nn.Linear(embed_dim, hidden_dim)

        # Output projection: C matrix (hidden_dim, embed_dim)
        self.C = nn.Linear(hidden_dim, embed_dim)

        # Bias
        self.D = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # Initialize state
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            # h_t = A @ h_{t-1} + B @ x_t
            x_t = x[:, t, :]  # (batch, embed_dim)
            h = torch.matmul(h, self.A.T) + self.B(x_t)  # (batch, hidden_dim)

            # y_t = C @ h_t + D @ x_t
            y_t = self.C(h) + self.D * x_t  # (batch, embed_dim)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, embed_dim)


class VisionMambaBlock(nn.Module):
    """Vision Mamba block: position embedding + SSM + residual."""

    def __init__(self, embed_dim, ssm_hidden_dim=128):
        super().__init__()
        self.ssm = LinearSSMBlock(embed_dim, hidden_dim=ssm_hidden_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, num_patches, embed_dim)
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        return x + residual


class VisionMambaPass1(nn.Module):
    """Minimal Vision Mamba implementation for pass 1."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 embed_dim=64, num_blocks=2, ssm_hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.patcher = ImagePatcher(image_size, patch_size, in_channels, embed_dim)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)

        self.blocks = nn.ModuleList([
            VisionMambaBlock(embed_dim, ssm_hidden_dim=ssm_hidden_dim)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.patcher(x)  # (batch, num_patches, embed_dim)
        x = self.pos_embed(x)  # Add positional embeddings

        for block in self.blocks:
            x = block(x)  # (batch, num_patches, embed_dim)

        x = self.norm(x)  # Final layer norm
        return x  # (batch, num_patches, embed_dim)
