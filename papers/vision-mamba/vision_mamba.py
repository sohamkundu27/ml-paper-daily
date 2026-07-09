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


class SelectiveSSMBlock(nn.Module):
    """Selective SSM block with input-dependent gating (Mamba's core mechanism)."""

    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Input projection to hidden dimension
        self.in_proj = nn.Linear(embed_dim, hidden_dim)

        # Gate control: learns to selectively activate parts of the state space
        self.gate_proj = nn.Linear(embed_dim, hidden_dim)

        # State transition matrix: initialize as near-identity with small perturbation
        # This helps stability: h_t ≈ h_{t-1} + input allows gradients to flow better
        A_init = torch.eye(hidden_dim) * 0.9 + torch.randn(hidden_dim, hidden_dim) * 0.05
        self.A = nn.Parameter(A_init)

        # Output projection back to embedding dimension
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, direction='forward'):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # Reverse sequence if processing backward
        if direction == 'backward':
            x = torch.flip(x, [1])

        # Project input and compute selective gate
        x_proj = self.in_proj(x)  # (batch, seq_len, hidden_dim)
        gate = torch.sigmoid(self.gate_proj(x))  # (batch, seq_len, hidden_dim)

        # Apply selective gating to input projection
        x_gated = x_proj * gate

        # Process through SSM step-by-step
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            x_t = x_gated[:, t, :]  # (batch, hidden_dim)
            # State update: h_t = A @ h_{t-1} + x_t
            h = torch.matmul(h, self.A.T) + x_t
            # Output: project back to embedding dimension
            y_t = self.out_proj(h)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=1)  # (batch, seq_len, embed_dim)

        # Reverse output if processing backward
        if direction == 'backward':
            out = torch.flip(out, [1])

        return out


class BidirectionalSSMBlock(nn.Module):
    """Bidirectional SSM: combines forward and backward selective SSM passes."""

    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.forward_ssm = SelectiveSSMBlock(embed_dim, hidden_dim)
        self.backward_ssm = SelectiveSSMBlock(embed_dim, hidden_dim)
        # Projection to combine forward and backward outputs
        self.combine_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        # Process forward
        out_fwd = self.forward_ssm(x, direction='forward')

        # Process backward
        out_bwd = self.backward_ssm(x, direction='backward')

        # Concatenate and project to combine
        out_combined = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.combine_proj(out_combined)

        return out


class VisionMambaBlock(nn.Module):
    """Vision Mamba block: selective bidirectional SSM + residual + layer norm."""

    def __init__(self, embed_dim, ssm_hidden_dim=128, use_bidirectional=False):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        if use_bidirectional:
            self.ssm = BidirectionalSSMBlock(embed_dim, hidden_dim=ssm_hidden_dim)
        else:
            self.ssm = LinearSSMBlock(embed_dim, hidden_dim=ssm_hidden_dim)

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
            VisionMambaBlock(embed_dim, ssm_hidden_dim=ssm_hidden_dim, use_bidirectional=False)
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


class VisionMambaPass2(nn.Module):
    """Vision Mamba with selective bidirectional SSM blocks (pass 2)."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 embed_dim=64, num_blocks=2, ssm_hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.patcher = ImagePatcher(image_size, patch_size, in_channels, embed_dim)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)

        # Use bidirectional selective SSM blocks
        self.blocks = nn.ModuleList([
            VisionMambaBlock(embed_dim, ssm_hidden_dim=ssm_hidden_dim, use_bidirectional=True)
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


class VisionMambaPass3(nn.Module):
    """Vision Mamba with classification head (pass 3)."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.patcher = ImagePatcher(image_size, patch_size, in_channels, embed_dim)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)

        # Use bidirectional selective SSM blocks
        self.blocks = nn.ModuleList([
            VisionMambaBlock(embed_dim, ssm_hidden_dim=ssm_hidden_dim, use_bidirectional=True)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head: global average pooling + linear layer
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.patcher(x)  # (batch, num_patches, embed_dim)
        x = self.pos_embed(x)  # Add positional embeddings

        for block in self.blocks:
            x = block(x)  # (batch, num_patches, embed_dim)

        x = self.norm(x)  # Final layer norm
        # Global average pooling over patches
        x = x.mean(dim=1)  # (batch, embed_dim)
        # Classification head
        logits = self.head(x)  # (batch, num_classes)
        return logits
