import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImagePatcher(nn.Module):
    """Convert images to patch embeddings."""

    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        num_patches = (image_size // patch_size) ** 2
        patch_embed_size = in_channels * patch_size * patch_size

        self.proj = nn.Linear(patch_embed_size, embed_dim)

    def forward(self, x):
        """
        Convert image to patches.

        Args:
            x: (B, C, H, W)

        Returns:
            patches: (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        P = self.patch_size

        x = x.reshape(
            B,
            C,
            H // P,
            P,
            W // P,
            P
        )
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, (H // P) * (W // P), C * P * P)

        return self.proj(x)


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings for image patches."""

    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

    def forward(self, x):
        """x: (B, T, D)"""
        return x + self.pos_embed


class LinearSSMBlock(nn.Module):
    """Simple linear state space model for sequence processing.

    This is a simplified SSM, not the full Mamba algorithm.
    It learns to transform sequences through learnable state transitions.
    """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Learnable state transition matrices
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, embed_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(embed_dim, hidden_dim) * 0.01)

        # Scale factor for numerical stability
        self.scale = 1.0 / np.sqrt(hidden_dim)

    def forward(self, x):
        """
        Process sequence through SSM.

        Args:
            x: (B, T, D) tensor of patch embeddings

        Returns:
            y: (B, T, D) tensor of output features
        """
        B, T, D = x.shape

        # Initialize hidden state
        h = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        outputs = []

        # Forward pass through sequence
        for t in range(T):
            x_t = x[:, t, :]  # (B, D)

            # Update state: h_{t+1} = A h_t + B x_t
            h = torch.matmul(h, self.A.t()) + torch.matmul(x_t, self.B.t())
            h = h * self.scale

            # Output: y_t = C h_t
            y_t = torch.matmul(h, self.C.t())  # (B, D)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, T, D)


class SelectiveSSMBlock(nn.Module):
    """Selective SSM block with gating and directional support.

    Unlike LinearSSMBlock, SelectiveSSM includes learnable gating to selectively
    attend to different sequence positions based on input content.
    """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # State transition matrices
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, embed_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(embed_dim, hidden_dim) * 0.01)

        # Selective gating: learn when to update state
        self.gate_proj = nn.Linear(embed_dim, hidden_dim)

        self.scale = 1.0 / np.sqrt(hidden_dim)

    def forward(self, x, direction='forward'):
        """
        Process sequence through selective SSM.

        Args:
            x: (B, T, D) tensor of patch embeddings
            direction: 'forward' or 'backward' for scanning direction

        Returns:
            y: (B, T, D) tensor of output features
        """
        if direction == 'backward':
            x = torch.flip(x, dims=[1])

        B, T, D = x.shape

        h = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            x_t = x[:, t, :]

            # Compute gating weights based on input
            gate = torch.sigmoid(self.gate_proj(x_t))  # (B, hidden_dim)

            # Update state with gating
            h_update = torch.matmul(h, self.A.t()) + torch.matmul(x_t, self.B.t())
            h = gate * h_update + (1 - gate) * h
            h = h * self.scale

            # Output
            y_t = torch.matmul(h, self.C.t())
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        if direction == 'backward':
            y = torch.flip(y, dims=[1])

        return y


class BidirectionalSSMBlock(nn.Module):
    """Bidirectional SSM block using selective scanning.

    Scans forward and backward through the sequence using SelectiveSSMBlock,
    then combines outputs with learnable gating.
    """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.forward_ssm = SelectiveSSMBlock(embed_dim, hidden_dim)
        self.backward_ssm = SelectiveSSMBlock(embed_dim, hidden_dim)

        # Learnable gating to combine forward and backward outputs
        self.gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        """
        Apply bidirectional SSM scanning.

        Args:
            x: (B, T, D)

        Returns:
            y: (B, T, D)
        """
        # Forward and backward passes
        y_fwd = self.forward_ssm(x, direction='forward')
        y_bwd = self.backward_ssm(x, direction='backward')

        # Combine outputs
        combined = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.gate(combined)

        return y


class VisionMambaBlock(nn.Module):
    """Vision Mamba block with configurable SSM type and directionality."""

    def __init__(self, embed_dim, ssm_hidden_dim, use_bidirectional=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.ssm_hidden_dim = ssm_hidden_dim
        self.use_bidirectional = use_bidirectional

        if use_bidirectional:
            self.ssm = BidirectionalSSMBlock(embed_dim, ssm_hidden_dim)
        else:
            self.ssm = LinearSSMBlock(embed_dim, ssm_hidden_dim)

        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) where T is number of patches, D is patch dimension

        Returns:
            y: (B, T, D)
        """
        y = self.ssm(x)
        y = y + x  # Residual
        y = self.ln(y)
        return y


class VisionMambaPass1(nn.Module):
    """Vision Mamba Pass 1: Core model with linear SSM blocks."""

    def __init__(self, image_size, patch_size, in_channels, embed_dim,
                 num_blocks, ssm_hidden_dim):
        super().__init__()
        self.patcher = ImagePatcher(image_size, patch_size, in_channels, embed_dim)

        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)

        self.blocks = nn.Sequential(*[
            VisionMambaBlock(embed_dim, ssm_hidden_dim, use_bidirectional=False)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            features: (B, num_patches, embed_dim)
        """
        x = self.patcher(x)
        x = self.pos_embed(x)
        x = self.blocks(x)
        return x


class VisionMambaPass2(nn.Module):
    """Vision Mamba Pass 2: Bidirectional selective SSM blocks."""

    def __init__(self, image_size, patch_size, in_channels, embed_dim,
                 num_blocks, ssm_hidden_dim):
        super().__init__()
        self.patcher = ImagePatcher(image_size, patch_size, in_channels, embed_dim)

        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)

        self.blocks = nn.Sequential(*[
            VisionMambaBlock(embed_dim, ssm_hidden_dim, use_bidirectional=True)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            features: (B, num_patches, embed_dim)
        """
        x = self.patcher(x)
        x = self.pos_embed(x)
        x = self.blocks(x)
        return x
