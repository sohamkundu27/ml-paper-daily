"""Binary tokenizer for autoregressive image generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Simple conv block with ReLU and batchnorm."""
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BinaryTokenizer(nn.Module):
    """
    Encodes images to binary tokens and decodes tokens back to images.

    Encoder: image -> spatial compression -> binary embedding
    Decoder: binary embedding -> spatial expansion -> image
    """

    def __init__(self, in_channels=3, token_dim=32, num_tokens_spatial=8):
        """
        Args:
            in_channels: number of input image channels (3 for RGB)
            token_dim: dimensionality of each token representation
            num_tokens_spatial: spatial resolution of token grid (8 -> 8x8 token map for 256x256 image)
        """
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.num_tokens_spatial = num_tokens_spatial

        # Encoder: compress image to token grid
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64, kernel=4, stride=2, padding=1),
            ConvBlock(64, 128, kernel=4, stride=2, padding=1),
            ConvBlock(128, 256, kernel=4, stride=2, padding=1),
            ConvBlock(256, token_dim, kernel=3, stride=1, padding=1),
        )

        # Decoder: expand token grid back to image
        self.decoder = nn.Sequential(
            ConvBlock(token_dim, 256, kernel=3, stride=1, padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x):
        """
        Encode image to binary token embedding.

        Args:
            x: input image tensor (B, C, H, W) normalized to [0, 1]

        Returns:
            tokens: token embedding (B, token_dim, H/8, W/8)
            binary_tokens: binarized tokens via threshold (B, token_dim, H/8, W/8)
        """
        tokens = self.encoder(x)

        # Binarize: threshold at 0.5
        binary_tokens = (tokens > 0.5).float()

        return tokens, binary_tokens

    def decode(self, tokens):
        """
        Decode token embedding back to image.

        Args:
            tokens: token embedding (B, token_dim, H/8, W/8)

        Returns:
            image: reconstructed image (B, C, H, W) in [0, 1]
        """
        x = self.decoder(tokens)
        # Clamp to valid image range
        x = torch.clamp(x, 0, 1)
        return x

    def encode_decode(self, x):
        """
        Full encode-decode cycle.

        Args:
            x: input image (B, C, H, W) in [0, 1]

        Returns:
            reconstructed: decoded image (B, C, H, W) in [0, 1]
            binary_tokens: binarized token representation
        """
        tokens, binary_tokens = self.encode(x)
        reconstructed = self.decode(binary_tokens)
        return reconstructed, binary_tokens


class BinaryTokenizerLoss(nn.Module):
    """Reconstruction loss for tokenizer training."""

    def __init__(self, reconstruction_weight=1.0, perplexity_weight=0.01):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.perplexity_weight = perplexity_weight

    def forward(self, x_orig, x_recon, tokens):
        """
        Args:
            x_orig: original image
            x_recon: reconstructed image
            tokens: soft token embeddings before binarization

        Returns:
            total_loss: combined loss
        """
        # Reconstruction loss (MSE between original and reconstructed)
        recon_loss = F.mse_loss(x_orig, x_recon)

        # Encourage binary tokens: push values toward 0 or 1
        # Minimize entropy by encouraging confident predictions
        binary_targets = (tokens > 0.5).float()
        binary_loss = F.mse_loss(tokens, binary_targets)

        total_loss = (self.reconstruction_weight * recon_loss +
                     self.perplexity_weight * binary_loss)

        return total_loss, recon_loss, binary_loss
