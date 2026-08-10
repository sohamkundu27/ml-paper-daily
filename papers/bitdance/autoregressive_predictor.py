"""Autoregressive token predictor with caching and multi-scale generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from binary_diffusion import BinaryDiffusionHead, BinaryDiffusionSampler


class TokenEmbedding(nn.Module):
    """Learnable embedding for binary tokens."""

    def __init__(self, token_dim, embed_dim):
        """
        Args:
            token_dim: dimensionality of input tokens
            embed_dim: dimensionality of embeddings
        """
        super().__init__()
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(token_dim, embed_dim)

    def forward(self, tokens):
        """
        Embed flattened token sequence.

        Args:
            tokens: (B, T, token_dim) where T is sequence length

        Returns:
            embeddings: (B, T, embed_dim)
        """
        return self.embed(tokens)


class AutoregressiveTokenPredictor(nn.Module):
    """
    Autoregressive model that predicts the next token given previous tokens.

    Uses transformer-style self-attention to capture dependencies among tokens,
    then uses the diffusion head for refinement.
    """

    def __init__(self, token_dim=32, embed_dim=128, num_heads=4, depth=2, num_timesteps=1000):
        """
        Args:
            token_dim: dimensionality of token embeddings
            embed_dim: dimensionality of transformer embeddings
            num_heads: number of attention heads
            depth: number of transformer layers
            num_timesteps: number of diffusion timesteps
        """
        super().__init__()
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps

        # Token embedding
        self.token_embed = TokenEmbedding(token_dim, embed_dim)

        # Positional encoding for sequence positions
        self.pos_embed = nn.Embedding(512, embed_dim)  # Support sequences up to 512 tokens

        # Transformer encoder for capturing token dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Projection back to token space
        self.token_proj = nn.Linear(embed_dim, token_dim)

        # Diffusion head for refinement
        self.diffusion_head = BinaryDiffusionHead(
            token_dim=token_dim, hidden_dim=128, num_timesteps=num_timesteps
        )

        # Cache for embeddings (for efficiency during autoregressive generation)
        self.cache_embeds = None
        self.cache_pos = None

    def forward(self, tokens):
        """
        Predict refined tokens given input tokens.

        Args:
            tokens: (B, T, token_dim) sequence of binary tokens

        Returns:
            pred_tokens: (B, T, token_dim) predicted refined tokens
        """
        batch_size, seq_len, token_dim = tokens.shape

        # Embed tokens
        embeds = self.token_embed(tokens)  # (B, T, embed_dim)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=tokens.device)
        pos_embeds = self.pos_embed(positions)  # (T, embed_dim)
        embeds = embeds + pos_embeds.unsqueeze(0)  # (B, T, embed_dim)

        # Self-attention to capture dependencies
        refined_embeds = self.transformer(embeds)  # (B, T, embed_dim)

        # Project back to token space
        pred_tokens = self.token_proj(refined_embeds)  # (B, T, token_dim)

        return pred_tokens

    def forward_with_diffusion(self, tokens, t=None, noise_scale=1.0):
        """
        Predict refined tokens using transformer.

        The diffusion head is designed for spatial grids, not sequences.
        For sequence-based predictions, we rely on the transformer.

        Args:
            tokens: (B, T, token_dim) sequence of binary tokens
            t: optional timestep (ignored for sequences, used for spatial grid mode)
            noise_scale: noise scale for diffusion

        Returns:
            pred_tokens: (B, T, token_dim) predictions
        """
        # Get transformer predictions
        pred_logits = self.forward(tokens)
        return pred_logits

    def set_cache(self, cache_embeds, cache_pos):
        """Store cached embeddings for efficient generation."""
        self.cache_embeds = cache_embeds
        self.cache_pos = cache_pos

    def clear_cache(self):
        """Clear cached embeddings."""
        self.cache_embeds = None
        self.cache_pos = None


class MultiScaleTokenGenerator:
    """
    Generates tokens using coarse-to-fine multi-scale strategy.

    Predicts coarse tokens first, then refines at progressively finer scales.
    """

    def __init__(self, predictor, diffusion_sampler=None, num_scales=3):
        """
        Args:
            predictor: AutoregressiveTokenPredictor model
            diffusion_sampler: BinaryDiffusionSampler for token generation
            num_scales: number of scales for coarse-to-fine generation
        """
        self.predictor = predictor
        self.diffusion_sampler = diffusion_sampler
        self.num_scales = num_scales

    def generate(self, initial_tokens, scales=None, num_diffusion_steps=50, device="cpu"):
        """
        Generate refined tokens using coarse-to-fine strategy.

        Args:
            initial_tokens: (B, C, H, W) initial token grid at coarsest scale
            scales: list of spatial scales to refine through (e.g., [4, 8, 16])
            num_diffusion_steps: number of diffusion steps for refinement
            device: device to generate on

        Returns:
            refined_tokens: final refined tokens
        """
        if scales is None:
            scales = [4, 8, 16][:self.num_scales]

        current_tokens = initial_tokens.to(device)
        batch_size, token_dim = current_tokens.shape[0], current_tokens.shape[1]

        # For each scale, refine tokens
        for scale in scales:
            # Flatten tokens to sequence for autoregressive prediction
            B, C, H, W = current_tokens.shape
            flattened = current_tokens.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)

            with torch.no_grad():
                # Get predictions from autoregressive model
                pred_tokens = self.predictor.forward_with_diffusion(flattened)

            # Reshape back to spatial
            pred_tokens = pred_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)

            # Use diffusion sampler for refinement if available
            if self.diffusion_sampler is not None:
                # Sample refined tokens via diffusion
                refined = self.diffusion_sampler.sample(
                    pred_tokens.shape, num_steps=num_diffusion_steps, device=device
                )
                current_tokens = refined
            else:
                # Otherwise use predictions directly
                current_tokens = (pred_tokens > 0.5).float()

        return current_tokens

    def generate_autoregressive(self, batch_size, seq_len, token_dim, device="cpu"):
        """
        Generate token sequence autoregressively.

        Args:
            batch_size: batch size
            seq_len: sequence length to generate
            token_dim: dimensionality of tokens
            device: device to generate on

        Returns:
            tokens: (B, seq_len, token_dim) generated tokens
        """
        # Start with random seed tokens
        tokens = torch.randn(batch_size, 1, token_dim, device=device)

        with torch.no_grad():
            for step in range(1, seq_len):
                # Predict next tokens given previous
                pred = self.predictor(tokens)  # (B, current_len, token_dim)

                # Take last prediction
                next_token = pred[:, -1:, :]  # (B, 1, token_dim)

                # Binarize
                next_token = (next_token > 0.5).float()

                # Append to sequence
                tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

    def generate_with_caching(self, initial_tokens, generate_steps=10, device="cpu"):
        """
        Generate tokens using caching for efficiency.

        Args:
            initial_tokens: (B, T, token_dim) starting tokens
            generate_steps: number of tokens to generate
            device: device to generate on

        Returns:
            extended_tokens: (B, T + generate_steps, token_dim)
        """
        current_tokens = initial_tokens.to(device)
        batch_size, seq_len, token_dim = current_tokens.shape

        self.predictor.clear_cache()

        with torch.no_grad():
            for step in range(generate_steps):
                # Predict next token
                pred = self.predictor(current_tokens)  # (B, seq_len, token_dim)

                # Take last prediction and binarize
                next_token = (pred[:, -1:, :] > 0.5).float()

                # Append
                current_tokens = torch.cat([current_tokens, next_token], dim=1)

        self.predictor.clear_cache()
        return current_tokens


class AutoregressiveLoss(nn.Module):
    """Loss for training autoregressive token predictor."""

    def __init__(self, token_pred_weight=1.0):
        super().__init__()
        self.token_pred_weight = token_pred_weight

    def forward(self, pred_tokens, target_tokens):
        """
        Args:
            pred_tokens: predicted tokens (B, T, token_dim)
            target_tokens: ground truth tokens (B, T, token_dim)

        Returns:
            loss: scalar loss
        """
        # Token prediction loss: binary cross-entropy
        pred_binary = torch.sigmoid(pred_tokens)
        target_binary = (target_tokens > 0.5).float()
        pred_loss = F.binary_cross_entropy(pred_binary, target_binary)

        total_loss = self.token_pred_weight * pred_loss
        return total_loss
