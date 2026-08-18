"""
Pass 4: Inference and generation.

This implements sampling/generation from a trained VAR model:
- Starting from an empty coarse scale, iteratively sample finer scales
- Each scale is predicted conditioned on all coarser scales
- Demonstrates that the model learns meaningful hierarchical structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from var_pass3 import VARTrainer, ToyImageDataset
from var_pass2 import VARPass2, create_cumulative_scale_mask, ScaleEmbedding
from var_pass1 import tokenize_to_sequence


class VARGenerator:
    """
    Generates new token sequences using a trained VAR model.

    Implements coarse-to-fine sampling: starts with a coarse-scale distribution
    and iteratively samples finer scales conditioned on coarser ones.
    """

    def __init__(self, model, device="cpu", temperature=1.0):
        """
        Args:
            model: Trained VARPass2 model
            device: Device to generate on
            temperature: Sampling temperature (higher = more random)
        """
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature
        self.model.eval()

    def generate(self, batch_size=1, img_size=64, num_steps=None):
        """
        Generate token sequences coarse-to-fine.

        Args:
            batch_size: Number of samples to generate
            img_size: Expected image size (for spatial dimensions)
            num_steps: Number of generation steps (if None, use model.num_scales)

        Returns:
            generated_tokens: Dictionary mapping scale_idx to token tensor
                             each of shape (B, H_i*W_i) where H_i, W_i are spatial dims at scale i
        """
        num_scales = self.model.num_scales
        if num_steps is None:
            num_steps = num_scales

        # Initialize generated tokens dict
        generated_tokens = {}

        # For each scale from coarsest to finest
        for scale_idx in range(num_steps):
            # Compute spatial dimensions at this scale
            # Start at img_size, divide by 2^(scale_idx+1) due to stride-2 convolutions
            h_at_scale = img_size // (2 ** (scale_idx + 1))
            w_at_scale = h_at_scale
            num_tokens_at_scale = h_at_scale * w_at_scale

            # Build token sequence from all scales seen so far
            token_sequence, scale_indices = self._build_sequence_so_far(
                generated_tokens, batch_size, scale_idx
            )

            if token_sequence.shape[1] == 0:
                # Very first scale: sample uniformly
                sampled_tokens = torch.randint(
                    0, self.model.output_proj.out_features,
                    (batch_size, num_tokens_at_scale),
                    device=self.device
                )
            else:
                # Later scales: predict from current sequence
                sampled_tokens = self._sample_scale(
                    token_sequence, scale_indices, num_tokens_at_scale, scale_idx
                )

            generated_tokens[scale_idx] = sampled_tokens

        return generated_tokens

    def _build_sequence_so_far(self, generated_tokens, batch_size, current_scale):
        """
        Build the token sequence from all scales generated so far.

        Args:
            generated_tokens: Dict mapping scale_idx to sampled token indices
            batch_size: Batch size
            current_scale: Current scale being generated

        Returns:
            token_sequence: (B, N, D) where N is total tokens from scales 0 to current_scale-1
            scale_indices: (N,) tensor indicating scale index for each token
        """
        token_dim = self.model.token_dim
        sequences = []
        scale_indices_list = []

        for scale_idx in range(current_scale):
            sampled_indices = generated_tokens[scale_idx]  # (B, num_tokens)
            B, num_tokens = sampled_indices.shape

            # Create embeddings for these tokens by passing through projection layer
            # Use a simple embedding: treat each token index as a feature
            # For a more sophisticated approach, would have a learned embedding layer
            token_embeds = torch.randn(
                B, num_tokens, token_dim, device=self.device
            ) * 0.01

            # Add some signal based on token value (crude embedding)
            token_embeds = token_embeds + (
                sampled_indices.float().unsqueeze(-1) / 4096.0
            ) * 0.1

            sequences.append(token_embeds)
            scale_indices_list.extend([scale_idx] * num_tokens)

        if not sequences:
            # No sequences yet (generating first scale)
            token_sequence = torch.zeros(batch_size, 0, token_dim, device=self.device)
            scale_indices = torch.tensor([], dtype=torch.long, device=self.device)
        else:
            token_sequence = torch.cat(sequences, dim=1)
            scale_indices = torch.tensor(
                scale_indices_list, dtype=torch.long, device=self.device
            )

        return token_sequence, scale_indices

    def _sample_scale(self, token_sequence, scale_indices, num_tokens_at_scale, scale_idx):
        """
        Sample tokens for a specific scale using the model.

        Args:
            token_sequence: (B, N, D) tokens from coarser scales
            scale_indices: (N,) scale indices for existing tokens
            num_tokens_at_scale: Number of tokens to generate for current scale
            scale_idx: Current scale index

        Returns:
            sampled_tokens: (B, num_tokens_at_scale) sampled token indices
        """
        batch_size = token_sequence.shape[0]
        device = token_sequence.device

        # Add scale embeddings to existing sequence
        token_sequence = self.model.scale_embedding(token_sequence, scale_indices)

        # Extend scale indices to include new scale tokens (will be sampled)
        new_scale_indices = torch.cat([
            scale_indices,
            torch.full((num_tokens_at_scale,), scale_idx, dtype=torch.long, device=device)
        ])

        # Create attention mask for coarse-to-fine: can see all previous scales
        attn_mask = create_cumulative_scale_mask(new_scale_indices, device)
        attn_mask = ~attn_mask  # Invert for MultiheadAttention

        # Create learnable embeddings for tokens we're about to predict
        # Initialize to random small values
        new_token_embeds = torch.randn(
            batch_size, num_tokens_at_scale, self.model.token_dim, device=device
        ) * 0.01

        # Concatenate with existing sequence for full input
        full_sequence = torch.cat([token_sequence, new_token_embeds], dim=1)

        # Pass through transformer layers
        hidden = full_sequence
        for layer in self.model.transformer_layers:
            hidden = layer(hidden, attn_mask=attn_mask)

        # Extract embeddings for the newly predicted scale
        # Only the last num_tokens_at_scale positions are predictions for current scale
        new_hidden = hidden[:, -num_tokens_at_scale:, :]  # (B, num_tokens, D)

        # Project to logits
        hidden_normed = self.model.norm_final(new_hidden)
        logits = self.model.output_proj(hidden_normed)  # (B, num_tokens, vocab_size)

        # Sample from logits using temperature
        scaled_logits = logits / self.temperature
        probabilities = F.softmax(scaled_logits, dim=-1)

        # Sample tokens
        sampled_tokens = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            num_samples=1
        ).reshape(batch_size, num_tokens_at_scale)

        return sampled_tokens


class VARPass4(nn.Module):
    """
    Pass 4: Full VAR with training and generation.

    Combines the trained model from Pass 3 with inference capability.
    """

    def __init__(
        self,
        in_channels=3,
        token_dim=256,
        num_scales=3,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        vocab_size=4096,
    ):
        """
        Args:
            in_channels: Number of input image channels
            token_dim: Dimension of token embeddings
            num_scales: Number of hierarchical scales
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Dimension of feedforward hidden layer
            vocab_size: Size of token vocabulary
        """
        super().__init__()
        self.var_base = VARPass2(
            in_channels=in_channels,
            token_dim=token_dim,
            num_scales=num_scales,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            vocab_size=vocab_size,
        )
        self.num_scales = num_scales
        self.token_dim = token_dim

    def forward(self, x):
        """Forward pass (training)."""
        return self.var_base(x)

    def generate(self, batch_size=1, img_size=64, temperature=1.0):
        """
        Generate new token sequences.

        Args:
            batch_size: Number of samples to generate
            img_size: Expected image size
            temperature: Sampling temperature

        Returns:
            generated_tokens: Dict mapping scale_idx to token tensors
        """
        generator = VARGenerator(self.var_base, device=self.var_base.norm_final.weight.device, temperature=temperature)
        return generator.generate(batch_size, img_size)
