"""Binary diffusion process for token generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Encode timestep t (scalar or batch) to sinusoidal embedding.

        Args:
            t: timestep, shape (B,) or scalar

        Returns:
            embedding: sinusoidal encoding, shape (B, dim) or (1, dim)
        """
        if isinstance(t, (int, float)):
            t = torch.tensor([t], dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        half_dim = self.dim // 2

        # Frequency schedule
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        ).to(device)

        # Compute angles
        angles = t.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, 1))

        return emb


class BinaryDiffusionHead(nn.Module):
    """
    Binary diffusion head for refining noisy token predictions.

    Takes noisy binary tokens and a timestep, outputs refined predictions
    using a continuous-space diffusion process.
    """

    def __init__(self, token_dim=32, hidden_dim=128, num_timesteps=1000):
        """
        Args:
            token_dim: dimensionality of token embeddings
            hidden_dim: hidden dimension for denoising network
            num_timesteps: number of diffusion timesteps
        """
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps

        # Timestep embedding
        self.time_embed = SinusoidalPositionalEncoding(hidden_dim)

        # Denoising network: takes noisy tokens + timestep -> refined tokens
        # Simple 2-layer MLP per spatial location (applied convolutionally)
        self.denoiser = nn.Sequential(
            nn.Conv2d(token_dim + hidden_dim, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, token_dim, kernel_size=1, padding=0),
        )

    def add_noise(self, tokens, t, noise_scale=1.0):
        """
        Add Gaussian noise to tokens, scaled by diffusion schedule.

        Args:
            tokens: binary tokens (B, token_dim, H, W) in {0, 1}
            t: timestep (B,) in [0, num_timesteps-1]
            noise_scale: overall noise amplitude

        Returns:
            noisy_tokens: tokens with added noise
        """
        # Compute noise schedule: variance increases with t
        # Use simple linear schedule: beta_t = min_beta + (max_beta - min_beta) * t / T
        min_beta = 0.0001
        max_beta = 0.02
        t_norm = t.float() / self.num_timesteps
        beta_t = min_beta + (max_beta - min_beta) * t_norm

        # Reshape for broadcasting: (B, 1, 1, 1)
        beta_t = beta_t.view(-1, 1, 1, 1)

        # Add Gaussian noise
        noise = torch.randn_like(tokens) * noise_scale
        noisy = tokens + torch.sqrt(beta_t) * noise

        return noisy

    def forward(self, tokens, t):
        """
        Denoise binary tokens at timestep t.

        Args:
            tokens: noisy token embedding (B, token_dim, H, W)
            t: timestep (B,)

        Returns:
            pred_tokens: denoised token predictions (B, token_dim, H, W)
        """
        # Get timestep embedding
        time_emb = self.time_embed(t)  # (B, hidden_dim)

        # Reshape for broadcasting to spatial dims: (B, hidden_dim, 1, 1)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)

        # Expand to match spatial resolution
        batch_size, _, h, w = tokens.shape
        time_emb = time_emb.expand(batch_size, -1, h, w)

        # Concatenate tokens with time embedding
        x = torch.cat([tokens, time_emb], dim=1)

        # Denoise
        pred_tokens = self.denoiser(x)

        return pred_tokens

    def diffusion_loss(self, clean_tokens, t, noise_scale=1.0):
        """
        Compute diffusion loss: MSE between predicted and clean tokens.

        Args:
            clean_tokens: ground truth binary tokens (B, token_dim, H, W)
            t: timestep (B,)
            noise_scale: noise amplitude

        Returns:
            loss: MSE loss
        """
        # Add noise to clean tokens
        noisy_tokens = self.add_noise(clean_tokens, t, noise_scale=noise_scale)

        # Predict clean tokens
        pred_clean = self.forward(noisy_tokens, t)

        # Compute MSE loss
        loss = F.mse_loss(pred_clean, clean_tokens)

        return loss


class BinaryDiffusionSampler:
    """
    Sampler for generating tokens via the reverse diffusion process.
    """

    def __init__(self, model, num_timesteps=1000, noise_scale=1.0):
        """
        Args:
            model: BinaryDiffusionHead model
            num_timesteps: number of diffusion steps
            noise_scale: noise amplitude
        """
        self.model = model
        self.num_timesteps = num_timesteps
        self.noise_scale = noise_scale

    def sample(self, shape, num_steps=50, device="cpu"):
        """
        Generate tokens via reverse diffusion.

        Args:
            shape: (B, C, H, W) shape of tokens to generate
            num_steps: number of reverse diffusion steps to use
            device: device to generate on

        Returns:
            tokens: generated tokens (B, C, H, W)
        """
        # Start with random noise
        tokens = torch.randn(shape, device=device)

        # Reverse diffusion: go from t=T-1 down to t=0
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device
        )

        with torch.no_grad():
            for step, t_step in enumerate(timesteps):
                # Prepare timestep batch
                t_batch = torch.full((shape[0],), t_step, dtype=torch.long, device=device)

                # Predict clean tokens
                pred_clean = self.model(tokens, t_batch)

                # Simple refinement: move toward predicted clean
                # This is a simplified reverse step (not the full DDPM schedule)
                alpha = 0.1
                tokens = (1 - alpha) * tokens + alpha * pred_clean

                # Optional: clip to reasonable range
                tokens = torch.clamp(tokens, -1, 1)

        # Quantize to binary at the end
        binary_tokens = (tokens > 0).float()

        return binary_tokens
