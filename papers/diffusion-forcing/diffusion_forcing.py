import numpy as np
import torch
import torch.nn as nn


def cosine_schedule(t, s=0.008):
    """Cosine annealing schedule for noise levels.

    Args:
        t: timestep in [0, 1]
        s: small constant to avoid 0 at start

    Returns:
        alpha (signal retention) in [0, 1]
    """
    return np.cos((t + s) / (1 + s) * np.pi / 2) ** 2


def get_alpha_beta(t, schedule_fn=cosine_schedule):
    """Get alpha and beta for timestep t.

    alpha = signal retention, beta = noise variance
    """
    alpha = schedule_fn(t)
    beta = 1.0 - alpha
    return alpha, beta


class Denoiser(nn.Module):
    """Simple MLP denoiser for per-token denoising."""

    def __init__(self, token_dim, hidden_dim=128, time_dim=64):
        super().__init__()
        self.token_dim = token_dim
        self.time_dim = time_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Main network: concatenate noisy token + time embedding
        self.net = nn.Sequential(
            nn.Linear(token_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, x_t, t):
        """Denoise x_t at timestep t.

        Args:
            x_t: [batch, seq_len, token_dim] noisy tokens
            t: [batch] timestep in [0, 1] or [batch, 1]

        Returns:
            [batch, seq_len, token_dim] predicted clean tokens
        """
        batch_size, seq_len, token_dim = x_t.shape

        # Ensure t is proper shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [batch, 1]

        # Time embedding: [batch, time_dim]
        t_emb = self.time_embed(t)  # [batch, 1] -> [batch, time_dim]

        # Reshape for broadcasting: [batch, 1, time_dim]
        t_emb = t_emb.unsqueeze(1)

        # Concatenate: [batch, seq_len, token_dim + time_dim]
        x_and_t = torch.cat([x_t, t_emb.expand(-1, seq_len, -1)], dim=-1)

        # Reshape to [batch * seq_len, token_dim + time_dim]
        x_and_t_flat = x_and_t.view(-1, token_dim + self.time_dim)

        # Denoise: [batch * seq_len, token_dim]
        pred = self.net(x_and_t_flat)

        # Reshape back: [batch, seq_len, token_dim]
        pred = pred.view(batch_size, seq_len, token_dim)

        return pred


class DiffusionForcing:
    """Foundational Diffusion Forcing mechanics."""

    def __init__(self, token_dim, schedule_fn=cosine_schedule, device='cpu'):
        self.token_dim = token_dim
        self.schedule_fn = schedule_fn
        self.device = device
        self.denoiser = Denoiser(token_dim).to(device)

    def forward_diffusion(self, x_0, t):
        """Add noise to clean tokens x_0 at timestep t.

        Args:
            x_0: [batch, seq_len, token_dim] clean tokens
            t: [batch] timestep in [0, 1]

        Returns:
            x_t: [batch, seq_len, token_dim] noisy tokens
            alpha: noise retention factor
            beta: noise variance
        """
        alpha, beta = get_alpha_beta(t.cpu().numpy() if isinstance(t, torch.Tensor) else t)

        # Ensure alpha and beta are tensors
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, dtype=torch.float32, device=self.device)

        # Reshape for broadcasting: [batch, 1, 1]
        alpha = alpha.view(-1, 1, 1)
        beta = beta.view(-1, 1, 1)

        # x_t = sqrt(alpha) * x_0 + sqrt(beta) * epsilon
        epsilon = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha) * x_0 + torch.sqrt(beta) * epsilon

        return x_t, alpha.squeeze(), beta.squeeze()

    def denoise(self, x_t, t):
        """Use denoiser network to predict clean tokens.

        Args:
            x_t: [batch, seq_len, token_dim] noisy tokens
            t: [batch] timestep in [0, 1]

        Returns:
            x_pred: [batch, seq_len, token_dim] predicted clean tokens
        """
        return self.denoiser(x_t, t)
