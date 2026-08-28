import torch
import torch.nn as nn
import math


class DiffusionScheduler:
    """Linear noise schedule for diffusion process."""

    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # Precalculate useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0, t, noise=None):
        """Add noise to clean sample at timestep t (forward diffusion)."""
        if noise is None:
            noise = torch.randn_like(x0)

        # Handle both 1D and scalar t
        if t.dim() == 0:
            t = t.unsqueeze(0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting with x0
        if x0.dim() > 1:
            # For multi-dimensional tensors, reshape to (batch, 1, 1, 1)
            sqrt_alpha = sqrt_alpha.view(-1, *([1] * (x0.dim() - 1)))
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, *([1] * (x0.dim() - 1)))

        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return xt, noise

    def get_alpha_prod(self, t):
        """Get alpha_t^cumprod for given timestep."""
        return self.alphas_cumprod[t]


class DiffusionLoss(nn.Module):
    """MSE loss for denoising prediction."""

    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        self.mse_loss = nn.MSELoss()

    def forward(self, model_output, noise):
        """Compute MSE between predicted and actual noise."""
        return self.mse_loss(model_output, noise)
