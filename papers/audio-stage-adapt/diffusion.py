import torch
import torch.nn as nn
import math


class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: (batch_size,) or scalar
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class AudioDiffusionModel(nn.Module):
    def __init__(self, audio_dim=128, time_dim=64, hidden_dim=256):
        super().__init__()
        self.audio_dim = audio_dim
        self.time_dim = time_dim

        self.time_embed = SinusoidalPosEmbed(time_dim)

        # Simple MLP-based denoising network
        self.net = nn.Sequential(
            nn.Linear(audio_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, audio_dim),
        )

    def forward(self, x, t):
        # x: (batch_size, audio_dim)
        # t: (batch_size,) or scalar, timestep in [0, 1]
        t_emb = self.time_embed(t)
        x_concat = torch.cat([x, t_emb], dim=-1)
        return self.net(x_concat)


class GaussianDiffusion:
    def __init__(self, timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precompute useful coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For reverse process
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

    def q_sample(self, x0, t, noise=None):
        # Forward diffusion: add noise to x0
        # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
        if noise is None:
            noise = torch.randn_like(x0)

        batch_size = x0.shape[0]
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        if sqrt_alpha.dim() == 0:
            sqrt_alpha = sqrt_alpha.unsqueeze(0)
        if sqrt_one_minus_alpha.dim() == 0:
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(0)

        sqrt_alpha = sqrt_alpha.unsqueeze(-1) if sqrt_alpha.shape[0] == batch_size else sqrt_alpha
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1) if sqrt_one_minus_alpha.shape[0] == batch_size else sqrt_one_minus_alpha

        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def sample(self, model, shape, device, num_steps=None):
        # Reverse diffusion: denoise from noise to sample
        if num_steps is None:
            num_steps = self.timesteps

        x = torch.randn(shape, device=device)

        # Use a subset of timesteps if num_steps < timesteps
        step_size = self.timesteps // num_steps
        timesteps = list(range(self.timesteps - 1, 0, -step_size))

        model.eval()
        with torch.no_grad():
            for t_idx in timesteps:
                t = torch.full((shape[0],), t_idx, dtype=torch.long, device=device)
                t_norm = t.float() / self.timesteps  # Normalize to [0, 1]

                # Predict noise
                noise_pred = model(x, t_norm)

                # Reverse step (simplified)
                alpha = self.alphas[t_idx]
                alpha_bar = self.alphas_cumprod[t_idx]

                # Simple reverse step without variance scheduling
                x = (x - (1 - alpha) / self.sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred) / torch.sqrt(alpha)

                # Add noise back in if not at the end
                if t_idx > 1:
                    z = torch.randn_like(x)
                    x = x + torch.sqrt(self.betas[t_idx]) * z

        return x
