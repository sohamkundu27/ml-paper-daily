import torch
import torch.nn as nn
import math
import numpy as np


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
    def __init__(self, audio_dim=128, time_dim=64, hidden_dim=256, cond_dim=None):
        super().__init__()
        self.audio_dim = audio_dim
        self.time_dim = time_dim
        self.cond_dim = cond_dim

        self.time_embed = SinusoidalPosEmbed(time_dim)

        # Input size: audio + time + optional conditioning
        input_size = audio_dim + time_dim
        if cond_dim is not None:
            input_size += cond_dim

        # Simple MLP-based denoising network
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, audio_dim),
        )

    def forward(self, x, t, cond=None):
        # x: (batch_size, audio_dim)
        # t: (batch_size,) or scalar, timestep in [0, 1]
        # cond: (batch_size, cond_dim) optional conditioning vector
        t_emb = self.time_embed(t)
        x_concat = torch.cat([x, t_emb], dim=-1)

        if cond is not None:
            x_concat = torch.cat([x_concat, cond], dim=-1)

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

    def sample(self, model, shape, device, num_steps=None, cond=None):
        # Reverse diffusion: denoise from noise to sample
        # cond: optional conditioning vector of shape (batch_size, cond_dim)
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

                # Predict noise (with optional conditioning)
                noise_pred = model(x, t_norm, cond=cond)

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


class StageAdaptiveScheduler:
    """Adapts loss weights across training stages.

    Early training (semantic stage): focus on coarse structure via high-noise timesteps.
    Late training (perceptual stage): focus on fine details via low-noise timesteps.
    """
    def __init__(self, num_training_steps, timesteps=100, strategy='linear'):
        self.num_training_steps = num_training_steps
        self.timesteps = timesteps
        self.strategy = strategy

    def get_weights(self, current_step):
        """Return (semantic_weight, perceptual_weight) for current step.

        Args:
            current_step: current training step in [0, num_training_steps)

        Returns:
            (semantic_weight, perceptual_weight): weights that sum to 1.0
        """
        progress = current_step / max(self.num_training_steps - 1, 1)  # [0, 1]

        if self.strategy == 'linear':
            semantic_weight = 1.0 - progress
            perceptual_weight = progress
        elif self.strategy == 'exponential':
            semantic_weight = np.exp(-3.0 * progress)
            semantic_weight = semantic_weight / (semantic_weight + 1.0)
            perceptual_weight = 1.0 - semantic_weight
        elif self.strategy == 'cosine':
            semantic_weight = 0.5 * (1.0 + np.cos(np.pi * progress))
            perceptual_weight = 0.5 * (1.0 - np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return semantic_weight, perceptual_weight

    def get_timestep_mask(self, batch_size, sampled_t, device, current_step):
        """Create masks for semantic vs perceptual timesteps.

        Semantic: focus on high-noise timesteps (t > mid)
        Perceptual: focus on low-noise timesteps (t < mid)

        Args:
            batch_size: batch size
            sampled_t: sampled timesteps, shape (batch_size,)
            device: torch device
            current_step: current training step

        Returns:
            (semantic_mask, perceptual_mask): boolean masks for each objective
        """
        semantic_weight, perceptual_weight = self.get_weights(current_step)

        # Midpoint of timestep range
        mid_t = self.timesteps // 2

        # Semantic loss emphasizes high-noise (large t)
        semantic_mask = sampled_t >= mid_t

        # Perceptual loss emphasizes low-noise (small t)
        perceptual_mask = sampled_t < mid_t

        return semantic_mask, perceptual_mask, semantic_weight, perceptual_weight


def stage_adaptive_loss(noise_pred, target_noise, t, scheduler, current_step):
    """Compute stage-adaptive loss with separate semantic and perceptual components.

    Args:
        noise_pred: predicted noise from model
        target_noise: ground truth noise
        t: timesteps for each sample in batch
        scheduler: StageAdaptiveScheduler instance
        current_step: current training step

    Returns:
        loss: weighted sum of semantic and perceptual losses
    """
    device = noise_pred.device
    batch_size = noise_pred.shape[0]

    semantic_mask, perceptual_mask, semantic_w, perceptual_w = scheduler.get_timestep_mask(
        batch_size, t, device, current_step
    )

    # Base MSE loss per sample
    mse_loss = torch.mean((noise_pred - target_noise) ** 2, dim=list(range(1, noise_pred.ndim)))

    # Semantic loss: high-noise timesteps
    semantic_loss = 0.0
    if semantic_mask.any():
        semantic_loss = mse_loss[semantic_mask].mean()

    # Perceptual loss: low-noise timesteps
    perceptual_loss = 0.0
    if perceptual_mask.any():
        perceptual_loss = mse_loss[perceptual_mask].mean()

    # Combine with stage-adaptive weights
    total_loss = semantic_w * semantic_loss + perceptual_w * perceptual_loss

    return total_loss


def make_class_embedding(class_id, num_classes, cond_dim=32, device='cpu'):
    """Create a learnable class embedding for conditioning.

    Args:
        class_id: integer class index or tensor of shape (batch_size,)
        num_classes: total number of classes
        cond_dim: embedding dimension
        device: torch device

    Returns:
        embedding: one-hot encoding expanded to cond_dim via random projection
    """
    if isinstance(class_id, int):
        class_id = torch.tensor([class_id], device=device)
    elif isinstance(class_id, torch.Tensor):
        if class_id.device != device:
            class_id = class_id.to(device)
    else:
        class_id = torch.tensor(class_id, device=device)

    batch_size = class_id.shape[0]

    # One-hot encoding
    one_hot = torch.zeros(batch_size, num_classes, device=device)
    one_hot.scatter_(1, class_id.view(-1, 1), 1.0)

    # Expand one-hot to cond_dim with fixed random projection
    # (ensures consistency across runs)
    torch.manual_seed(0)
    proj = torch.randn(num_classes, cond_dim, device=device) / math.sqrt(num_classes)

    embedding = torch.mm(one_hot, proj)  # (batch_size, cond_dim)
    return embedding
