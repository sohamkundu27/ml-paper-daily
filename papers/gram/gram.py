import torch
import torch.nn as nn
import torch.optim as optim


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() if activation else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GRAMPass1(nn.Module):
    """
    Pass 1: Basic recursive latent reasoning framework.

    Implements a simple recursive reasoning loop in latent space:
    - Encode input to latent representation
    - Apply transition function N times deterministically
    - Decode latent to output
    - Optimize via reconstruction loss
    """

    def __init__(self, input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_steps = num_steps

        # Encoder: input -> latent
        self.encoder = MLPBlock(input_dim, hidden_dim, latent_dim, activation=True)

        # Transition: latent -> latent (fixed linear for Pass 1)
        self.transition = nn.Linear(latent_dim, latent_dim)

        # Decoder: latent -> output
        self.decoder = MLPBlock(latent_dim, hidden_dim, input_dim, activation=True)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, input_dim)
        Returns:
            output: reconstructed tensor of shape (batch_size, input_dim)
            latent_trajectory: list of latent states at each step
        """
        # Encode to latent space
        latent = self.encoder(x)
        latent_trajectory = [latent.clone().detach()]

        # Recursive reasoning: apply transition N times
        for _ in range(self.num_steps):
            latent = torch.relu(self.transition(latent))
            latent_trajectory.append(latent.clone().detach())

        # Decode back to output space
        output = self.decoder(latent)

        return output, latent_trajectory

    def compute_loss(self, output, target):
        """Reconstruction loss (MSE)"""
        return torch.mean((output - target) ** 2)


class GRAMPass2(nn.Module):
    """
    Pass 2: Stochastic trajectory generation with variational inference.

    Extends Pass 1 with:
    - Stochastic transitions via learned Gaussian perturbations
    - Amortized variational inference (transition network outputs mean and log_var)
    - KL divergence regularization on latent transitions
    - Multiple trajectory sampling at inference time
    """

    def __init__(self, input_dim=5, latent_dim=16, hidden_dim=32, num_steps=3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_steps = num_steps

        # Encoder: input -> latent
        self.encoder = MLPBlock(input_dim, hidden_dim, latent_dim, activation=True)

        # Stochastic transition: latent -> (mean, log_var)
        # Amortized variational inference networks
        self.transition_mean = MLPBlock(
            latent_dim, hidden_dim, latent_dim, activation=True
        )
        self.transition_log_var = MLPBlock(
            latent_dim, hidden_dim, latent_dim, activation=True
        )

        # Decoder: latent -> output
        self.decoder = MLPBlock(latent_dim, hidden_dim, input_dim, activation=True)

    def forward(self, x, sample=True):
        """
        Args:
            x: input tensor of shape (batch_size, input_dim)
            sample: if True, sample from stochastic transition; if False, use mean
        Returns:
            output: reconstructed tensor of shape (batch_size, input_dim)
            latent_trajectory: list of latent states at each step
            kl_loss: KL divergence loss (accumulated over all steps)
        """
        # Encode to latent space
        latent = self.encoder(x)
        latent_trajectory = [latent.clone().detach()]
        total_kl = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Recursive reasoning: apply stochastic transition N times
        for _ in range(self.num_steps):
            # Get parameters of transition distribution
            mean = self.transition_mean(latent)
            log_var = self.transition_log_var(latent)
            var = torch.exp(log_var)

            # Sample or use mean
            if sample:
                eps = torch.randn_like(latent)
                latent = mean + torch.sqrt(var) * eps
            else:
                latent = mean

            latent_trajectory.append(latent.clone().detach())

            # KL divergence: KL(N(mean, var) || N(0, 1))
            # = -0.5 * sum(1 + log_var - mean^2 - var)
            kl = -0.5 * torch.mean(1 + log_var - mean**2 - var)
            total_kl = total_kl + kl

        # Decode back to output space
        output = self.decoder(latent)

        return output, latent_trajectory, total_kl

    def compute_loss(self, output, target, kl_loss, kl_weight=0.01):
        """
        Reconstruction loss + weighted KL regularization

        Args:
            output: model output
            target: target tensor
            kl_loss: KL divergence loss from forward pass
            kl_weight: weight for KL term (default 0.01 for balanced loss)
        """
        recon_loss = torch.mean((output - target) ** 2)
        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss
