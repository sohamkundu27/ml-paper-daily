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
