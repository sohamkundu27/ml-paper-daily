import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SyntheticDataGenerator:
    """Generate simple synthetic sequence data for training."""

    @staticmethod
    def random_sequences(num_samples, seq_len, token_dim, seed=None):
        """Generate random token sequences.

        Args:
            num_samples: number of sequences
            seq_len: length of each sequence
            token_dim: dimension of each token
            seed: random seed

        Returns:
            [num_samples, seq_len, token_dim] tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(num_samples, seq_len, token_dim)

    @staticmethod
    def repeating_pattern_sequences(num_samples, seq_len, token_dim, pattern_len=4):
        """Generate sequences with repeating patterns.

        Args:
            num_samples: number of sequences
            seq_len: length of each sequence
            token_dim: dimension of each token
            pattern_len: length of repeating pattern

        Returns:
            [num_samples, seq_len, token_dim] tensor
        """
        sequences = []
        for _ in range(num_samples):
            # Create a random pattern
            pattern = torch.randn(pattern_len, token_dim)
            # Repeat it to fill seq_len
            seq = pattern.repeat(seq_len // pattern_len + 1, 1)[:seq_len]
            sequences.append(seq)
        return torch.stack(sequences)

    @staticmethod
    def sine_wave_sequences(num_samples, seq_len, token_dim, freq=1.0):
        """Generate sequences based on sine waves.

        Args:
            num_samples: number of sequences
            seq_len: length of each sequence
            token_dim: dimension of each token
            freq: frequency of sine wave

        Returns:
            [num_samples, seq_len, token_dim] tensor
        """
        sequences = []
        for i in range(num_samples):
            # Each token dimension gets a sine wave with different phase
            t = np.linspace(0, 2 * np.pi * freq, seq_len)
            seq = torch.zeros(seq_len, token_dim)
            for d in range(token_dim):
                phase = 2 * np.pi * d / token_dim
                seq[:, d] = torch.from_numpy(np.sin(t + phase)).float()
            sequences.append(seq)
        return torch.stack(sequences)


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

    def compute_loss(self, x_0, t):
        """Compute L2 loss between predicted and actual clean tokens.

        Args:
            x_0: [batch, seq_len, token_dim] clean tokens
            t: [batch] timestep in [0, 1]

        Returns:
            loss: scalar L2 loss
        """
        # Add noise at timestep t
        x_t, _, _ = self.forward_diffusion(x_0, t)

        # Predict clean tokens
        x_pred = self.denoise(x_t, t)

        # L2 loss
        loss = torch.nn.functional.mse_loss(x_pred, x_0)
        return loss

    def train_step(self, x_0, t, optimizer):
        """Single training step.

        Args:
            x_0: [batch, seq_len, token_dim] clean tokens
            t: [batch] timestep in [0, 1]
            optimizer: torch optimizer

        Returns:
            loss: scalar loss value
        """
        optimizer.zero_grad()
        loss = self.compute_loss(x_0, t)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, data_loader, num_epochs, learning_rate=1e-3):
        """Train the denoiser on a dataset.

        Args:
            data_loader: iterable of batches of [batch, seq_len, token_dim] tensors
            num_epochs: number of training epochs
            learning_rate: optimizer learning rate

        Returns:
            losses: list of average losses per epoch
        """
        optimizer = optim.Adam(self.denoiser.parameters(), lr=learning_rate)
        losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for x_0 in data_loader:
                # Move to device
                x_0 = x_0.to(self.device)

                # Random timestep for each sample in batch
                t = torch.rand(x_0.shape[0]).to(self.device)

                # Training step
                loss = self.train_step(x_0, t, optimizer)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

        return losses
