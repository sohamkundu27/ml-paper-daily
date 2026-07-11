"""Linear SSM with learned diffusion-based transitions via score matching."""

import numpy as np
from typing import Tuple, List, Callable
import torch
import torch.nn as nn
import torch.optim as optim


class LinearSSM:
    """Linear Gaussian state space model: z_{t+1} = A @ z_t + w, y_t = C @ z_t + v."""

    def __init__(self, state_dim: int, obs_dim: int, seed: int = 0):
        """
        Args:
            state_dim: Dimension of latent state z.
            obs_dim: Dimension of observations y.
            seed: Random seed.
        """
        np.random.seed(seed)
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # Transition matrix (learnable)
        self.A = np.eye(state_dim) + 0.1 * np.random.randn(state_dim, state_dim)
        # Observation matrix (learnable)
        self.C = np.random.randn(obs_dim, state_dim)
        # Process noise covariance
        self.Q = 0.1 * np.eye(state_dim)
        # Observation noise covariance
        self.R = 0.1 * np.eye(obs_dim)

    def sample_trajectory(self, T: int, z0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a trajectory of length T.

        Args:
            T: Sequence length.
            z0: Initial state. If None, sample from standard normal.

        Returns:
            (z, y) where z is [T, state_dim] and y is [T, obs_dim].
        """
        if z0 is None:
            z0 = np.random.randn(self.state_dim)

        z_seq = []
        y_seq = []
        z = z0

        for _ in range(T):
            z_seq.append(z)
            # Observation with noise
            y = self.C @ z + np.random.randn(self.obs_dim) * np.sqrt(np.diag(self.R))
            y_seq.append(y)
            # Transition with process noise
            z = self.A @ z + np.random.randn(self.state_dim) * np.sqrt(np.diag(self.Q))

        return np.array(z_seq), np.array(y_seq)


class DiffusionProcess:
    """Minimal diffusion process for learning transitions between latent states."""

    def __init__(self, state_dim: int, num_steps: int = 100, beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        Args:
            state_dim: Dimension of states to diffuse.
            num_steps: Number of diffusion steps.
            beta_start: Initial noise schedule value.
            beta_end: Final noise schedule value.
        """
        self.state_dim = state_dim
        self.num_steps = num_steps

        # Linear noise schedule
        self.betas = np.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])

        # Precompute variance schedule
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def forward_diffusion(self, x0: np.ndarray, t: int, noise: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.

        Args:
            x0: Clean sample [state_dim].
            t: Diffusion step (0 to num_steps-1).
            noise: Optional noise to use; if None, sample standard normal.

        Returns:
            (x_t, noise_used) where x_t is diffused state and noise_used is the noise applied.
        """
        if noise is None:
            noise = np.random.randn(self.state_dim)

        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]

        x_t = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise

    def reverse_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        """
        Simplified reverse step: predict one step back given predicted noise.

        Args:
            x_t: Current noisy state.
            t: Current diffusion step.
            predicted_noise: Model's prediction of the noise.

        Returns:
            x_{t-1}: One step back in reverse.
        """
        if t == 0:
            return x_t

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t]

        # Simplified reverse (not exact Gaussian; for demo only)
        sqrt_alpha_cumprod = np.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - alpha_cumprod_t)

        # Estimate x_0 from x_t and predicted noise
        x_0_est = (x_t - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod

        # Reweight: simplified posterior mean (ignoring variance for now)
        posterior_coef = np.sqrt((1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)) * beta_t
        x_t_minus_1 = (x_0_est * np.sqrt(alpha_cumprod_prev_t) +
                        x_t * (1.0 - np.sqrt(alpha_cumprod_prev_t)))

        return x_t_minus_1


class NoisePredictor(nn.Module):
    """Neural network that predicts noise for diffusion-based state transitions."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        """
        Args:
            state_dim: Dimension of states to denoise.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise in x_t at timestep t.

        Args:
            x_t: Noisy states [batch, state_dim].
            t: Timestep indices [batch] in [0, num_steps).

        Returns:
            Predicted noise [batch, state_dim].
        """
        # Normalize timestep to [0, 1]
        t_norm = t.float().unsqueeze(1) / 100.0
        x_t_input = torch.cat([x_t, t_norm], dim=1)
        return self.net(x_t_input)


def train_diffusion_ssm(
    ssm: LinearSSM,
    diffusion: DiffusionProcess,
    noise_predictor: NoisePredictor,
    num_trajectories: int = 100,
    trajectory_length: int = 20,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cpu"
) -> List[float]:
    """
    Train noise predictor and SSM parameters jointly via score matching.

    Args:
        ssm: LinearSSM to train.
        diffusion: DiffusionProcess for noise schedule.
        noise_predictor: Neural network to train.
        num_trajectories: Number of trajectories to sample per epoch.
        trajectory_length: Length of each trajectory.
        num_epochs: Training epochs.
        learning_rate: Optimizer learning rate.
        device: Device to train on ("cpu" or "cuda").

    Returns:
        List of loss values per epoch.
    """
    noise_predictor = noise_predictor.to(device)
    optimizer = optim.Adam(
        list(noise_predictor.parameters()) + [
            nn.Parameter(torch.from_numpy(ssm.A.copy()).float()),
            nn.Parameter(torch.from_numpy(ssm.C.copy()).float()),
        ],
        lr=learning_rate
    )

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_transitions = 0

        for _ in range(num_trajectories):
            # Sample trajectory
            z_traj, _ = ssm.sample_trajectory(trajectory_length)

            # Extract transitions (z_{t+1} - z_t)
            transitions = z_traj[1:] - z_traj[:-1]

            # Sample random timesteps for each transition
            t_indices = np.random.randint(0, diffusion.num_steps, size=len(transitions))

            # Convert to tensors
            transitions_torch = torch.from_numpy(transitions).float().to(device)
            t_torch = torch.from_numpy(t_indices).long().to(device)

            # Forward diffusion: add noise to transitions
            x_t_list = []
            noise_true_list = []
            for trans, t_idx in zip(transitions, t_indices):
                x_t, noise_true = diffusion.forward_diffusion(trans, t_idx)
                x_t_list.append(x_t)
                noise_true_list.append(noise_true)

            x_t_torch = torch.from_numpy(np.array(x_t_list)).float().to(device)
            noise_true_torch = torch.from_numpy(np.array(noise_true_list)).float().to(device)

            # Score matching loss: predict noise and compare to true noise
            noise_pred = noise_predictor(x_t_torch, t_torch)
            loss = nn.MSELoss()(noise_pred, noise_true_torch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(transitions)
            num_transitions += len(transitions)

        avg_loss = epoch_loss / max(num_transitions, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return losses
