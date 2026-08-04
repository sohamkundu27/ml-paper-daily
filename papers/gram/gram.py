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


class GRAMPass3(GRAMPass2):
    """
    Pass 3: Inference-time scaling and trajectory management.

    Extends Pass 2 with:
    - Parallel trajectory sampling and merging
    - Trajectory resampling based on likelihood scores
    - Variable recursion depth per trajectory (adaptive early stopping)
    - Trajectory pruning for efficiency
    """

    def sample_trajectories(self, x, num_trajectories=5, resample=True, keep_ratio=0.5):
        """
        Sample multiple trajectories in parallel with optional resampling.

        Args:
            x: input tensor of shape (batch_size, input_dim)
            num_trajectories: number of parallel trajectories to sample
            resample: if True, resample trajectories based on likelihood
            keep_ratio: fraction of trajectories to keep after resampling

        Returns:
            trajectories: list of trajectory dicts with 'output', 'trajectory', 'kl_loss', 'score'
        """
        trajectories = []

        for _ in range(num_trajectories):
            output, latent_traj, kl_loss = self.forward(x, sample=True)
            # Compute score: negative MSE reconstruction loss (higher = better)
            recon_loss = torch.mean((output - x) ** 2, dim=1)  # per-sample loss
            score = -recon_loss  # (batch_size,)

            trajectories.append(
                {
                    "output": output,
                    "trajectory": latent_traj,
                    "kl_loss": kl_loss,
                    "score": score,  # (batch_size,)
                }
            )

        # Resample trajectories if requested
        if resample:
            trajectories = self._resample_trajectories(trajectories, keep_ratio)

        return trajectories

    def _resample_trajectories(self, trajectories, keep_ratio=0.5):
        """
        Resample trajectories: keep top-scoring ones.

        Uses average score across batch for ranking.
        """
        avg_scores = []
        for traj in trajectories:
            avg_score = traj["score"].mean().item()
            avg_scores.append(avg_score)

        num_keep = max(1, int(len(trajectories) * keep_ratio))
        top_indices = sorted(
            range(len(avg_scores)), key=lambda i: avg_scores[i], reverse=True
        )[:num_keep]

        resampled = [trajectories[i] for i in top_indices]
        return resampled

    def forward_variable_depth(
        self,
        x,
        num_trajectories=5,
        max_depth=None,
        early_stopping=True,
        convergence_threshold=0.01,
    ):
        """
        Forward pass with variable recursion depth per trajectory.

        Each trajectory can stop early if the latent state converges
        (change in latent representation is below threshold).

        Args:
            x: input tensor
            num_trajectories: number of trajectories
            max_depth: maximum recursion depth (default: self.num_steps)
            early_stopping: if True, stop trajectory when converged
            convergence_threshold: threshold for convergence (L2 change in latent state)

        Returns:
            trajectories: list with outputs from trajectories at different depths
        """
        if max_depth is None:
            max_depth = self.num_steps

        trajectories = []

        for _ in range(num_trajectories):
            # Encode
            latent = self.encoder(x)
            latent_trajectory = [latent.clone().detach()]
            total_kl = torch.tensor(0.0, device=x.device, dtype=x.dtype)

            actual_depth = 0

            for step in range(max_depth):
                # Stochastic transition
                mean = self.transition_mean(latent)
                log_var = self.transition_log_var(latent)
                var = torch.exp(log_var)

                eps = torch.randn_like(latent)
                latent_new = mean + torch.sqrt(var) * eps

                latent_trajectory.append(latent_new.clone().detach())

                # KL divergence
                kl = -0.5 * torch.mean(1 + log_var - mean**2 - var)
                total_kl = total_kl + kl

                # Check for convergence (average change across batch)
                change = torch.norm(latent_new - latent, dim=1).mean().item()
                latent = latent_new
                actual_depth = step + 1

                if early_stopping and change < convergence_threshold:
                    break

            output = self.decoder(latent)

            trajectories.append(
                {
                    "output": output,
                    "trajectory": latent_trajectory,
                    "kl_loss": total_kl,
                    "depth": actual_depth,
                }
            )

        return trajectories

    def ensemble_outputs(self, trajectories, method="mean"):
        """
        Combine multiple trajectories into a single output.

        Args:
            trajectories: list of trajectory dicts from sample_trajectories or forward_variable_depth
            method: 'mean' (average outputs), 'best' (select best trajectory by score)

        Returns:
            combined_output: tensor of shape (batch_size, output_dim)
        """
        outputs = torch.stack([traj["output"] for traj in trajectories], dim=0)

        if method == "mean":
            return outputs.mean(dim=0)
        elif method == "best":
            # Return output from trajectory with highest average score
            if "score" in trajectories[0]:
                scores = [traj["score"].mean().item() for traj in trajectories]
                best_idx = scores.index(max(scores))
            else:
                # Fallback: use KL loss as score (lower is better)
                scores = [-traj["kl_loss"].item() for traj in trajectories]
                best_idx = scores.index(max(scores))
            return trajectories[best_idx]["output"]
        else:
            raise ValueError(f"Unknown method: {method}")
