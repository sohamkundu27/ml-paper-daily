"""Discrete categorical diffusion model - forward and reverse process."""

import numpy as np
import torch
import torch.nn as nn


class CategoricalDiffusion:
    """Categorical diffusion with K possible states."""

    def __init__(self, num_classes: int, num_steps: int):
        """
        Args:
            num_classes: Number of discrete categories (K)
            num_steps: Number of diffusion steps
        """
        self.num_classes = num_classes
        self.num_steps = num_steps

        # Precompute transition matrices Q_t: P(x_t | x_0)
        # Linear schedule: alpha_t decreases from 1 to 0
        self.alpha = np.linspace(1.0, 1e-3, num_steps + 1)
        self.q_matrix = self._build_transition_matrices()

    def _build_transition_matrices(self) -> np.ndarray:
        """Build Q_t matrices for all timesteps.

        Q_t is a (K x K) transition matrix where Q_t[i,j] = P(x_t = i | x_0 = j).
        For a categorical diffusion, we use:
          Q_t[i,j] = alpha_t * delta(i,j) + (1 - alpha_t) / K
        This means: with prob alpha_t we stay in state j, else uniform over all states.
        """
        Q = np.zeros((self.num_steps + 1, self.num_classes, self.num_classes))

        for t in range(self.num_steps + 1):
            alpha_t = self.alpha[t]
            uniform_prob = (1 - alpha_t) / self.num_classes

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i == j:
                        Q[t, i, j] = alpha_t + uniform_prob
                    else:
                        Q[t, i, j] = uniform_prob

        return Q

    def forward(self, x0: np.ndarray, t: int) -> np.ndarray:
        """Sample x_t from q(x_t | x_0) for a given timestep.

        Args:
            x0: Discrete tokens, shape (batch_size, seq_len), values in [0, num_classes)
            t: Timestep index in [0, num_steps]

        Returns:
            x_t: Noised tokens, same shape as x0
        """
        assert 0 <= t <= self.num_steps, f"t={t} out of range [0, {self.num_steps}]"
        assert np.all((x0 >= 0) & (x0 < self.num_classes)), "x0 contains invalid class indices"

        batch_size, seq_len = x0.shape
        Q_t = self.q_matrix[t]  # (num_classes, num_classes)

        x_t = np.zeros_like(x0)
        for i in range(batch_size):
            for j in range(seq_len):
                cls = x0[i, j].astype(int)
                # Sample from q(x_t | x_0 = cls) using the transition matrix row
                x_t[i, j] = np.random.choice(
                    self.num_classes,
                    p=Q_t[:, cls]  # Probabilities for transitioning from cls
                )

        return x_t

    def forward_batch(self, x0: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
        """Sample x_t for multiple timesteps at once.

        Args:
            x0: Discrete tokens, shape (batch_size, seq_len)
            timesteps: Array of timestep indices, shape (batch_size,)

        Returns:
            x_t: Noised tokens, shape (batch_size, seq_len)
        """
        batch_size, seq_len = x0.shape
        x_t = np.zeros_like(x0)

        for i in range(batch_size):
            t = timesteps[i]
            Q_t = self.q_matrix[t]
            for j in range(seq_len):
                cls = x0[i, j].astype(int)
                x_t[i, j] = np.random.choice(
                    self.num_classes,
                    p=Q_t[:, cls]
                )

        return x_t

    def forward_with_rehash(
        self,
        x0: np.ndarray,
        t: int,
        num_corrupts: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample x_t with rehashing: corrupt only a random subset of positions.

        This implements the key ReDDiT innovation: instead of using a single
        absorbing state, we use randomized multi-index corruption patterns that
        create diverse noise paths during training.

        Args:
            x0: Discrete tokens, shape (batch_size, seq_len)
            t: Timestep index in [0, num_steps]
            num_corrupts: Number of positions to randomly corrupt per sample

        Returns:
            x_t: Noised tokens with some positions corrupted, same shape as x0
            corruption_mask: Boolean mask indicating which positions were corrupted
        """
        assert 0 <= t <= self.num_steps, f"t={t} out of range [0, {self.num_steps}]"
        assert np.all((x0 >= 0) & (x0 < self.num_classes)), "x0 contains invalid class indices"

        batch_size, seq_len = x0.shape
        Q_t = self.q_matrix[t]

        x_t = x0.copy().astype(np.int32)
        corruption_mask = np.zeros((batch_size, seq_len), dtype=bool)

        # Randomly select positions to corrupt for each sample
        for i in range(batch_size):
            # Sample random positions to corrupt, ensuring num_corrupts <= seq_len
            actual_corrupts = min(num_corrupts, seq_len)
            corrupt_positions = np.random.choice(
                seq_len,
                size=actual_corrupts,
                replace=False
            )

            for pos in corrupt_positions:
                cls = x0[i, pos].astype(int)
                x_t[i, pos] = np.random.choice(
                    self.num_classes,
                    p=Q_t[:, cls]
                )
                corruption_mask[i, pos] = True

        return x_t, corruption_mask

    def forward_batch_with_rehash(
        self,
        x0: np.ndarray,
        timesteps: np.ndarray,
        num_corrupts: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample x_t with rehashing for multiple timesteps at once.

        Args:
            x0: Discrete tokens, shape (batch_size, seq_len)
            timesteps: Array of timestep indices, shape (batch_size,)
            num_corrupts: Number of positions to randomly corrupt per sample

        Returns:
            x_t: Noised tokens with randomized corruption, shape (batch_size, seq_len)
            corruption_masks: Boolean masks for each sample, shape (batch_size, seq_len)
        """
        batch_size, seq_len = x0.shape
        x_t = x0.copy().astype(np.int32)
        corruption_masks = np.zeros((batch_size, seq_len), dtype=bool)

        for i in range(batch_size):
            t = timesteps[i]
            Q_t = self.q_matrix[t]

            # Sample random positions to corrupt
            actual_corrupts = min(num_corrupts, seq_len)
            corrupt_positions = np.random.choice(
                seq_len,
                size=actual_corrupts,
                replace=False
            )

            for pos in corrupt_positions:
                cls = x0[i, pos].astype(int)
                x_t[i, pos] = np.random.choice(
                    self.num_classes,
                    p=Q_t[:, cls]
                )
                corruption_masks[i, pos] = True

        return x_t, corruption_masks

    def reverse_kernel(
        self,
        x_t: np.ndarray,
        t: int,
        x_0_pred: np.ndarray,
    ) -> np.ndarray:
        """Compute reverse distribution q(x_{t-1} | x_t, x_0_pred).

        Args:
            x_t: Current noised tokens, shape (batch_size, seq_len)
            t: Current timestep
            x_0_pred: Predicted x_0 (argmax of model predictions), shape (batch_size, seq_len)

        Returns:
            x_t_minus_1: Sampled previous timestep, shape (batch_size, seq_len)
        """
        assert t > 0, "Cannot reverse from t=0"
        assert x_t.shape == x_0_pred.shape
        assert np.all((x_0_pred >= 0) & (x_0_pred < self.num_classes))

        batch_size, seq_len = x_t.shape
        x_t_minus_1 = np.zeros_like(x_t)

        Q_t = self.q_matrix[t]
        Q_t_minus_1 = self.q_matrix[t - 1]
        alpha_t = self.alpha[t]
        alpha_t_minus_1 = self.alpha[t - 1]

        for i in range(batch_size):
            for j in range(seq_len):
                x_t_ij = int(x_t[i, j])
                x_0_ij = int(x_0_pred[i, j])

                # Compute posterior p(x_{t-1} | x_t, x_0) analytically.
                # Using Bayes: p(x_{t-1} | x_t, x_0) ∝ p(x_t | x_{t-1}) * p(x_{t-1} | x_0)
                # This gives: p(x_{t-1} = k | x_t, x_0) ∝ Q[t](x_t | k) * Q[t-1](k | x_0)

                posterior = np.zeros(self.num_classes)
                for k in range(self.num_classes):
                    q_t_k_to_xt = Q_t[x_t_ij, k]
                    q_t1_k_from_x0 = Q_t_minus_1[k, x_0_ij]
                    posterior[k] = q_t_k_to_xt * q_t1_k_from_x0

                # Normalize to get valid probability distribution
                posterior_sum = np.sum(posterior)
                if posterior_sum > 0:
                    posterior = posterior / posterior_sum
                else:
                    posterior = np.ones(self.num_classes) / self.num_classes

                x_t_minus_1[i, j] = np.random.choice(
                    self.num_classes,
                    p=posterior
                )

        return x_t_minus_1

    def sample_with_rehash_sampler(
        self,
        denoiser,
        x_T: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """Run reverse sampling loop using rehash sampler strategy.

        Args:
            denoiser: Trained denoiser network (callable)
            x_T: Initial noised sample (random), shape (batch_size, seq_len)
            device: Torch device for inference

        Returns:
            x_0: Generated sample, shape (batch_size, seq_len)
        """
        batch_size, seq_len = x_T.shape
        x_t = x_T.copy()

        # Iteratively denoise from t=num_steps down to t=1
        for t in range(self.num_steps, 0, -1):
            # Get model prediction for x_0
            x_0_pred = self._denoise_step(denoiser, x_t, t, device)

            # Compute reverse distribution and sample x_{t-1}
            x_t = self.reverse_kernel(x_t, t, x_0_pred)

        return x_t

    def _denoise_step(
        self,
        denoiser,
        x_t: np.ndarray,
        t: int,
        device: str = "cpu",
    ) -> np.ndarray:
        """Use denoiser to predict x_0 from x_t at timestep t.

        Args:
            denoiser: Denoiser network
            x_t: Noised sample, shape (batch_size, seq_len)
            t: Current timestep
            device: Torch device

        Returns:
            x_0_pred: Predicted clean sample (argmax of logits), shape (batch_size, seq_len)
        """
        batch_size, seq_len = x_t.shape

        # Convert to torch tensors
        x_t_torch = torch.tensor(x_t, dtype=torch.long, device=device)
        t_torch = torch.tensor([t], dtype=torch.long, device=device)

        # Run denoiser (no gradients needed during inference)
        with torch.no_grad():
            logits = denoiser(x_t_torch, t_torch)  # (batch_size, seq_len, num_classes)

        # Take argmax to get predicted class for each position
        x_0_pred = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
        x_0_pred = x_0_pred.cpu().numpy().astype(np.int32)

        return x_0_pred


class SimpleDenoiser(nn.Module):
    """Simple MLP denoiser for discrete diffusion reverse process."""

    def __init__(self, num_classes: int, seq_len: int, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.time_embed_dim = 16

        # Timestep embedding
        self.time_embedding = nn.Embedding(1001, self.time_embed_dim)  # Support up to 1000 steps

        # Input: one-hot(x_t) + time_embedding concatenated over sequence
        input_dim = (num_classes + self.time_embed_dim) * seq_len

        # Simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes * seq_len),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Token indices, shape (batch_size, seq_len)
            t: Timestep(s), shape (batch_size,) or scalar

        Returns:
            logits: (batch_size, seq_len, num_classes)
        """
        batch_size, seq_len = x_t.shape

        # Handle scalar timestep
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        elif t.shape[0] == 1:
            t = t.expand(batch_size)

        # Ensure t is within bounds for embedding
        t = torch.clamp(t, 0, 1000)

        # One-hot encode x_t
        x_t_onehot = torch.nn.functional.one_hot(x_t, num_classes=self.num_classes)
        # Shape: (batch_size, seq_len, num_classes)

        # Embed timestep
        t_embed = self.time_embedding(t)  # (batch_size, time_embed_dim)
        # Expand to match sequence length
        t_embed = t_embed.unsqueeze(1).expand(batch_size, seq_len, self.time_embed_dim)
        # (batch_size, seq_len, time_embed_dim)

        # Concatenate x_t_onehot and t_embed
        combined = torch.cat([x_t_onehot, t_embed], dim=-1)
        # (batch_size, seq_len, num_classes + time_embed_dim)

        # Flatten for MLP
        combined_flat = combined.reshape(batch_size, -1)
        # (batch_size, seq_len * (num_classes + time_embed_dim))

        # Pass through MLP
        output = self.mlp(combined_flat)  # (batch_size, num_classes * seq_len)

        # Reshape back to (batch_size, seq_len, num_classes)
        logits = output.reshape(batch_size, seq_len, self.num_classes)

        return logits
