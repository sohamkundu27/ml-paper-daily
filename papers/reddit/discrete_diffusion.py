"""Discrete categorical diffusion model - forward process."""

import numpy as np


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
