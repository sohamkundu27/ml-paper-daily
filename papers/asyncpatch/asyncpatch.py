import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class HeterogeneousNoiseScheduler:
    """
    Assigns independent noise levels to spatial patches.

    For each patch, we sample or assign a timestep t ∈ [0, T],
    then apply diffusion corruption level α(t) to that patch.
    """

    def __init__(self, T: int = 1000):
        """
        Args:
            T: Total diffusion steps
        """
        self.T = T
        self.alpha_cumsum = self._compute_alpha_cumsum()

    def _compute_alpha_cumsum(self) -> np.ndarray:
        """Linear schedule: α_t = 1 - t/T, α_cumprod_t = product of α_i for i <= t"""
        beta = np.linspace(0.0001, 0.02, self.T)
        alpha = 1 - beta
        alpha_cumprod = np.cumprod(alpha)
        return alpha_cumprod

    def get_alpha_t(self, t: int) -> float:
        """Signal retention at step t: sqrt(α̅_t)"""
        return np.sqrt(self.alpha_cumsum[min(t, self.T - 1)])

    def get_sigma_t(self, t: int) -> float:
        """Noise level at step t: sqrt(1 - α̅_t)"""
        return np.sqrt(1.0 - self.alpha_cumsum[min(t, self.T - 1)])

    def sample_heterogeneous_timesteps(
        self,
        batch_size: int,
        num_patches: int,
        strategy: str = "uniform"
    ) -> np.ndarray:
        """
        Sample timesteps for each patch independently.

        Args:
            batch_size: Number of images
            num_patches: Total number of patches per image
            strategy: 'uniform' (each patch gets random t) or 'mixed' (some patches fixed at 0)

        Returns:
            Array of shape (batch_size, num_patches) with timestep assignments
        """
        if strategy == "uniform":
            return np.random.randint(0, self.T, size=(batch_size, num_patches))
        elif strategy == "mixed":
            # 50% of patches stay clean (t=0), others sample uniformly
            timesteps = np.random.randint(0, self.T, size=(batch_size, num_patches))
            mask = np.random.rand(batch_size, num_patches) < 0.5
            timesteps[mask] = 0
            return timesteps
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class AsyncPatchDiffusion:
    """
    Forward diffusion with heterogeneous per-patch noise levels.

    Given an image and per-patch timestep assignments, applies
    independent Gaussian noise to each patch according to its schedule.
    """

    def __init__(self, scheduler: HeterogeneousNoiseScheduler):
        self.scheduler = scheduler

    def forward(
        self,
        x0: torch.Tensor,
        timesteps: np.ndarray,
        patch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion with heterogeneous noise.

        Args:
            x0: Clean image, shape (batch, channels, height, width)
            timesteps: Timestep per patch, shape (batch, num_patches)
            patch_size: Size of each patch (for reconstruction)

        Returns:
            xt: Corrupted image, shape (batch, channels, height, width)
            noise: Applied noise, same shape as xt
        """
        batch_size, channels, height, width = x0.shape

        # Reshape into patches
        # Assume patches tile the image: num_patches = (height*width) / patch_size^2
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        num_patches = num_patches_h * num_patches_w

        # Unfold into patches: (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        patches = x0.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(
            batch_size, channels, num_patches_h * num_patches_w, patch_size, patch_size
        )  # (B, C, num_patches, patch_size, patch_size)

        # Apply heterogeneous noise to each patch
        noise = torch.randn_like(patches)
        corrupted_patches = patches.clone()

        for b in range(batch_size):
            for p in range(num_patches):
                t = timesteps[b, p]
                alpha_t = self.scheduler.get_alpha_t(t)
                sigma_t = self.scheduler.get_sigma_t(t)

                corrupted_patches[b, :, p] = alpha_t * patches[b, :, p] + sigma_t * noise[b, :, p]

        # Fold patches back into image
        xt = self._fold_patches(corrupted_patches, batch_size, channels, height, width, patch_size)
        noise_unfolded = self._fold_patches(noise, batch_size, channels, height, width, patch_size)

        return xt, noise_unfolded

    def _fold_patches(
        self,
        patches: torch.Tensor,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        patch_size: int
    ) -> torch.Tensor:
        """Reconstruct image from patches."""
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size

        # Reshape patches for fold
        patches_folded = patches.view(
            batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size
        )
        patches_folded = patches_folded.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_folded = patches_folded.view(batch_size, channels, height, width)

        return patches_folded

    def get_patch_variance(
        self,
        timesteps: np.ndarray,
        patch_size: int = 1
    ) -> np.ndarray:
        """
        Compute variance at each patch based on its timestep.
        For validation: variance should be 1 across all patches.

        Returns:
            Array of shape (batch_size, num_patches) with variances
        """
        batch_size, num_patches = timesteps.shape
        variance = np.zeros((batch_size, num_patches))

        for b in range(batch_size):
            for p in range(num_patches):
                t = timesteps[b, p]
                alpha_t = self.scheduler.get_alpha_t(t)
                sigma_t = self.scheduler.get_sigma_t(t)
                variance[b, p] = alpha_t**2 + sigma_t**2

        return variance

    def get_patch_snr(self, timesteps: np.ndarray) -> np.ndarray:
        """
        Compute signal-to-noise ratio (SNR) at each patch.
        SNR(t) = log(α̅_t / (1 - α̅_t))

        Returns:
            Array of shape (batch_size, num_patches) with SNR values in dB
        """
        batch_size, num_patches = timesteps.shape
        snr = np.zeros((batch_size, num_patches))

        for b in range(batch_size):
            for p in range(num_patches):
                t = timesteps[b, p]
                alpha_t = self.scheduler.get_alpha_t(t)
                sigma_t = self.scheduler.get_sigma_t(t)
                # SNR = (signal_power) / (noise_power) = α̅_t / (1 - α̅_t)
                # In dB: 10 * log10(SNR)
                alpha_cumprod = self.scheduler.alpha_cumsum[min(t, self.scheduler.T - 1)]
                snr_val = alpha_cumprod / (1.0 - alpha_cumprod + 1e-10)
                snr[b, p] = 10.0 * np.log10(snr_val + 1e-10)

        return snr

    def compute_joint_diffusion_properties(
        self,
        timesteps: np.ndarray
    ) -> dict:
        """
        Compute comprehensive theoretical properties of the joint diffusion process.

        Returns:
            Dictionary with:
              - mean_snr: average SNR across all patches
              - min_snr, max_snr: range of SNR values
              - variance_preservation: boolean indicating if all variances ≈ 1
              - patch_statistics: per-patch alpha, sigma, SNR values
        """
        batch_size, num_patches = timesteps.shape
        variances = self.get_patch_variance(timesteps)
        snr_values = self.get_patch_snr(timesteps)

        alpha_vals = np.zeros((batch_size, num_patches))
        sigma_vals = np.zeros((batch_size, num_patches))

        for b in range(batch_size):
            for p in range(num_patches):
                t = timesteps[b, p]
                alpha_vals[b, p] = self.scheduler.get_alpha_t(t)
                sigma_vals[b, p] = self.scheduler.get_sigma_t(t)

        # Check that each patch's alpha^2 + sigma^2 ≈ 1 (variance preservation)
        patch_variances = alpha_vals**2 + sigma_vals**2

        return {
            "mean_snr": snr_values.mean(),
            "min_snr": snr_values.min(),
            "max_snr": snr_values.max(),
            "mean_alpha": alpha_vals.mean(),
            "mean_sigma": sigma_vals.mean(),
            "variance_preservation": np.allclose(variances, 1.0, atol=0.01),
            "patch_variances_preserved": np.allclose(patch_variances, 1.0, atol=0.01),
            "max_variance_error": np.abs(variances - 1.0).max(),
            "timestep_range": (timesteps.min(), timesteps.max()),
            "patch_statistics": {
                "alpha": alpha_vals,
                "sigma": sigma_vals,
                "snr": snr_values,
                "variance": variances
            }
        }
