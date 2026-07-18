import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


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


class SimpleDenoiser(nn.Module):
    """
    Simple ConvNet denoiser for heterogeneous diffusion.

    Conditions on per-patch timesteps by concatenating a timestep map
    to the input. Outputs predicted noise for the corrupted image.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 16):
        super().__init__()
        # in_channels + 1: original channels + timestep map
        self.conv1 = nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: 'HeterogeneousNoiseScheduler',
        patch_size: int = 1
    ) -> torch.Tensor:
        """
        Denoise by predicting noise.

        Args:
            xt: Corrupted image, shape (batch, channels, height, width)
            timesteps: Timestep per patch, shape (batch, num_patches)
            scheduler: HeterogeneousNoiseScheduler instance
            patch_size: Size of each patch

        Returns:
            Predicted noise, shape (batch, channels, height, width)
        """
        batch_size, channels, height, width = xt.shape

        # Create timestep map: normalize timesteps to [0, 1] and expand to spatial dims
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size

        # Expand per-patch timesteps to full spatial resolution
        timestep_map = torch.zeros(batch_size, 1, height, width, device=xt.device)
        for b in range(batch_size):
            idx = 0
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    t_val = timesteps[b, idx].item() if isinstance(timesteps[b, idx], torch.Tensor) else timesteps[b, idx]
                    normalized_t = t_val / scheduler.T
                    timestep_map[b, 0,
                                 i * patch_size:(i + 1) * patch_size,
                                 j * patch_size:(j + 1) * patch_size] = normalized_t
                    idx += 1

        # Concatenate timestep map to input
        x = torch.cat([xt, timestep_map], dim=1)

        # Simple ConvNet forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class DenoisingTrainer:
    """
    Trainer for the heterogeneous denoiser.

    Handles forward pass corruption, denoising predictions, and loss computation.
    """

    def __init__(
        self,
        scheduler: HeterogeneousNoiseScheduler,
        denoiser: SimpleDenoiser,
        device: str = "cpu"
    ):
        self.scheduler = scheduler
        self.denoiser = denoiser.to(device)
        self.device = device
        self.diffusion = AsyncPatchDiffusion(scheduler)

    def compute_loss(
        self,
        x0: torch.Tensor,
        timesteps: np.ndarray,
        patch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute denoising loss (MSE between predicted and true noise).

        Args:
            x0: Clean image, shape (batch, channels, height, width)
            timesteps: Per-patch timesteps, shape (batch, num_patches)
            patch_size: Size of each patch

        Returns:
            Scalar loss
        """
        x0 = x0.to(self.device)

        # Forward diffusion: corrupt x0
        xt, noise_true = self.diffusion.forward(x0, timesteps, patch_size=patch_size)
        xt = xt.to(self.device)
        noise_true = noise_true.to(self.device)

        # Convert timesteps to tensor
        timesteps_tensor = torch.from_numpy(timesteps).long().to(self.device)

        # Denoiser predicts noise
        noise_pred = self.denoiser(xt, timesteps_tensor, self.scheduler, patch_size=patch_size)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise_true)
        return loss

    def train_step(
        self,
        x0: torch.Tensor,
        timesteps: np.ndarray,
        optimizer: torch.optim.Optimizer,
        patch_size: int = 1
    ) -> float:
        """
        Single training step.

        Args:
            x0: Clean image batch
            timesteps: Per-patch timesteps
            optimizer: PyTorch optimizer
            patch_size: Size of each patch

        Returns:
            Loss value (scalar, detached)
        """
        self.denoiser.train()
        optimizer.zero_grad()

        loss = self.compute_loss(x0, timesteps, patch_size=patch_size)
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_loss(
        self,
        x0: torch.Tensor,
        timesteps: np.ndarray,
        patch_size: int = 1
    ) -> float:
        """
        Evaluate loss without training.

        Args:
            x0: Clean image batch
            timesteps: Per-patch timesteps
            patch_size: Size of each patch

        Returns:
            Loss value (scalar, detached)
        """
        self.denoiser.eval()
        with torch.no_grad():
            loss = self.compute_loss(x0, timesteps, patch_size=patch_size)
        return loss.item()

    def sample(
        self,
        xt: torch.Tensor,
        timesteps: np.ndarray,
        num_steps: int = 5,
        patch_size: int = 1
    ) -> torch.Tensor:
        """
        Simple iterative denoising (reverse diffusion).

        Starts from corrupted image xt and iteratively denoises by predicting
        noise and reducing timesteps. This is a simplified reverse process.

        Args:
            xt: Corrupted image, shape (batch, channels, height, width)
            timesteps: Per-patch timesteps, shape (batch, num_patches)
            num_steps: Number of denoising iterations
            patch_size: Size of each patch

        Returns:
            Denoised image, shape (batch, channels, height, width)
        """
        self.denoiser.eval()
        xt = xt.to(self.device)

        current_timesteps = timesteps.copy()
        current_x = xt.clone()

        with torch.no_grad():
            for step in range(num_steps):
                # Predict noise
                timesteps_tensor = torch.from_numpy(current_timesteps).long().to(self.device)
                noise_pred = self.denoiser(
                    current_x,
                    timesteps_tensor,
                    self.scheduler,
                    patch_size=patch_size
                )

                # Reduce timestep (simple linear schedule)
                step_size = int(self.scheduler.T / (num_steps + 1))
                current_timesteps = np.maximum(current_timesteps - step_size, 0)

                # Update x: subtract noise scaled by step size
                current_x = current_x - 0.1 * noise_pred

        return current_x

    def inpaint(
        self,
        x0_masked: torch.Tensor,
        inpaint_mask: torch.Tensor,
        timesteps: np.ndarray,
        num_steps: int = 5,
        patch_size: int = 1
    ) -> torch.Tensor:
        """
        Inpainting: sample unknown regions while keeping known regions fixed.

        Args:
            x0_masked: Input image with unknown regions (will be corrupted), shape (B, C, H, W)
            inpaint_mask: Binary mask where 1 = known (keep fixed), 0 = unknown (inpaint),
                         shape (B, 1, H, W)
            timesteps: Per-patch timesteps for corruption, shape (batch, num_patches)
            num_steps: Number of denoising iterations
            patch_size: Size of each patch

        Returns:
            Inpainted image with unknown regions filled in, shape (B, C, H, W)
        """
        self.denoiser.eval()
        x0_masked = x0_masked.to(self.device)
        inpaint_mask = inpaint_mask.to(self.device)

        # Corrupt the unknown regions by applying forward diffusion
        xt, _ = self.diffusion.forward(x0_masked, timesteps, patch_size=patch_size)
        xt = xt.to(self.device)

        # Blend: keep known regions from original, use corrupted for unknown
        xt = x0_masked * inpaint_mask + xt * (1.0 - inpaint_mask)

        current_timesteps = timesteps.copy()
        current_x = xt.clone()

        with torch.no_grad():
            for step in range(num_steps):
                # Predict noise
                timesteps_tensor = torch.from_numpy(current_timesteps).long().to(self.device)
                noise_pred = self.denoiser(
                    current_x,
                    timesteps_tensor,
                    self.scheduler,
                    patch_size=patch_size
                )

                # Reduce timestep
                step_size = int(self.scheduler.T / (num_steps + 1))
                current_timesteps = np.maximum(current_timesteps - step_size, 0)

                # Update only unknown regions
                current_x = current_x - 0.1 * noise_pred
                current_x = x0_masked * inpaint_mask + current_x * (1.0 - inpaint_mask)

        return current_x
