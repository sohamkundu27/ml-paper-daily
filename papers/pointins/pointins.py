import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List


class PointCloudAugmentation:
    """Point cloud augmentation with rotation, jittering, and scaling."""

    def __init__(self, jitter_std: float = 0.01, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.jitter_std = jitter_std
        self.scale_range = scale_range

    def random_rotation(self, points: np.ndarray) -> np.ndarray:
        """Apply random rotation around z-axis."""
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        return points @ rotation.T

    def random_jitter(self, points: np.ndarray) -> np.ndarray:
        """Add Gaussian jitter to point coordinates."""
        noise = np.random.normal(0, self.jitter_std, points.shape)
        return points + noise

    def random_scale(self, points: np.ndarray) -> np.ndarray:
        """Apply random isotropic scaling."""
        scale = np.random.uniform(*self.scale_range)
        return points * scale

    def augment(self, points: np.ndarray) -> np.ndarray:
        """Apply all augmentations in sequence."""
        points = self.random_rotation(points)
        points = self.random_jitter(points)
        points = self.random_scale(points)
        return points


class SimplePointCloudEncoder(nn.Module):
    """Simple PointNet-style encoder: MLPs on per-point features."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            points: (batch_size, num_points, input_dim) point coordinates

        Returns:
            cloud_feature: (batch_size, output_dim) global point cloud feature
        """
        # Apply per-point MLP: (batch, num_points, input_dim) -> (batch, num_points, output_dim)
        per_point_features = self.mlp(points)

        # Max pooling over points: (batch, num_points, output_dim) -> (batch, output_dim, 1)
        batch_size = per_point_features.size(0)
        per_point_features_t = per_point_features.transpose(1, 2)  # (batch, output_dim, num_points)
        cloud_feature = self.pool(per_point_features_t).squeeze(-1)  # (batch, output_dim)

        return cloud_feature


class PointCloudDataset:
    """Dataset for generating positive/negative point cloud pairs."""

    def __init__(self, num_objects: int = 32, num_points: int = 1024, point_range: float = 1.0):
        """
        Initialize synthetic point cloud dataset.

        Args:
            num_objects: Number of distinct 3D point cloud instances
            num_points: Points per cloud
            point_range: Spatial range of generated points
        """
        self.num_objects = num_objects
        self.num_points = num_points
        self.point_range = point_range
        self.augmentation = PointCloudAugmentation()
        self.clouds = self._generate_synthetic_clouds()

    def _generate_synthetic_clouds(self) -> List[np.ndarray]:
        """Generate random synthetic point clouds."""
        clouds = []
        for _ in range(self.num_objects):
            points = np.random.uniform(-self.point_range, self.point_range,
                                      (self.num_points, 3)).astype(np.float32)
            clouds.append(points)
        return clouds

    def get_positive_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get two augmentations of the same point cloud (positive pair)."""
        cloud = self.clouds[idx]
        aug1 = self.augmentation.augment(cloud.copy())
        aug2 = self.augmentation.augment(cloud.copy())
        return aug1.astype(np.float32), aug2.astype(np.float32)

    def get_negative_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get augmentations from two different point clouds (negative pair)."""
        idx2 = (idx + 1 + np.random.randint(0, self.num_objects - 1)) % self.num_objects
        cloud1 = self.clouds[idx]
        cloud2 = self.clouds[idx2]
        aug1 = self.augmentation.augment(cloud1.copy())
        aug2 = self.augmentation.augment(cloud2.copy())
        return aug1.astype(np.float32), aug2.astype(np.float32)

    def __len__(self) -> int:
        return self.num_objects


def normalize_points(points: torch.Tensor) -> torch.Tensor:
    """Normalize point cloud to zero mean and unit variance."""
    mean = points.mean(dim=1, keepdim=True)
    std = points.std(dim=1, keepdim=True)
    return (points - mean) / (std + 1e-6)
