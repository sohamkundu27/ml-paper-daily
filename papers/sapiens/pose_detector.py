import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class SimpleCNNBackbone(nn.Module):
    """Lightweight CNN backbone for image feature extraction"""
    def __init__(self, input_channels: int = 3, feature_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, feature_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(feature_channels)
        self.conv2 = nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_channels * 2)
        self.conv3 = nn.Conv2d(feature_channels * 2, feature_channels * 4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class KeypointDetector(nn.Module):
    """Predict 2D keypoints as heatmaps"""
    def __init__(self, num_keypoints: int = 17, image_size: int = 256):
        """
        Args:
            num_keypoints: Number of body keypoints (e.g., 17 for standard pose)
            image_size: Input image size (assumes square images)
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.image_size = image_size
        self.heatmap_size = image_size // 4

        self.backbone = SimpleCNNBackbone(input_channels=3, feature_channels=64)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_keypoints, kernel_size=1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Batch of images (B, 3, H, W)

        Returns:
            heatmaps: (B, num_keypoints, heatmap_H, heatmap_W)
        """
        features = self.backbone(images)
        heatmaps = self.heatmap_head(features)
        return heatmaps

    def get_keypoints_from_heatmaps(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Extract (x, y) coordinates from heatmaps via argmax

        Args:
            heatmaps: (B, num_keypoints, H, W)

        Returns:
            keypoints: (B, num_keypoints, 2) in image space
        """
        B, K, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.view(B, K, -1)
        indices = torch.argmax(heatmaps_flat, dim=2)

        y_coords = indices // W
        x_coords = indices % W

        scale = self.image_size / H
        keypoints = torch.stack([x_coords.float() * scale, y_coords.float() * scale], dim=2)
        return keypoints


def create_synthetic_body_points(image_size: int = 256, num_points: int = 17) -> np.ndarray:
    """Generate synthetic body keypoints for testing

    Returns:
        points: (num_points, 2) array of (x, y) coordinates
    """
    points = []
    base_y = image_size // 4
    center_x = image_size // 2
    max_offset = min(60, image_size // 4)

    points.append([center_x, base_y])
    points.append([center_x - 25, base_y + 30])
    points.append([center_x + 25, base_y + 30])
    points.append([center_x - 35, base_y + 60])
    points.append([center_x + 35, base_y + 60])
    points.append([center_x - 40, base_y + 90])
    points.append([center_x + 40, base_y + 90])
    points.append([center_x - 15, base_y + 75])
    points.append([center_x + 15, base_y + 75])
    points.append([center_x - 15, base_y + 120])
    points.append([center_x + 15, base_y + 120])
    points.append([center_x - 15, base_y + 150])
    points.append([center_x + 15, base_y + 150])

    while len(points) < 17:
        points.append([center_x, base_y + 40])

    points_array = np.array(points[:17], dtype=np.float32)
    points_array[:, 0] = np.clip(points_array[:, 0], 5, image_size - 5)
    points_array[:, 1] = np.clip(points_array[:, 1], 5, image_size - 5)

    return points_array


def points_to_heatmap(points: np.ndarray, image_size: int = 256, heatmap_size: int = 64,
                      sigma: float = 2.0) -> np.ndarray:
    """Convert keypoint locations to Gaussian heatmaps

    Args:
        points: (num_keypoints, 2) array of (x, y) in image space
        image_size: Original image size
        heatmap_size: Size of output heatmap
        sigma: Gaussian standard deviation

    Returns:
        heatmaps: (num_keypoints, heatmap_size, heatmap_size)
    """
    num_keypoints = points.shape[0]
    heatmaps = np.zeros((num_keypoints, heatmap_size, heatmap_size), dtype=np.float32)
    scale = heatmap_size / image_size

    for k, (x, y) in enumerate(points):
        x_hm = x * scale
        y_hm = y * scale

        for i in range(heatmap_size):
            for j in range(heatmap_size):
                dist = np.sqrt((i - y_hm)**2 + (j - x_hm)**2)
                heatmaps[k, i, j] = np.exp(-(dist**2) / (2 * sigma**2))

    return heatmaps


def create_synthetic_image_with_points(image_size: int = 256, num_points: int = 17) -> Tuple[np.ndarray, np.ndarray]:
    """Create a synthetic image with visible keypoints

    Returns:
        image: (H, W, 3) RGB image
        keypoints: (num_points, 2) keypoint coordinates
    """
    image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 200
    keypoints = create_synthetic_body_points(image_size, num_points)

    for x, y in keypoints:
        x, y = int(x), int(y)
        y = np.clip(y, 0, image_size - 1)
        x = np.clip(x, 0, image_size - 1)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = y + dy, x + dx
                if 0 <= ny < image_size and 0 <= nx < image_size:
                    if dx**2 + dy**2 <= 9:
                        image[ny, nx] = [50, 100, 200]

    return image, keypoints
