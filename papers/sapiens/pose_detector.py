import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


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


def create_synthetic_depth(image_size: int = 256, heatmap_size: int = 64) -> np.ndarray:
    """Generate synthetic depth map for body

    Returns:
        depth: (heatmap_size, heatmap_size) depth values in [0.5, 2.0] meters
    """
    depth = np.ones((heatmap_size, heatmap_size), dtype=np.float32) * 1.5
    center_y, center_x = heatmap_size // 2, heatmap_size // 2
    for i in range(heatmap_size):
        for j in range(heatmap_size):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            depth[i, j] = 1.5 + 0.3 * np.exp(-(dist**2) / (2 * 20**2))
    return np.clip(depth, 0.5, 2.0)


def create_synthetic_normals(image_size: int = 256, heatmap_size: int = 64) -> np.ndarray:
    """Generate synthetic surface normal map for body

    Returns:
        normals: (heatmap_size, heatmap_size, 3) unit normal vectors
    """
    normals = np.zeros((heatmap_size, heatmap_size, 3), dtype=np.float32)
    center_y, center_x = heatmap_size // 2, heatmap_size // 2

    for i in range(heatmap_size):
        for j in range(heatmap_size):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            angle = 2 * np.pi * dist / heatmap_size
            nx = np.sin(angle)
            ny = np.cos(angle)
            nz = np.sqrt(max(0, 1 - nx**2 - ny**2))
            normals[i, j] = np.array([nx, ny, nz])

    return normals


class MultiTaskPoseDepthNormal(nn.Module):
    """Multi-task model predicting keypoints, depth, and normals"""
    def __init__(self, num_keypoints: int = 17, image_size: int = 256):
        """
        Args:
            num_keypoints: Number of body keypoints
            image_size: Input image size
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

        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.normal_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1)
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) batch of images

        Returns:
            dict with keys:
                'keypoints': (B, num_keypoints, heatmap_H, heatmap_W)
                'depth': (B, 1, heatmap_H, heatmap_W)
                'normals': (B, 3, heatmap_H, heatmap_W)
        """
        features = self.backbone(images)

        heatmaps = self.heatmap_head(features)
        depth = self.depth_head(features)
        normals = self.normal_head(features)
        normals = F.normalize(normals, p=2, dim=1)

        return {
            'keypoints': heatmaps,
            'depth': depth,
            'normals': normals
        }

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


class MultiTaskLoss(nn.Module):
    """Combined loss for keypoint, depth, and normal prediction"""
    def __init__(self, keypoint_weight: float = 1.0, depth_weight: float = 1.0,
                 normal_weight: float = 1.0):
        super().__init__()
        self.keypoint_weight = keypoint_weight
        self.depth_weight = depth_weight
        self.normal_weight = normal_weight

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict with keys 'keypoints', 'depth', 'normals'
            targets: dict with keys 'keypoints', 'depth', 'normals'

        Returns:
            dict with loss components and total loss
        """
        keypoint_loss = F.binary_cross_entropy_with_logits(
            predictions['keypoints'], targets['keypoints']
        )

        depth_loss = F.mse_loss(predictions['depth'], targets['depth'])

        normal_pred = predictions['normals']
        normal_target = targets['normals']
        normal_pred_norm = F.normalize(normal_pred, p=2, dim=1)
        normal_target_norm = F.normalize(normal_target, p=2, dim=1)
        normal_loss = 1.0 - torch.mean(torch.sum(normal_pred_norm * normal_target_norm, dim=1))

        total_loss = (
            self.keypoint_weight * keypoint_loss +
            self.depth_weight * depth_loss +
            self.normal_weight * normal_loss
        )

        return {
            'total': total_loss,
            'keypoint': keypoint_loss,
            'depth': depth_loss,
            'normal': normal_loss
        }


def train_step(model: MultiTaskPoseDepthNormal, optimizer: torch.optim.Optimizer,
               images: torch.Tensor, targets: Dict[str, torch.Tensor],
               loss_fn: MultiTaskLoss) -> Dict[str, float]:
    """Perform one training step

    Args:
        model: Multi-task model
        optimizer: Optimizer
        images: (B, 3, H, W) batch of images
        targets: dict with keys 'keypoints', 'depth', 'normals'
        loss_fn: Loss function

    Returns:
        dict with loss components
    """
    model.train()
    optimizer.zero_grad()

    predictions = model(images)
    losses = loss_fn(predictions, targets)

    losses['total'].backward()
    optimizer.step()

    return {
        'total': losses['total'].item(),
        'keypoint': losses['keypoint'].item(),
        'depth': losses['depth'].item(),
        'normal': losses['normal'].item()
    }


class SyntheticBodyDataset:
    """Dataset that generates random synthetic human body samples on-the-fly"""
    def __init__(self, num_samples: int = 100, image_size: int = 256,
                 heatmap_size: int = 64, num_keypoints: int = 17):
        self.num_samples = num_samples
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_np, keypoints_np = create_synthetic_image_with_points(
            self.image_size, self.num_keypoints
        )
        heatmaps_np = points_to_heatmap(keypoints_np, self.image_size, self.heatmap_size)
        depth_np = create_synthetic_depth(self.image_size, self.heatmap_size)
        normals_np = create_synthetic_normals(self.image_size, self.heatmap_size)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        heatmap_tensor = torch.from_numpy(heatmaps_np).float()
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()
        normal_tensor = torch.from_numpy(normals_np).permute(2, 0, 1).float()

        return {
            'image': image_tensor,
            'keypoints': heatmap_tensor,
            'depth': depth_tensor,
            'normals': normal_tensor
        }


def create_data_batch(dataset: SyntheticBodyDataset, batch_size: int,
                     indices: np.ndarray) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Create a batch from dataset indices

    Args:
        dataset: SyntheticBodyDataset instance
        batch_size: Size of batch
        indices: Array of sample indices

    Returns:
        images batch and targets dict
    """
    images = []
    keypoints_batch = []
    depth_batch = []
    normals_batch = []

    for idx in indices:
        sample = dataset[idx]
        images.append(sample['image'])
        keypoints_batch.append(sample['keypoints'])
        depth_batch.append(sample['depth'])
        normals_batch.append(sample['normals'])

    images = torch.stack(images)
    targets = {
        'keypoints': torch.stack(keypoints_batch),
        'depth': torch.stack(depth_batch),
        'normals': torch.stack(normals_batch)
    }

    return images, targets


def apply_data_augmentation(image: torch.Tensor, heatmaps: torch.Tensor,
                           depth: torch.Tensor, normals: torch.Tensor,
                           p_flip: float = 0.5, p_rotate: float = 0.3,
                           max_brightness_delta: float = 0.1) -> Tuple:
    """Apply data augmentation to a single sample

    Args:
        image: (3, H, W) image tensor
        heatmaps: (K, H, W) keypoint heatmaps
        depth: (1, H, W) depth map
        normals: (3, H, W) normal map
        p_flip: Probability of horizontal flip
        p_rotate: Probability of rotation
        max_brightness_delta: Max brightness change ([-delta, +delta])

    Returns:
        Augmented tensors
    """
    # Horizontal flip
    if np.random.rand() < p_flip:
        image = torch.flip(image, dims=[2])
        heatmaps = torch.flip(heatmaps, dims=[2])
        depth = torch.flip(depth, dims=[2])
        normals = normals.clone()
        normals[0] = -normals[0]
        normals = torch.flip(normals, dims=[2])

    # Random brightness
    brightness_delta = np.random.uniform(-max_brightness_delta, max_brightness_delta)
    image = torch.clamp(image + brightness_delta, 0.0, 1.0)

    return image, heatmaps, depth, normals


def compute_pck(pred_keypoints: np.ndarray, target_keypoints: np.ndarray,
               threshold: float = 0.2, image_size: int = 256) -> float:
    """Compute Percentage of Correct Keypoints (PCK)

    Args:
        pred_keypoints: (B, K, 2) predicted keypoints in image space
        target_keypoints: (B, K, 2) target keypoints in image space
        threshold: PCK threshold as fraction of image diagonal
        image_size: Image size

    Returns:
        PCK score (0.0 to 1.0)
    """
    diagonal = np.sqrt(2) * image_size
    pck_threshold = threshold * diagonal

    distances = np.sqrt(np.sum((pred_keypoints - target_keypoints)**2, axis=2))
    correct = (distances <= pck_threshold).astype(np.float32)
    pck = np.mean(correct)

    return pck


def compute_depth_metrics(pred_depth: np.ndarray, target_depth: np.ndarray) -> Dict[str, float]:
    """Compute depth prediction metrics

    Args:
        pred_depth: (B, 1, H, W) predicted depth
        target_depth: (B, 1, H, W) target depth

    Returns:
        dict with 'mse' and 'mae'
    """
    pred_depth_flat = pred_depth.reshape(-1)
    target_depth_flat = target_depth.reshape(-1)

    mse = np.mean((pred_depth_flat - target_depth_flat) ** 2)
    mae = np.mean(np.abs(pred_depth_flat - target_depth_flat))

    return {'mse': mse, 'mae': mae}


def compute_normal_metrics(pred_normals: np.ndarray, target_normals: np.ndarray) -> Dict[str, float]:
    """Compute surface normal prediction metrics

    Args:
        pred_normals: (B, 3, H, W) predicted normals
        target_normals: (B, 3, H, W) target normals

    Returns:
        dict with 'mae_angle' in degrees
    """
    B, C, H, W = pred_normals.shape

    pred_normals_reshaped = pred_normals.reshape(B * H * W, 3)
    target_normals_reshaped = target_normals.reshape(B * H * W, 3)

    pred_normalized = pred_normals_reshaped / (np.linalg.norm(pred_normals_reshaped, axis=1, keepdims=True) + 1e-8)
    target_normalized = target_normals_reshaped / (np.linalg.norm(target_normals_reshaped, axis=1, keepdims=True) + 1e-8)

    dot_product = np.sum(pred_normalized * target_normalized, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angles = np.arccos(dot_product) * 180.0 / np.pi
    mae_angle = np.mean(angles)

    return {'mae_angle': mae_angle}


class Trainer:
    """Training loop for multi-task pose, depth, and normal prediction"""
    def __init__(self, model: MultiTaskPoseDepthNormal, optimizer: torch.optim.Optimizer,
                 loss_fn: MultiTaskLoss, device: str = 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(device)

        self.train_losses = []
        self.val_losses = []
        self.metrics = []

    def train_epoch(self, train_dataset: SyntheticBodyDataset, batch_size: int = 8) -> Dict[str, float]:
        """Train for one epoch

        Args:
            train_dataset: Training dataset
            batch_size: Batch size

        Returns:
            Average losses over epoch
        """
        self.model.train()
        epoch_losses = {'total': 0.0, 'keypoint': 0.0, 'depth': 0.0, 'normal': 0.0}
        num_batches = 0

        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)

        for batch_start in range(0, len(train_dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(train_dataset))
            batch_indices = indices[batch_start:batch_end]

            images, targets = create_data_batch(train_dataset, batch_size, batch_indices)
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            self.optimizer.zero_grad()
            predictions = self.model(images)
            losses = self.loss_fn(predictions, targets)

            losses['total'].backward()
            self.optimizer.step()

            epoch_losses['total'] += losses['total'].item()
            epoch_losses['keypoint'] += losses['keypoint'].item()
            epoch_losses['depth'] += losses['depth'].item()
            epoch_losses['normal'] += losses['normal'].item()
            num_batches += 1

        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        self.train_losses.append(epoch_losses)
        return epoch_losses

    def validate(self, val_dataset: SyntheticBodyDataset, batch_size: int = 8) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate model on validation set

        Args:
            val_dataset: Validation dataset
            batch_size: Batch size

        Returns:
            Validation losses and evaluation metrics
        """
        self.model.eval()
        val_losses = {'total': 0.0, 'keypoint': 0.0, 'depth': 0.0, 'normal': 0.0}
        num_batches = 0

        all_keypoint_preds = []
        all_keypoint_targets = []
        all_depth_preds = []
        all_depth_targets = []
        all_normal_preds = []
        all_normal_targets = []

        with torch.no_grad():
            for batch_start in range(0, len(val_dataset), batch_size):
                batch_end = min(batch_start + batch_size, len(val_dataset))
                batch_indices = np.arange(batch_start, batch_end)

                images, targets = create_data_batch(val_dataset, batch_size, batch_indices)
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                predictions = self.model(images)
                losses = self.loss_fn(predictions, targets)

                val_losses['total'] += losses['total'].item()
                val_losses['keypoint'] += losses['keypoint'].item()
                val_losses['depth'] += losses['depth'].item()
                val_losses['normal'] += losses['normal'].item()

                all_depth_preds.append(predictions['depth'].cpu().numpy())
                all_depth_targets.append(targets['depth'].cpu().numpy())
                all_normal_preds.append(predictions['normals'].cpu().numpy())
                all_normal_targets.append(targets['normals'].cpu().numpy())

                num_batches += 1

        for key in val_losses:
            val_losses[key] /= max(1, num_batches)

        all_depth_preds = np.concatenate(all_depth_preds, axis=0)
        all_depth_targets = np.concatenate(all_depth_targets, axis=0)
        all_normal_preds = np.concatenate(all_normal_preds, axis=0)
        all_normal_targets = np.concatenate(all_normal_targets, axis=0)

        depth_metrics = compute_depth_metrics(all_depth_preds, all_depth_targets)
        normal_metrics = compute_normal_metrics(all_normal_preds, all_normal_targets)

        metrics = {
            'depth_mse': depth_metrics['mse'],
            'depth_mae': depth_metrics['mae'],
            'normal_mae_angle': normal_metrics['mae_angle']
        }

        self.val_losses.append(val_losses)
        self.metrics.append(metrics)

        return val_losses, metrics

    def get_training_history(self) -> Dict:
        """Get full training history"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': self.metrics
        }
