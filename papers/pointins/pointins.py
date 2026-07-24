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


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss for positive pairs.

        For batch_size > 1: uses full NT-Xent with batch negatives
        For batch_size = 1: uses simplified pairwise loss

        Args:
            z_i: (batch_size, feature_dim) representations from view i
            z_j: (batch_size, feature_dim) representations from view j

        Returns:
            loss: scalar loss value
        """
        batch_size = z_i.size(0)

        # Normalize features
        z_i = torch.nn.functional.normalize(z_i, dim=1)
        z_j = torch.nn.functional.normalize(z_j, dim=1)

        if batch_size == 1:
            # For single pair, use simple negative of cosine similarity
            # This encourages z_i and z_j to be similar (cosine similarity close to 1)
            sim = torch.nn.functional.cosine_similarity(z_i, z_j)
            loss = 1.0 - sim
            return loss.mean()

        # For batch_size > 1: use full NT-Xent
        # Concatenate: (2*batch_size, feature_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix: (2*batch_size, 2*batch_size)
        sim_matrix = torch.mm(z, z.T) / self.temperature

        # Create labels
        labels = torch.arange(batch_size, dtype=torch.long, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix_masked = sim_matrix.masked_fill(mask, -1e9)

        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(sim_matrix_masked, labels)

        return loss


class MomentumEncoder(nn.Module):
    """Momentum-updated encoder for stable contrastive learning."""

    def __init__(self, encoder: nn.Module, momentum: float = 0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum_encoder = self._clone_encoder(encoder)
        self.momentum = momentum

        # Freeze momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False

    def _clone_encoder(self, encoder: nn.Module) -> nn.Module:
        """Create a deep copy of the encoder."""
        import copy
        return copy.deepcopy(encoder)

    @torch.no_grad()
    def update_momentum(self):
        """Update momentum encoder weights."""
        for p_main, p_momentum in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            p_momentum.data = p_momentum.data * self.momentum + p_main.data * (1 - self.momentum)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both encoders.

        Args:
            x: (batch_size, num_points, 3) point cloud

        Returns:
            z_online: features from main encoder
            z_momentum: features from momentum encoder (detached)
        """
        z_online = self.encoder(x)
        with torch.no_grad():
            z_momentum = self.momentum_encoder(x)
        return z_online, z_momentum


class PointCloudEncoder(nn.Module):
    """PointNet-style encoder with optional orthogonal offset branch."""

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

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning global features and per-point features.

        Args:
            points: (batch_size, num_points, input_dim) point coordinates

        Returns:
            cloud_feature: (batch_size, output_dim) global point cloud feature
            per_point_features: (batch_size, num_points, output_dim) per-point features
        """
        per_point_features = self.mlp(points)
        batch_size = per_point_features.size(0)
        per_point_features_t = per_point_features.transpose(1, 2)
        cloud_feature = self.pool(per_point_features_t).squeeze(-1)

        return cloud_feature, per_point_features


class OrthogonalOffsetBranch(nn.Module):
    """Predicts per-point orthogonal offsets for geometry-aware learning."""

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.offset_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, per_point_features: torch.Tensor) -> torch.Tensor:
        """
        Predict per-point offset vectors.

        Args:
            per_point_features: (batch_size, num_points, feature_dim)

        Returns:
            offsets: (batch_size, num_points, 3) offset vectors
        """
        offsets = self.offset_head(per_point_features)
        return offsets


def compute_local_normals(points: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Estimate local surface normals using k-nearest neighbors PCA.

    Args:
        points: (batch_size, num_points, 3) point coordinates
        k: number of nearest neighbors for local geometry

    Returns:
        normals: (batch_size, num_points, 3) estimated normal vectors
    """
    batch_size, num_points, _ = points.shape
    normals = torch.zeros_like(points)

    for b in range(batch_size):
        cloud = points[b]  # (num_points, 3)

        for i in range(num_points):
            center = cloud[i:i+1]  # (1, 3)
            distances = torch.norm(cloud - center, dim=1)

            if num_points > k:
                _, nearest_indices = torch.topk(distances, k, largest=False)
            else:
                nearest_indices = torch.arange(num_points, device=points.device)

            neighbors = cloud[nearest_indices]  # (k, 3)
            centered = neighbors - neighbors.mean(dim=0, keepdim=True)

            _, _, V = torch.svd(centered)
            normal = V[:, -1]  # Last singular vector = normal to the plane
            normals[b, i] = normal

    return normals


def geometry_aware_loss(offsets: torch.Tensor, points: torch.Tensor, lambda_geo: float = 0.1) -> torch.Tensor:
    """
    Geometry-aware regularization loss encouraging offsets to align with surface normals.

    Args:
        offsets: (batch_size, num_points, 3) predicted offset vectors
        points: (batch_size, num_points, 3) point coordinates
        lambda_geo: weight for geometry loss

    Returns:
        loss: scalar geometry regularization loss
    """
    normals = compute_local_normals(points, k=min(8, points.size(1)))

    offset_norms = torch.norm(offsets, dim=2, keepdim=True) + 1e-8
    offset_normalized = offsets / offset_norms

    alignment = torch.abs((offset_normalized * normals).sum(dim=2))
    geometry_loss = lambda_geo * (1.0 - alignment.mean())

    return geometry_loss


class ContrastiveTrainer:
    """Trainer for instance discrimination with contrastive learning."""

    def __init__(self, encoder: nn.Module, device: str = "cpu", lr: float = 1e-3, use_momentum: bool = True):
        self.device = device
        self.encoder = encoder.to(device)
        self.use_momentum = use_momentum
        if use_momentum:
            self.momentum_encoder_wrapper = MomentumEncoder(encoder, momentum=0.999).to(device)
        self.loss_fn = NTXentLoss(temperature=0.07)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def train_step(self, cloud1: torch.Tensor, cloud2: torch.Tensor) -> float:
        """
        Single training step on a positive pair.

        Args:
            cloud1: (batch_size, num_points, 3) first augmentation
            cloud2: (batch_size, num_points, 3) second augmentation

        Returns:
            loss: scalar loss value
        """
        cloud1 = cloud1.to(self.device)
        cloud2 = cloud2.to(self.device)

        # Normalize point clouds
        cloud1 = normalize_points(cloud1)
        cloud2 = normalize_points(cloud2)

        # Forward pass through encoder
        z1 = self.encoder(cloud1)
        z2 = self.encoder(cloud2)

        # Compute contrastive loss between the two augmentations
        loss = self.loss_fn(z1, z2)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update momentum encoder if used
        if self.use_momentum:
            self.momentum_encoder_wrapper.update_momentum()

        return loss.item()

    def evaluate_similarity(self, dataset: PointCloudDataset, num_samples: int = 16) -> Tuple[float, float]:
        """
        Evaluate the encoder by computing cosine similarity on positive and negative pairs.

        Args:
            dataset: PointCloudDataset instance
            num_samples: number of pairs to evaluate

        Returns:
            pos_sim: mean cosine similarity for positive pairs
            neg_sim: mean cosine similarity for negative pairs
        """
        self.encoder.eval()

        pos_sims = []
        neg_sims = []

        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                # Positive pair
                cloud1, cloud2 = dataset.get_positive_pair(i)
                cloud1_t = torch.from_numpy(cloud1).unsqueeze(0).to(self.device)
                cloud2_t = torch.from_numpy(cloud2).unsqueeze(0).to(self.device)

                cloud1_t = normalize_points(cloud1_t)
                cloud2_t = normalize_points(cloud2_t)

                feat1 = self.encoder(cloud1_t)
                feat2 = self.encoder(cloud2_t)

                feat1_norm = torch.nn.functional.normalize(feat1, dim=1)
                feat2_norm = torch.nn.functional.normalize(feat2, dim=1)

                pos_sim = (feat1_norm * feat2_norm).sum(dim=1).item()
                pos_sims.append(pos_sim)

                # Negative pair
                cloud1, cloud2 = dataset.get_negative_pair(i)
                cloud1_t = torch.from_numpy(cloud1).unsqueeze(0).to(self.device)
                cloud2_t = torch.from_numpy(cloud2).unsqueeze(0).to(self.device)

                cloud1_t = normalize_points(cloud1_t)
                cloud2_t = normalize_points(cloud2_t)

                feat1 = self.encoder(cloud1_t)
                feat2 = self.encoder(cloud2_t)

                feat1_norm = torch.nn.functional.normalize(feat1, dim=1)
                feat2_norm = torch.nn.functional.normalize(feat2, dim=1)

                neg_sim = (feat1_norm * feat2_norm).sum(dim=1).item()
                neg_sims.append(neg_sim)

        self.encoder.train()

        mean_pos_sim = np.mean(pos_sims)
        mean_neg_sim = np.mean(neg_sims)

        return mean_pos_sim, mean_neg_sim


class GeometryAwareTrainer:
    """Trainer combining instance discrimination with geometry-aware offset learning."""

    def __init__(self, encoder: nn.Module, offset_branch: nn.Module, device: str = "cpu",
                 lr: float = 1e-3, lambda_geo: float = 0.1, use_momentum: bool = False):
        self.device = device
        self.encoder = encoder.to(device)
        self.offset_branch = offset_branch.to(device)
        self.use_momentum = use_momentum
        if use_momentum:
            self.momentum_encoder_wrapper = MomentumEncoder(encoder, momentum=0.999).to(device)
        self.loss_fn = NTXentLoss(temperature=0.07)
        self.lambda_geo = lambda_geo

        # Combined optimizer for encoder and offset branch
        params = list(encoder.parameters()) + list(offset_branch.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def train_step(self, cloud1: torch.Tensor, cloud2: torch.Tensor) -> Tuple[float, float, float]:
        """
        Single training step on a positive pair with geometry-aware loss.

        Args:
            cloud1: (batch_size, num_points, 3) first augmentation
            cloud2: (batch_size, num_points, 3) second augmentation

        Returns:
            total_loss: combined loss value
            contrastive_loss: instance discrimination loss
            geo_loss: geometry-aware regularization loss
        """
        cloud1 = cloud1.to(self.device)
        cloud2 = cloud2.to(self.device)

        cloud1 = normalize_points(cloud1)
        cloud2 = normalize_points(cloud2)

        # Forward pass through encoder (returns global feature + per-point features)
        z1_global, z1_per_point = self.encoder(cloud1)
        z2_global, z2_per_point = self.encoder(cloud2)

        # Contrastive loss on global features
        contrastive_loss = self.loss_fn(z1_global, z2_global)

        # Geometry-aware loss from offset predictions
        offsets1 = self.offset_branch(z1_per_point)
        offsets2 = self.offset_branch(z2_per_point)

        geo_loss1 = geometry_aware_loss(offsets1, cloud1, lambda_geo=self.lambda_geo)
        geo_loss2 = geometry_aware_loss(offsets2, cloud2, lambda_geo=self.lambda_geo)
        geo_loss = (geo_loss1 + geo_loss2) / 2.0

        # Combined loss
        total_loss = contrastive_loss + geo_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update momentum encoder if used
        if self.use_momentum:
            self.momentum_encoder_wrapper.update_momentum()

        return total_loss.item(), contrastive_loss.item(), geo_loss.item()

    def get_offset_patterns(self, dataset: PointCloudDataset, num_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract offset patterns learned by the model.

        Args:
            dataset: PointCloudDataset instance
            num_samples: number of samples to visualize

        Returns:
            points_batch: (num_samples, num_points, 3) batch of point clouds
            offsets_batch: (num_samples, num_points, 3) batch of learned offsets
        """
        self.encoder.eval()
        self.offset_branch.eval()

        points_list = []
        offsets_list = []

        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                cloud, _ = dataset.get_positive_pair(i)
                cloud_t = torch.from_numpy(cloud).unsqueeze(0).to(self.device)
                cloud_t = normalize_points(cloud_t)

                _, per_point_features = self.encoder(cloud_t)
                offsets = self.offset_branch(per_point_features)

                points_list.append(cloud_t.cpu())
                offsets_list.append(offsets.cpu())

        self.encoder.train()
        self.offset_branch.train()

        points_batch = torch.cat(points_list, dim=0)
        offsets_batch = torch.cat(offsets_list, dim=0)

        return points_batch, offsets_batch
