import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelGrid(nn.Module):
    """3D voxel grid that stores feature vectors at each grid position."""

    def __init__(self, grid_size, feature_dim):
        """
        Args:
            grid_size (int): size of the cubic voxel grid (grid_size x grid_size x grid_size)
            feature_dim (int): dimension of features stored at each voxel
        """
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        # Shape: (1, feature_dim, grid_size, grid_size, grid_size)
        self.features = nn.Parameter(torch.randn(1, feature_dim, grid_size, grid_size, grid_size) * 0.01)

    def forward(self, coords):
        """
        Sample features at continuous 3D coordinates using trilinear interpolation.

        Args:
            coords: (B, N, 3) tensor with coordinates in [0, grid_size-1]

        Returns:
            features: (B, N, feature_dim) sampled features
        """
        B, N, _ = coords.shape
        # Normalize coordinates to [-1, 1] for grid_sample
        coords_normalized = 2.0 * coords / (self.grid_size - 1) - 1.0

        # Reshape coordinates to (B, 1, 1, N, 3) for grid_sample
        # grid_sample expects (B, D, H, W, 3) for 5D input
        coords_normalized = coords_normalized.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 3)

        # Use grid_sample for trilinear interpolation
        # features: (B, C, D, H, W), grid: (B, D, H, W, 3)
        features_expanded = self.features.expand(B, -1, -1, -1, -1)
        sampled = F.grid_sample(
            features_expanded,
            coords_normalized,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        # sampled shape: (B, feature_dim, 1, 1, N)
        sampled = sampled.squeeze(2).squeeze(2).permute(0, 2, 1)  # (B, N, feature_dim)
        return sampled


class VelocityPredictor(nn.Module):
    """Simple neural network that predicts 3D velocity fields."""

    def __init__(self, grid_size, input_dim=0):
        """
        Args:
            grid_size (int): size of the cubic voxel grid
            input_dim (int): dimension of input features (0 for position-only)
        """
        super().__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim
        # Simple MLP: maps position (and optionally features) to 3D velocity
        total_input = 3 + input_dim
        self.net = nn.Sequential(
            nn.Linear(total_input, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, coords, features=None):
        """
        Predict 3D velocity at given coordinates.

        Args:
            coords: (B, N, 3) coordinates in [0, grid_size-1]
            features: (B, N, input_dim) optional input features

        Returns:
            velocity: (B, N, 3) predicted velocity vectors
        """
        B, N, _ = coords.shape
        # Normalize coordinates to [0, 1] for input
        coords_normalized = coords / (self.grid_size - 1)

        if features is not None:
            net_input = torch.cat([coords_normalized, features], dim=-1)
        else:
            net_input = coords_normalized

        # Reshape for MLP
        net_input = net_input.reshape(B * N, -1)
        velocity = self.net(net_input)
        velocity = velocity.reshape(B, N, 3)
        return velocity


class FeatureAdvection(nn.Module):
    """Core advection operator: moves features through voxel space based on velocity fields."""

    def __init__(self, grid_size, feature_dim, input_dim=0):
        """
        Args:
            grid_size (int): size of the cubic voxel grid
            feature_dim (int): dimension of features
            input_dim (int): dimension of input features for velocity prediction
        """
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        self.voxel_grid = VoxelGrid(grid_size, feature_dim)
        self.velocity_predictor = VelocityPredictor(grid_size, input_dim)

    def forward(self, coords, dt=0.1, features=None):
        """
        Advect features forward in time by one timestep.

        Args:
            coords: (B, N, 3) current coordinates
            dt: timestep size
            features: (B, N, feature_dim) optional input features for velocity prediction

        Returns:
            advected_features: (B, N, feature_dim) features at new locations
            new_coords: (B, N, 3) new coordinates after advection
        """
        # Predict velocity at current locations
        velocity = self.velocity_predictor(coords, features)

        # Semi-implicit Euler step: update coordinates
        new_coords = coords + dt * velocity

        # Clamp coordinates to valid range [0, grid_size-1]
        new_coords = torch.clamp(new_coords, 0, self.grid_size - 1)

        # Sample features at new locations
        advected_features = self.voxel_grid(new_coords)

        return advected_features, new_coords
