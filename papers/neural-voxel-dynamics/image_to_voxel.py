import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraModel:
    """Simple pinhole camera model with perspective projection."""

    def __init__(self, fx, fy, cx, cy):
        """
        Args:
            fx, fy: focal length in x and y
            cx, cy: principal point (image center) in x and y
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def project_3d_to_2d(self, points_3d):
        """Project 3D points to 2D image coordinates.

        Args:
            points_3d: (B, N, 3) or (N, 3) in world/camera frame

        Returns:
            points_2d: (B, N, 2) or (N, 2) in image frame (u, v)
            depth: (B, N) or (N,) depth values
        """
        if len(points_3d.shape) == 2:
            points_3d = points_3d.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]

        # Perspective division
        u = (x / z) * self.fx + self.cx
        v = (y / z) * self.fy + self.cy

        points_2d = torch.stack([u, v], dim=-1)

        if squeeze_batch:
            points_2d = points_2d.squeeze(0)
            z = z.squeeze(0)

        return points_2d, z

    def unproject_2d_to_3d(self, points_2d, depth):
        """Lift 2D image points to 3D using depth.

        Args:
            points_2d: (B, N, 2) or (N, 2) in image frame (u, v)
            depth: (B, N) or (N,) depth values

        Returns:
            points_3d: (B, N, 3) or (N, 3) in world/camera frame
        """
        squeeze_batch = False
        if len(points_2d.shape) == 2:
            points_2d = points_2d.unsqueeze(0)
            squeeze_batch = True
            depth = depth.unsqueeze(0)

        u, v = points_2d[..., 0], points_2d[..., 1]

        # Inverse perspective projection
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        points_3d = torch.stack([x, y, z], dim=-1)

        if squeeze_batch:
            points_3d = points_3d.squeeze(0)

        return points_3d


class SimpleDepthEstimator(nn.Module):
    """Predicts depth map from RGB image using a simple CNN."""

    def __init__(self, image_channels=3, output_channels=1):
        """
        Args:
            image_channels: number of input channels (3 for RGB)
            output_channels: number of output channels (1 for single depth)
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.min_depth = 0.1
        self.max_depth = 10.0

    def forward(self, images):
        """
        Predict depth maps.

        Args:
            images: (B, 3, H, W) RGB images

        Returns:
            depth: (B, 1, H, W) depth maps in [min_depth, max_depth]
        """
        encoded = self.encoder(images)
        depth = self.decoder(encoded)
        depth = depth * (self.max_depth - self.min_depth) + self.min_depth
        return depth


class SimpleImageFeatureExtractor(nn.Module):
    """Extracts feature maps from RGB images using a simple CNN."""

    def __init__(self, image_channels=3, feature_channels=32):
        """
        Args:
            image_channels: number of input channels
            feature_channels: number of output feature channels
        """
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(image_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.feature_channels = feature_channels

    def forward(self, images):
        """
        Extract features.

        Args:
            images: (B, 3, H, W) RGB images

        Returns:
            features: (B, feature_channels, H, W) feature maps
        """
        return self.backbone(images)


class ImageToVoxelProjector(nn.Module):
    """Projects 2D image features to 3D voxel grid using depth."""

    def __init__(self, image_height, image_width, grid_size, feature_dim,
                 fx=400.0, fy=400.0, cx=None, cy=None,
                 voxel_bounds=None):
        """
        Args:
            image_height, image_width: image dimensions
            grid_size: size of voxel grid
            feature_dim: dimension of voxel features
            fx, fy: focal lengths (in pixels)
            cx, cy: principal point (default: image center)
            voxel_bounds: (min_depth, max_depth) bounds for voxel space (default: (0.1, 10.0))
        """
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.grid_size = grid_size
        self.feature_dim = feature_dim

        if cx is None:
            cx = image_width / 2.0
        if cy is None:
            cy = image_height / 2.0
        if voxel_bounds is None:
            voxel_bounds = (0.1, 10.0)

        self.camera = CameraModel(fx, fy, cx, cy)
        self.min_depth, self.max_depth = voxel_bounds

        # Feature extractor
        self.feature_extractor = SimpleImageFeatureExtractor(
            image_channels=3,
            feature_channels=feature_dim
        )

        # Depth estimator
        self.depth_estimator = SimpleDepthEstimator(image_channels=3, output_channels=1)

        # Voxel accumulation buffer (not learnable)
        self.register_buffer(
            'voxel_accum_count',
            torch.zeros(1, 1, grid_size, grid_size, grid_size)
        )

    def forward(self, images, depth_maps=None):
        """
        Lift 2D image features to 3D voxel grid.

        Args:
            images: (B, 3, H, W) RGB images
            depth_maps: (B, 1, H, W) depth maps (if None, estimated from images)

        Returns:
            voxel_features: (B, feature_dim, grid_size, grid_size, grid_size) accumulated voxel features
        """
        B = images.shape[0]
        device = images.device

        # Extract features from image
        image_features = self.feature_extractor(images)  # (B, feature_dim, H, W)

        # Estimate depth if not provided
        if depth_maps is None:
            depth_maps = self.depth_estimator(images)  # (B, 1, H, W)

        # Initialize voxel features accumulator
        voxel_features = torch.zeros(
            B, self.feature_dim, self.grid_size, self.grid_size, self.grid_size,
            device=device
        )
        voxel_count = torch.zeros(
            B, 1, self.grid_size, self.grid_size, self.grid_size,
            device=device
        )

        # Create pixel coordinate grid
        v_coords, u_coords = torch.meshgrid(
            torch.arange(self.image_height, dtype=torch.float32, device=device),
            torch.arange(self.image_width, dtype=torch.float32, device=device),
            indexing='ij'
        )

        for b in range(B):
            # Get features and depth for this batch
            feat = image_features[b]  # (feature_dim, H, W)
            depth = depth_maps[b, 0]  # (H, W)

            # Stack all pixels
            pixels_2d = torch.stack([u_coords, v_coords], dim=-1)  # (H, W, 2)
            pixels_2d = pixels_2d.reshape(-1, 2)  # (H*W, 2)
            depth_flat = depth.reshape(-1)  # (H*W,)
            feat_flat = feat.reshape(self.feature_dim, -1).T  # (H*W, feature_dim)

            # Unproject to 3D
            points_3d = self.camera.unproject_2d_to_3d(pixels_2d, depth_flat)  # (H*W, 3)

            # Map 3D points to voxel coordinates
            # Normalize by depth range and map to [0, grid_size-1]
            depth_norm = (depth_flat - self.min_depth) / (self.max_depth - self.min_depth)
            depth_norm = torch.clamp(depth_norm, 0, 1)

            # For xyz coordinates, normalize by camera coordinates range
            # Assume camera is at origin, looking down +Z, with range roughly [-5, 5] in X and Y
            x_norm = (points_3d[:, 0] / self.max_depth + 0.5)
            y_norm = (points_3d[:, 1] / self.max_depth + 0.5)
            z_norm = depth_norm

            # Clamp to valid range
            x_norm = torch.clamp(x_norm, 0, 1)
            y_norm = torch.clamp(y_norm, 0, 1)
            z_norm = torch.clamp(z_norm, 0, 1)

            # Convert to voxel indices
            voxel_x = (x_norm * (self.grid_size - 1)).long()
            voxel_y = (y_norm * (self.grid_size - 1)).long()
            voxel_z = (z_norm * (self.grid_size - 1)).long()

            # Accumulate features in voxels
            valid_mask = (voxel_x >= 0) & (voxel_x < self.grid_size) & \
                        (voxel_y >= 0) & (voxel_y < self.grid_size) & \
                        (voxel_z >= 0) & (voxel_z < self.grid_size)

            valid_x = voxel_x[valid_mask]
            valid_y = voxel_y[valid_mask]
            valid_z = voxel_z[valid_mask]
            valid_feat = feat_flat[valid_mask]  # (N_valid, feature_dim)

            # Use scatter_add to accumulate features
            for i in range(len(valid_x)):
                voxel_features[b, :, valid_z[i], valid_y[i], valid_x[i]] += valid_feat[i]
                voxel_count[b, 0, valid_z[i], valid_y[i], valid_x[i]] += 1.0

        # Normalize by count (avoid division by zero)
        voxel_count = torch.clamp(voxel_count, min=1.0)
        voxel_features = voxel_features / voxel_count

        return voxel_features


class VoxelToImageUnprojector(nn.Module):
    """Reprojects 3D voxel features back to 2D image space."""

    def __init__(self, image_height, image_width, grid_size, feature_dim,
                 fx=400.0, fy=400.0, cx=None, cy=None,
                 voxel_bounds=None):
        """
        Args:
            image_height, image_width: image dimensions
            grid_size: size of voxel grid
            feature_dim: dimension of voxel features
            fx, fy: focal lengths
            cx, cy: principal point
            voxel_bounds: (min_depth, max_depth) bounds
        """
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.grid_size = grid_size
        self.feature_dim = feature_dim

        if cx is None:
            cx = image_width / 2.0
        if cy is None:
            cy = image_height / 2.0
        if voxel_bounds is None:
            voxel_bounds = (0.1, 10.0)

        self.camera = CameraModel(fx, fy, cx, cy)
        self.min_depth, self.max_depth = voxel_bounds

    def forward(self, voxel_features):
        """
        Unproject voxel features back to 2D image.

        Args:
            voxel_features: (B, feature_dim, grid_size, grid_size, grid_size)

        Returns:
            image_features: (B, feature_dim, image_height, image_width) reprojected features
        """
        B, C, D, H, W = voxel_features.shape
        device = voxel_features.device

        # Initialize output
        image_features = torch.zeros(B, C, self.image_height, self.image_width, device=device)
        image_count = torch.zeros(B, 1, self.image_height, self.image_width, device=device)

        # Create voxel coordinate grids
        z_coords, y_coords, x_coords = torch.meshgrid(
            torch.arange(self.grid_size, dtype=torch.float32, device=device),
            torch.arange(self.grid_size, dtype=torch.float32, device=device),
            torch.arange(self.grid_size, dtype=torch.float32, device=device),
            indexing='ij'
        )

        for b in range(B):
            voxel_feat = voxel_features[b]  # (C, D, H, W)

            # Flatten voxel grid
            x_flat = x_coords.reshape(-1)
            y_flat = y_coords.reshape(-1)
            z_flat = z_coords.reshape(-1)
            feat_flat = voxel_feat.reshape(C, -1).T  # (D*H*W, C)

            # Convert voxel coords to normalized coordinates
            x_norm = x_flat / (self.grid_size - 1)
            y_norm = y_flat / (self.grid_size - 1)
            z_norm = z_flat / (self.grid_size - 1)

            # Convert back to 3D world coordinates
            # This is the inverse of the projection in ImageToVoxelProjector
            depth = z_norm * (self.max_depth - self.min_depth) + self.min_depth
            x_3d = (x_norm - 0.5) * self.max_depth
            y_3d = (y_norm - 0.5) * self.max_depth

            points_3d = torch.stack([x_3d, y_3d, depth], dim=-1)

            # Project to 2D
            points_2d, _ = self.camera.project_3d_to_2d(points_3d)
            u = points_2d[:, 0]
            v = points_2d[:, 1]

            # Check which points are within image bounds
            valid_mask = (u >= 0) & (u < self.image_width) & \
                        (v >= 0) & (v < self.image_height)

            valid_u = u[valid_mask].long()
            valid_v = v[valid_mask].long()
            valid_feat = feat_flat[valid_mask]  # (N_valid, C)

            # Accumulate features in image space
            for i in range(len(valid_u)):
                image_features[b, :, valid_v[i], valid_u[i]] += valid_feat[i]
                image_count[b, 0, valid_v[i], valid_u[i]] += 1.0

        # Normalize by count
        image_count = torch.clamp(image_count, min=1.0)
        image_features = image_features / image_count

        return image_features
