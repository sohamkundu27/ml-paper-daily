import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DepthEncoder:
    """Extract depth features using a lightweight depth estimation model."""

    def __init__(self, model_type="depth_anything_v2_small"):
        """
        Initialize depth encoder.

        Args:
            model_type: Type of depth model to use
        """
        self.model_type = model_type
        self.model = None
        self.device = None
        self._init_model()

    def _init_model(self):
        """Initialize the depth model."""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self.processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.model.eval()
        except ImportError:
            print("Warning: transformers not available, using fallback depth extraction")
            self.model = None
            self.processor = None

    def extract(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract depth features from an image.

        Args:
            img: (b, c, h, w) input image

        Returns:
            depth_feat: (b, 1, h, w) normalized depth map
        """
        if self.model is None:
            # Fallback: use simple edge-based depth proxy
            return self._fallback_depth(img)

        b, c, h, w = img.shape

        # Ensure model is on the same device as input
        if self.device != img.device:
            self.device = img.device
            self.model = self.model.to(self.device)

        with torch.no_grad():
            # Prepare input for the model (expects images in [0, 1] or [-1, 1] range)
            # Normalize to [0, 1] if needed
            img_normed = img.clone()
            if img_normed.min() < 0 or img_normed.max() > 1:
                img_normed = (img_normed - img_normed.min()) / (img_normed.max() - img_normed.min() + 1e-8)

            # Depth Anything V2 expects RGB input, process one at a time for safety
            depth_maps = []
            for i in range(b):
                # Get single image
                single_img = img_normed[i:i+1]

                # Resize to model input size if needed (typically 518x518 or similar)
                # For now, just use the image as-is and let the model handle it
                outputs = self.model(single_img)
                depth = outputs.predicted_depth

                # Resize depth to original resolution
                depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)
                depth_maps.append(depth)

            depth_feat = torch.cat(depth_maps, dim=0)

        # Normalize depth to [-1, 1]
        depth_mean = depth_feat.mean()
        depth_std = depth_feat.std() + 1e-8
        depth_feat = (depth_feat - depth_mean) / depth_std

        return depth_feat

    def _fallback_depth(self, img: torch.Tensor) -> torch.Tensor:
        """Fallback depth extraction using simple gradient-based proxy."""
        # Compute depth proxy as inverse of image gradients
        # This is a simplified fallback when transformers is not available

        # Compute image gradients
        if img.shape[1] > 1:
            img_gray = img.mean(dim=1, keepdim=True)
        else:
            img_gray = img

        # Compute gradients
        grad_x = torch.abs(img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1])
        grad_y = torch.abs(img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :])

        # Pad gradients back to original size
        grad_x = F.pad(grad_x, (0, 1))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        # Combine gradients
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Normalize
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

        # Invert (edges = low depth, smooth areas = high depth)
        depth_feat = 1.0 - grad_mag

        return depth_feat


class CorrelationVolume:
    """Compute correlation volume for optical flow estimation."""

    def __init__(self, search_range=4):
        self.search_range = search_range

    def compute(self, feat1, feat2):
        """
        Compute correlation volume between two feature maps.

        Args:
            feat1: (b, c, h, w) features from first image
            feat2: (b, c, h, w) features from second image

        Returns:
            corr: (b, (2*sr+1)^2, h, w) correlation volume
        """
        b, c, h, w = feat1.shape
        sr = self.search_range

        # Correlations for each displacement
        correlations = []

        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                # Determine valid regions for this displacement
                if dy >= 0:
                    y_s, y_e = dy, h
                    y2_s, y2_e = 0, h - dy
                else:
                    y_s, y_e = 0, h + dy
                    y2_s, y2_e = -dy, h

                if dx >= 0:
                    x_s, x_e = dx, w
                    x2_s, x2_e = 0, w - dx
                else:
                    x_s, x_e = 0, w + dx
                    x2_s, x2_e = -dx, w

                # Extract regions
                f1 = feat1[:, :, y_s:y_e, x_s:x_e]
                f2 = feat2[:, :, y2_s:y2_e, x2_s:x2_e]

                # Compute correlation (L2 distance, negated for similarity)
                if f1.shape[2] > 0 and f1.shape[3] > 0:
                    dist = torch.sum((f1 - f2) ** 2, dim=1)
                    corr_map = torch.zeros_like(feat1[:, 0:1, :, :])
                    corr_map[:, 0, y_s:y_e, x_s:x_e] = -dist
                else:
                    corr_map = torch.zeros_like(feat1[:, 0:1, :, :])

                correlations.append(corr_map)

        return torch.cat(correlations, dim=1)


class CorrelationPyramid:
    """Build multi-scale correlation pyramid with depth-guided features."""

    def __init__(self, num_levels=4, search_range=4, use_depth=True):
        self.num_levels = num_levels
        self.search_range = search_range
        self.corr_volume = CorrelationVolume(search_range)
        self.use_depth = use_depth
        self.depth_encoder = DepthEncoder() if use_depth else None

    def extract_features(self, img):
        """Extract features and optionally incorporate depth priors."""
        # Normalize RGB features to zero mean and unit variance per channel
        b, c, h, w = img.shape
        mean = img.mean(dim=(2, 3), keepdim=True)
        std = img.std(dim=(2, 3), keepdim=True) + 1e-8
        rgb_feat = (img - mean) / std

        if not self.use_depth or self.depth_encoder is None:
            return rgb_feat

        # Extract depth features
        depth_feat = self.depth_encoder.extract(img)

        # Concatenate depth with RGB features
        # This allows the correlation computation to leverage both appearance and depth
        combined_feat = torch.cat([rgb_feat, depth_feat], dim=1)

        return combined_feat

    def build(self, img1, img2):
        """Build correlation pyramid with depth-guided features."""
        pyramid = []

        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)

        for level in range(self.num_levels):
            if level > 0:
                feat1 = F.avg_pool2d(feat1, kernel_size=2, stride=2)
                feat2 = F.avg_pool2d(feat2, kernel_size=2, stride=2)

            corr = self.corr_volume.compute(feat1, feat2)
            pyramid.append(corr)

        return pyramid


class FlowEstimator:
    """Optical flow estimator using depth-guided correlation pyramid."""

    def __init__(self, num_levels=4, search_range=4, use_depth=True):
        self.num_levels = num_levels
        self.search_range = search_range
        self.use_depth = use_depth
        self.pyramid = CorrelationPyramid(num_levels, search_range, use_depth=use_depth)

    def estimate(self, img1, img2):
        """
        Estimate optical flow between two images.

        Args:
            img1, img2: Input images (numpy or tensor)

        Returns:
            flow: (b, 2, h, w) optical flow field
        """
        # Convert to tensor if needed
        if isinstance(img1, np.ndarray):
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()

        # Handle different input formats
        if img1.ndim == 2:
            img1 = img1.unsqueeze(0).unsqueeze(0)
            img2 = img2.unsqueeze(0).unsqueeze(0)
        elif img1.ndim == 3:
            if img1.shape[0] in [1, 3]:  # (c, h, w)
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
            else:  # (h, w, c)
                img1 = img1.permute(2, 0, 1).unsqueeze(0)
                img2 = img2.permute(2, 0, 1).unsqueeze(0)
        elif img1.ndim == 4:
            if img1.shape[1] > img1.shape[3]:  # (b, h, w, c)
                img1 = img1.permute(0, 3, 1, 2)
                img2 = img2.permute(0, 3, 1, 2)

        b, c, h, w = img1.shape

        # Build pyramid
        pyr = self.pyramid.build(img1, img2)

        # Get coarsest level
        corr = pyr[-1]
        b_c, sz, h_c, w_c = corr.shape

        # Reshape correlation for argmax
        sr = self.search_range
        corr_reshaped = corr.view(b_c, 2 * sr + 1, 2 * sr + 1, h_c, w_c)

        # Find best match in each search window (simplified: use argmax)
        flat_corr = corr_reshaped.view(b_c, sz, h_c, w_c)
        _, best_idx = torch.max(flat_corr, dim=1)

        # Convert indices to (dx, dy)
        best_idx = best_idx.float()
        u = (best_idx % (2 * sr + 1)) - sr
        v = (best_idx // (2 * sr + 1)) - sr

        # Stack into flow
        flow = torch.stack([u, v], dim=1)

        # Upsample to original resolution
        if self.num_levels > 1:
            scale = 2 ** (self.num_levels - 1)
            flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
            flow = flow * scale

        return flow


def estimate_optical_flow(img1, img2, num_levels=4, search_range=4, use_depth=True):
    """
    Estimate optical flow between two images with optional depth guidance.

    Args:
        img1, img2: Input images (numpy or tensor)
        num_levels: Number of pyramid levels
        search_range: Search range for correlation
        use_depth: Whether to use depth priors (Depth Anything V2)

    Returns:
        flow: (b, 2, h, w) optical flow field
    """
    estimator = FlowEstimator(num_levels=num_levels, search_range=search_range, use_depth=use_depth)
    return estimator.estimate(img1, img2)
