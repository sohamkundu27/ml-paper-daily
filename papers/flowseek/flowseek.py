import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


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


class MotionBasis:
    """Parameterize optical flow using low-dimensional motion models."""

    def __init__(self, motion_type="affine"):
        """
        Initialize motion basis.

        Args:
            motion_type: Type of motion model ("translation", "affine", or "homography")
        """
        self.motion_type = motion_type
        self.params = None

        # Parameter counts for each motion type
        self.param_counts = {
            "translation": 2,      # u0, v0
            "affine": 6,          # a0, a1, a2, b0, b1, b2
            "homography": 8        # 8 parameters for projective transform
        }

    def fit(self, flow: torch.Tensor, weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Fit motion basis parameters to optical flow field.

        Args:
            flow: (b, 2, h, w) optical flow field
            weights: (b, 1, h, w) per-pixel confidence weights (optional)

        Returns:
            params: Dictionary of fitted parameters
        """
        b, c, h, w = flow.shape

        # Create coordinate grid
        yy, xx = torch.meshgrid(torch.arange(h, dtype=flow.dtype, device=flow.device),
                                 torch.arange(w, dtype=flow.dtype, device=flow.device),
                                 indexing='ij')

        if self.motion_type == "translation":
            return self._fit_translation(flow, weights)
        elif self.motion_type == "affine":
            return self._fit_affine(flow, xx, yy, weights)
        elif self.motion_type == "homography":
            return self._fit_homography(flow, xx, yy, weights)
        else:
            raise ValueError(f"Unknown motion type: {self.motion_type}")

    def _fit_translation(self, flow: torch.Tensor,
                        weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Fit simple translation model: u=u0, v=v0."""
        # Average flow across image, weighted by confidence
        if weights is not None:
            weights_norm = weights / (weights.sum() + 1e-8)
            u0 = (flow[:, 0:1] * weights_norm).sum()
            v0 = (flow[:, 1:2] * weights_norm).sum()
        else:
            u0 = flow[:, 0].mean()
            v0 = flow[:, 1].mean()

        self.params = {"u0": float(u0), "v0": float(v0)}
        return self.params

    def _fit_affine(self, flow: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor,
                   weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Fit affine motion model: u = a0 + a1*x + a2*y, v = b0 + b1*x + b2*y."""
        b, c, h, w = flow.shape

        # Build design matrix [1, x, y] for each pixel
        ones = torch.ones_like(xx)

        # Flatten coordinates
        X = torch.stack([ones, xx, yy], dim=0).reshape(3, -1)  # (3, h*w)
        u = flow[0, 0].reshape(-1)  # (h*w,)
        v = flow[0, 1].reshape(-1)  # (h*w,)

        # Optionally apply weights
        if weights is not None:
            w = weights[0, 0].reshape(-1)
            w_sqrt = torch.sqrt(w + 1e-8)
            X = X * w_sqrt.unsqueeze(0)
            u = u * w_sqrt
            v = v * w_sqrt

        # Solve least squares: X^T * params = flow
        # params = (X * X^T)^-1 * X * flow
        XXT = X @ X.T
        Xu = X @ u
        Xv = X @ v

        try:
            params_u = torch.linalg.solve(XXT, Xu)
            params_v = torch.linalg.solve(XXT, Xv)
        except:
            # Fall back to pseudo-inverse if solve fails
            XXT_inv = torch.linalg.pinv(XXT)
            params_u = XXT_inv @ Xu
            params_v = XXT_inv @ Xv

        self.params = {
            "a0": float(params_u[0]), "a1": float(params_u[1]), "a2": float(params_u[2]),
            "b0": float(params_v[0]), "b1": float(params_v[1]), "b2": float(params_v[2])
        }
        return self.params

    def _fit_homography(self, flow: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor,
                       weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Fit homography motion model (simplified: use affine as approximation)."""
        # Homography fitting is complex; for pass 3, approximate with affine
        return self._fit_affine(flow, xx, yy, weights)

    def compute_motion_field(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Compute synthetic motion field from fitted parameters.

        Args:
            h, w: Height and width of output field
            device: Device to create tensor on

        Returns:
            flow: (1, 2, h, w) synthetic flow field from motion model
        """
        if self.params is None:
            raise ValueError("Must fit parameters first")

        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device),
                                 torch.arange(w, dtype=torch.float32, device=device),
                                 indexing='ij')

        if self.motion_type == "translation":
            u = torch.full_like(xx, self.params["u0"])
            v = torch.full_like(yy, self.params["v0"])
        elif self.motion_type == "affine":
            u = (self.params["a0"] + self.params["a1"] * xx + self.params["a2"] * yy)
            v = (self.params["b0"] + self.params["b1"] * xx + self.params["b2"] * yy)
        elif self.motion_type == "homography":
            # Approximate homography with affine
            u = (self.params["a0"] + self.params["a1"] * xx + self.params["a2"] * yy)
            v = (self.params["b0"] + self.params["b1"] * xx + self.params["b2"] * yy)
        else:
            raise ValueError(f"Unknown motion type: {self.motion_type}")

        flow = torch.stack([u, v], dim=0).unsqueeze(0)
        return flow

    def regularize_flow(self, flow: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
        """
        Regularize optical flow by blending with motion basis prediction.

        Args:
            flow: (b, 2, h, w) estimated optical flow
            strength: Blending factor (0=no regularization, 1=full motion basis)

        Returns:
            regularized_flow: (b, 2, h, w) blended flow
        """
        b, c, h, w = flow.shape

        # Compute motion field from fitted parameters
        motion_field = self.compute_motion_field(h, w, flow.device)

        # Blend estimated flow with motion field
        regularized = flow * (1 - strength) + motion_field * strength
        return regularized


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
    """Optical flow estimator with motion basis regularization."""

    def __init__(self, num_levels=4, search_range=4, use_depth=True,
                 motion_basis_type=None, motion_basis_strength=0.5):
        self.num_levels = num_levels
        self.search_range = search_range
        self.use_depth = use_depth
        self.pyramid = CorrelationPyramid(num_levels, search_range, use_depth=use_depth)
        self.motion_basis_type = motion_basis_type
        self.motion_basis_strength = motion_basis_strength
        self.motion_basis = MotionBasis(motion_basis_type) if motion_basis_type else None

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

        # Apply motion basis regularization if enabled
        if self.motion_basis is not None:
            # Fit motion basis to the estimated flow
            self.motion_basis.fit(flow)
            # Regularize flow by blending with motion basis
            flow = self.motion_basis.regularize_flow(flow, self.motion_basis_strength)

        return flow


def estimate_optical_flow(img1, img2, num_levels=4, search_range=4, use_depth=True,
                         motion_basis_type=None, motion_basis_strength=0.5):
    """
    Estimate optical flow between two images with optional depth guidance and motion regularization.

    Args:
        img1, img2: Input images (numpy or tensor)
        num_levels: Number of pyramid levels
        search_range: Search range for correlation
        use_depth: Whether to use depth priors (Depth Anything V2)
        motion_basis_type: Motion model type ("translation", "affine", "homography") or None
        motion_basis_strength: Strength of motion basis regularization (0-1)

    Returns:
        flow: (b, 2, h, w) optical flow field
    """
    estimator = FlowEstimator(num_levels=num_levels, search_range=search_range, use_depth=use_depth,
                             motion_basis_type=motion_basis_type, motion_basis_strength=motion_basis_strength)
    return estimator.estimate(img1, img2)
