import numpy as np
import torch
import torch.nn.functional as F


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
    """Build multi-scale correlation pyramid."""

    def __init__(self, num_levels=4, search_range=4):
        self.num_levels = num_levels
        self.search_range = search_range
        self.corr_volume = CorrelationVolume(search_range)

    def extract_features(self, img):
        """Extract features (use normalized image as features)."""
        # Normalize to zero mean and unit variance per channel
        b, c, h, w = img.shape
        mean = img.mean(dim=(2, 3), keepdim=True)
        std = img.std(dim=(2, 3), keepdim=True) + 1e-8
        return (img - mean) / std

    def build(self, img1, img2):
        """Build correlation pyramid."""
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
    """Optical flow estimator using correlation pyramid."""

    def __init__(self, num_levels=4, search_range=4):
        self.num_levels = num_levels
        self.search_range = search_range
        self.pyramid = CorrelationPyramid(num_levels, search_range)

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


def estimate_optical_flow(img1, img2, num_levels=4, search_range=4):
    """Estimate optical flow between two images."""
    estimator = FlowEstimator(num_levels=num_levels, search_range=search_range)
    return estimator.estimate(img1, img2)
