import numpy as np
from scipy.ndimage import gaussian_filter


class PatchExtractor:
    """Extract patches at multiple scales from a single image."""

    def __init__(self, patch_size=5, scales=None):
        """
        Args:
            patch_size: size of patches to extract (assumed square)
            scales: list of scales (downsample factors); defaults to [1, 2]
        """
        self.patch_size = patch_size
        self.scales = scales if scales is not None else [1, 2]
        self.patches = []
        self.scales_used = []

    def extract(self, image):
        """Extract patches from image at multiple scales.

        Args:
            image: numpy array of shape (H, W) or (H, W, C)

        Returns:
            None (patches stored internally)
        """
        if image.ndim == 3:
            # Convert RGB to grayscale for simplicity
            image = np.mean(image, axis=2)

        self.patches = []
        self.scales_used = []

        for scale in self.scales:
            # Downsample by scale factor
            h, w = image.shape
            downsampled = image[::scale, ::scale]

            # Extract patches using sliding window
            patches_at_scale = []
            for i in range(downsampled.shape[0] - self.patch_size + 1):
                for j in range(downsampled.shape[1] - self.patch_size + 1):
                    patch = downsampled[i:i+self.patch_size, j:j+self.patch_size].copy()
                    patches_at_scale.append(patch.flatten())

            if patches_at_scale:
                self.patches.extend(patches_at_scale)
                self.scales_used.extend([scale] * len(patches_at_scale))

        self.patches = np.array(self.patches)  # Shape: (num_patches, patch_size^2)

    def get_statistics(self):
        """Compute mean and covariance of patch distribution.

        Returns:
            mean: mean patch vector
            cov: covariance matrix
        """
        if len(self.patches) == 0:
            raise ValueError("No patches extracted yet. Call extract() first.")

        mean = np.mean(self.patches, axis=0)
        cov = np.cov(self.patches.T)

        return mean, cov

    def get_patches(self):
        """Return all extracted patches."""
        return self.patches

    def nearest_neighbor(self, query, k=1):
        """Find k nearest neighbor patches to a query patch.

        Args:
            query: patch vector (flattened)
            k: number of nearest neighbors

        Returns:
            distances: distances to k nearest neighbors
            indices: indices of k nearest neighbors
        """
        if len(self.patches) == 0:
            raise ValueError("No patches extracted yet. Call extract() first.")

        # Compute L2 distances
        distances = np.linalg.norm(self.patches - query, axis=1)

        # Get k smallest distances
        indices = np.argsort(distances)[:k]
        return distances[indices], indices


def add_gaussian_noise(image, noise_level):
    """Add Gaussian noise to an image.

    Args:
        image: input image (numpy array)
        noise_level: standard deviation of Gaussian noise

    Returns:
        noisy_image: image with added noise
    """
    noise = np.random.normal(0, noise_level, image.shape)
    return np.clip(image + noise, 0, 1)


def create_simple_image(size=32):
    """Create a simple synthetic image for testing.

    Args:
        size: height/width of image

    Returns:
        image: normalized grayscale image in [0, 1]
    """
    # Create a simple image with geometric patterns
    y, x = np.meshgrid(np.linspace(0, 4*np.pi, size), np.linspace(0, 4*np.pi, size))
    image = 0.5 + 0.5 * np.sin(x) * np.cos(y)

    # Add a gradient
    gradient = np.linspace(0, 1, size)
    image = image * (0.5 + 0.5 * gradient[:, np.newaxis])

    return np.clip(image, 0, 1)
