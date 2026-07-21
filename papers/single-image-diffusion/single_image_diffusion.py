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


class SimplePCA:
    """Simple PCA implementation using SVD."""

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        """Fit PCA on data X.

        Args:
            X: array of shape (n_samples, n_features)
        """
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[: self.n_components]
        self.explained_variance = (S ** 2) / (X.shape[0] - 1)

        return self

    def transform(self, X):
        """Project X onto PCA components.

        Args:
            X: array of shape (n_samples, n_features) or (n_features,)

        Returns:
            X_transformed: projected data
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def inverse_transform(self, X_transformed):
        """Reconstruct from PCA space.

        Args:
            X_transformed: array of shape (n_samples, n_components) or (n_components,)

        Returns:
            X_reconstructed: reconstructed data in original space
        """
        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(1, -1)
        return X_transformed @ self.components + self.mean


class ClosedFormDenoiser:
    """Closed-form denoiser using patch statistics (Wiener filter)."""

    def __init__(self, mean, cov, noise_level, pca_components=None):
        """
        Args:
            mean: mean patch vector (shape: patch_dim,)
            cov: covariance matrix (shape: patch_dim, patch_dim)
            noise_level: noise standard deviation
            pca_components: if not None, use PCA for dimension reduction
        """
        self.mean = mean
        self.cov = cov
        self.noise_level = noise_level
        self.pca_components = pca_components
        self.patch_dim = mean.shape[0]

        # Setup PCA if dimension reduction is requested
        self.pca = None
        self.reduced_mean = None
        self.reduced_cov = None

        if pca_components is not None and pca_components < self.patch_dim:
            # For training PCA, we need the original patch data
            # This will be set in fit_pca() method
            pass
        else:
            # Use full covariance
            self.reduced_mean = mean.copy()
            self.reduced_cov = cov.copy()

    def fit_pca(self, patch_data):
        """Fit PCA on patch data for dimension reduction.

        Args:
            patch_data: array of shape (num_patches, patch_dim)
        """
        if self.pca_components is None or self.pca_components >= self.patch_dim:
            return

        # Fit PCA
        self.pca = SimplePCA(self.pca_components)
        self.pca.fit(patch_data)

        # Transform statistics to PCA space
        mean_reduced = self.pca.transform(self.mean.reshape(1, -1))[0]
        patch_data_reduced = self.pca.transform(patch_data)
        cov_reduced = np.cov(patch_data_reduced.T)

        self.reduced_mean = mean_reduced
        self.reduced_cov = cov_reduced

    def denoise_patch(self, noisy_patch):
        """Denoise a single patch using Wiener filter.

        Args:
            noisy_patch: noisy patch vector (shape: patch_dim,)

        Returns:
            denoised_patch: denoised patch vector
        """
        # Use PCA-reduced space if available, otherwise full space
        if self.pca is not None:
            noisy_reduced = self.pca.transform(noisy_patch.reshape(1, -1))[0]
            mean_to_use = self.reduced_mean
            cov_to_use = self.reduced_cov
        else:
            noisy_reduced = noisy_patch
            mean_to_use = self.mean
            cov_to_use = self.cov

        sigma_sq = self.noise_level ** 2

        # Wiener filter: denoised = mean + cov @ (cov + sigma^2*I)^{-1} @ (noisy - mean)
        posterior_cov_inv = np.linalg.inv(cov_to_use + sigma_sq * np.eye(mean_to_use.shape[0]))
        denoised_reduced = (
            mean_to_use
            + cov_to_use @ posterior_cov_inv @ (noisy_reduced - mean_to_use)
        )

        # Transform back to original space if using PCA
        if self.pca is not None:
            denoised = self.pca.inverse_transform(denoised_reduced.reshape(1, -1))[0]
        else:
            denoised = denoised_reduced

        return denoised

    def score_function(self, noisy_patch):
        """Compute score function ∇_x log p(x | noisy_patch).

        Args:
            noisy_patch: noisy patch vector (shape: patch_dim,)

        Returns:
            score: score vector (shape: patch_dim,)
        """
        sigma_sq = self.noise_level ** 2

        # Denoise the patch
        denoised = self.denoise_patch(noisy_patch)

        # Score is (denoised - noisy) / sigma^2
        score = (denoised - noisy_patch) / sigma_sq

        return score


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
