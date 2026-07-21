import numpy as np
from single_image_diffusion import PatchExtractor, create_simple_image, add_gaussian_noise


def test_patch_extraction():
    """Test that patches can be extracted from an image."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1, 2])

    extractor.extract(image)
    patches = extractor.get_patches()

    # Should have extracted patches from both scales
    assert len(patches) > 0, "No patches extracted"
    assert patches.shape[1] == 5 * 5, f"Expected patch dim {5*5}, got {patches.shape[1]}"
    print(f"✓ Extracted {len(patches)} patches")


def test_patch_statistics():
    """Test that statistics can be computed from patches."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])

    extractor.extract(image)
    mean, cov = extractor.get_statistics()

    # Mean should be a vector of size patch_size^2
    assert mean.shape == (25,), f"Expected mean shape (25,), got {mean.shape}"

    # Covariance should be a square matrix
    assert cov.shape == (25, 25), f"Expected cov shape (25, 25), got {cov.shape}"

    # Covariance should be symmetric (up to numerical precision)
    assert np.allclose(cov, cov.T), "Covariance matrix not symmetric"

    # Mean values should be in reasonable range (image is in [0, 1])
    assert np.all(mean >= -0.5) and np.all(mean <= 1.5), "Mean values out of expected range"
    print(f"✓ Computed statistics: mean ∈ [{mean.min():.3f}, {mean.max():.3f}]")


def test_nearest_neighbor():
    """Test that nearest neighbor lookup works."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])

    extractor.extract(image)
    patches = extractor.get_patches()

    # Query with first patch (should find itself)
    query = patches[0]
    distances, indices = extractor.nearest_neighbor(query, k=3)

    # First result should have distance ~0 (the patch itself)
    assert distances[0] < 1e-6, f"First NN distance should be ~0, got {distances[0]}"
    assert indices[0] == 0, f"First NN should be patch 0, got {indices[0]}"

    # Results should be sorted by distance
    assert np.all(np.diff(distances) >= 0), "Nearest neighbors not sorted by distance"
    print(f"✓ Nearest neighbor search works (query distances: {distances})")


def test_multiple_scales():
    """Test that multiple scales are handled correctly."""
    image = create_simple_image(size=64)
    extractor = PatchExtractor(patch_size=5, scales=[1, 2, 4])

    extractor.extract(image)
    patches = extractor.get_patches()

    # Should have patches from all scales
    assert len(extractor.scales_used) == len(patches), "Scale tracking mismatch"

    # Count patches per scale
    scale_counts = {}
    for scale in extractor.scales_used:
        scale_counts[scale] = scale_counts.get(scale, 0) + 1

    # Smaller scales should have more patches
    assert scale_counts[1] > scale_counts[2], "Scale 1 should have more patches than scale 2"
    print(f"✓ Multiple scales work: {scale_counts}")


def test_noisy_patches():
    """Test patch extraction on noisy image."""
    image = create_simple_image(size=32)
    noisy_image = add_gaussian_noise(image, noise_level=0.2)

    extractor_clean = PatchExtractor(patch_size=5, scales=[1])
    extractor_clean.extract(image)

    extractor_noisy = PatchExtractor(patch_size=5, scales=[1])
    extractor_noisy.extract(noisy_image)

    # Both should extract same number of patches
    assert len(extractor_clean.get_patches()) == len(extractor_noisy.get_patches())

    # Noisy patches should have different statistics
    mean_clean, cov_clean = extractor_clean.get_statistics()
    mean_noisy, cov_noisy = extractor_noisy.get_statistics()

    # Variance should increase with noise (trace of covariance grows)
    trace_clean = np.trace(cov_clean)
    trace_noisy = np.trace(cov_noisy)
    assert trace_noisy > trace_clean, "Noisy patches should have higher variance"
    print(f"✓ Handles noisy images correctly (variance: {trace_clean:.3f} → {trace_noisy:.3f})")


if __name__ == "__main__":
    test_patch_extraction()
    test_patch_statistics()
    test_nearest_neighbor()
    test_multiple_scales()
    test_noisy_patches()
    print("\nAll tests passed! ✓")
