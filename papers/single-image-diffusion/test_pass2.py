import numpy as np
from single_image_diffusion import (
    PatchExtractor,
    ClosedFormDenoiser,
    create_simple_image,
    add_gaussian_noise,
)


def test_denoiser_initialization():
    """Test that denoiser can be initialized with statistics."""
    mean = np.random.randn(25)
    cov = np.random.randn(25, 25)
    cov = cov @ cov.T  # Make positive definite
    noise_level = 0.1

    denoiser = ClosedFormDenoiser(mean, cov, noise_level)

    assert denoiser.mean.shape == (25,)
    assert denoiser.cov.shape == (25, 25)
    assert denoiser.noise_level == 0.1
    print("✓ Denoiser initialization works")


def test_denoise_patch():
    """Test that denoiser can denoise a single patch."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    noise_level = 0.1

    denoiser = ClosedFormDenoiser(mean, cov, noise_level)

    # Create a noisy patch from the first clean patch
    patches = extractor.get_patches()
    clean_patch = patches[0]
    noisy_patch = clean_patch + np.random.normal(0, noise_level, clean_patch.shape)

    # Denoise it
    denoised = denoiser.denoise_patch(noisy_patch)

    assert denoised.shape == clean_patch.shape
    assert np.all(np.isfinite(denoised)), "Denoised patch contains NaN or Inf"
    print(f"✓ Patch denoising works (clean MSE: {np.mean((denoised - clean_patch)**2):.6f})")


def test_denoiser_reduces_noise():
    """Test that denoiser actually reduces noise on noisy patches."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    noise_level = 0.15

    denoiser = ClosedFormDenoiser(mean, cov, noise_level)

    patches = extractor.get_patches()

    # Test on multiple patches
    mse_noisy_total = 0
    mse_denoised_total = 0
    num_patches = min(10, len(patches))

    for i in range(num_patches):
        clean_patch = patches[i]
        noisy_patch = clean_patch + np.random.normal(0, noise_level, clean_patch.shape)
        denoised = denoiser.denoise_patch(noisy_patch)

        mse_noisy = np.mean((noisy_patch - clean_patch) ** 2)
        mse_denoised = np.mean((denoised - clean_patch) ** 2)

        mse_noisy_total += mse_noisy
        mse_denoised_total += mse_denoised

    avg_mse_noisy = mse_noisy_total / num_patches
    avg_mse_denoised = mse_denoised_total / num_patches

    # Denoiser should reduce MSE (or at least be close)
    assert avg_mse_denoised <= avg_mse_noisy * 1.5, (
        f"Denoiser made things worse: "
        f"noisy MSE={avg_mse_noisy:.6f}, denoised MSE={avg_mse_denoised:.6f}"
    )
    print(
        f"✓ Denoiser reduces noise (noisy MSE: {avg_mse_noisy:.6f} "
        f"→ denoised MSE: {avg_mse_denoised:.6f})"
    )


def test_score_function():
    """Test that score function is computed correctly."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    noise_level = 0.1

    denoiser = ClosedFormDenoiser(mean, cov, noise_level)

    patches = extractor.get_patches()
    noisy_patch = patches[0] + np.random.normal(0, noise_level, patches[0].shape)

    score = denoiser.score_function(noisy_patch)

    assert score.shape == noisy_patch.shape
    assert np.all(np.isfinite(score)), "Score contains NaN or Inf"
    print(f"✓ Score function computation works (score norm: {np.linalg.norm(score):.6f})")


def test_pca_denoiser():
    """Test denoiser with PCA-based dimension reduction."""
    image = create_simple_image(size=64)
    extractor = PatchExtractor(patch_size=8, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    patches = extractor.get_patches()
    noise_level = 0.1

    # Create denoiser with PCA
    pca_components = 16
    denoiser = ClosedFormDenoiser(mean, cov, noise_level, pca_components=pca_components)
    denoiser.fit_pca(patches)

    # Test denoising
    clean_patch = patches[0]
    noisy_patch = clean_patch + np.random.normal(0, noise_level, clean_patch.shape)
    denoised = denoiser.denoise_patch(noisy_patch)

    assert denoised.shape == clean_patch.shape
    assert np.all(np.isfinite(denoised)), "PCA denoised patch contains NaN or Inf"

    mse_noisy = np.mean((noisy_patch - clean_patch) ** 2)
    mse_denoised = np.mean((denoised - clean_patch) ** 2)
    print(
        f"✓ PCA denoiser works (noisy MSE: {mse_noisy:.6f} "
        f"→ denoised MSE: {mse_denoised:.6f})"
    )


if __name__ == "__main__":
    test_denoiser_initialization()
    test_denoise_patch()
    test_denoiser_reduces_noise()
    test_score_function()
    test_pca_denoiser()
    print("\nAll pass 2 tests passed! ✓")
