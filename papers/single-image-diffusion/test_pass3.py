import numpy as np
from single_image_diffusion import (
    PatchExtractor,
    ClosedFormDenoiser,
    DiffusionSampler,
    linear_noise_schedule,
    create_simple_image,
)


def test_noise_schedule():
    """Test that noise schedule is created correctly."""
    num_steps = 50
    schedule = linear_noise_schedule(num_steps, min_sigma=0.01, max_sigma=1.0)

    assert schedule.shape == (num_steps,), f"Expected shape {(num_steps,)}, got {schedule.shape}"
    assert schedule[0] == 1.0, "First noise level should be max_sigma"
    assert schedule[-1] == 0.01, "Last noise level should be min_sigma"
    assert np.all(np.diff(schedule) <= 0), "Noise schedule should be decreasing"
    print(f"✓ Noise schedule created correctly ({num_steps} steps)")


def test_sampler_initialization():
    """Test that sampler can be initialized."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    noise_level = 0.1

    denoiser = ClosedFormDenoiser(mean, cov, noise_level)
    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=10)

    assert sampler.num_steps == 10
    assert sampler.patch_size == 5
    assert len(sampler.noise_schedule) == 10
    print("✓ Sampler initialization works")


def test_patch_extraction_from_image():
    """Test that patches can be extracted from a noise image."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.1)
    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=5)

    # Generate a noisy image
    noisy_image = np.random.randn(32, 32)

    # Extract patches
    patches = sampler._extract_patches_from_image(noisy_image)

    assert len(patches) > 0, "Should extract at least one patch"
    assert patches[0].shape == (25,), "Each patch should be flattened to 25 elements"
    print(f"✓ Extracted {len(patches)} patches from noisy image")


def test_patch_reconstruction():
    """Test that image can be reconstructed from patches."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.1)
    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=5)

    # Create test patches
    patches = [np.random.randn(25) for _ in range(25)]  # Arbitrary number of patches

    # Reconstruct
    reconstructed = sampler._reconstruct_image_from_patches(patches, (32, 32))

    assert reconstructed.shape == (32, 32), f"Expected shape (32, 32), got {reconstructed.shape}"
    assert np.all(np.isfinite(reconstructed)), "Reconstructed image contains NaN or Inf"
    print(f"✓ Reconstructed image from patches (shape: {reconstructed.shape})")


def test_basic_generation():
    """Test that sampler can generate an image from noise."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.1)

    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=10)

    # Generate image
    generated = sampler.sample((32, 32))

    assert generated.shape == (32, 32), f"Expected shape (32, 32), got {generated.shape}"
    assert np.all(np.isfinite(generated)), "Generated image contains NaN or Inf"
    assert np.all((generated >= 0) & (generated <= 1)), "Generated image not in [0, 1]"
    print(f"✓ Generated image (mean: {generated.mean():.3f}, std: {generated.std():.3f})")


def test_generation_with_custom_schedule():
    """Test generation with custom noise schedule."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.1)

    # Create custom schedule (shorter, more aggressive)
    custom_schedule = linear_noise_schedule(5, min_sigma=0.05, max_sigma=0.5)
    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=5, noise_schedule=custom_schedule)

    generated = sampler.sample((32, 32))

    assert generated.shape == (32, 32)
    assert np.all(np.isfinite(generated))
    print(f"✓ Generation with custom schedule works")


def test_generation_with_initial_noise():
    """Test generation starting from specific initial noise."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.1)

    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=10)

    # Generate with specific initial noise
    initial_noise = np.random.randn(32, 32)
    generated = sampler.sample((32, 32), initial_noise=initial_noise)

    assert generated.shape == (32, 32)
    assert np.all(np.isfinite(generated))
    print(f"✓ Generation with initial noise works")


def test_multiple_generations():
    """Test that multiple generations produce different results."""
    image = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(image)

    mean, cov = extractor.get_statistics()
    denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.1)

    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=10)

    # Generate two images
    gen1 = sampler.sample((32, 32))
    gen2 = sampler.sample((32, 32))

    # They should be different (with high probability)
    diff = np.mean((gen1 - gen2) ** 2)
    assert diff > 1e-4, f"Generated images should be different, but MSE diff was {diff}"
    print(f"✓ Multiple generations produce different results (MSE diff: {diff:.6f})")


if __name__ == "__main__":
    test_noise_schedule()
    test_sampler_initialization()
    test_patch_extraction_from_image()
    test_patch_reconstruction()
    test_basic_generation()
    test_generation_with_custom_schedule()
    test_generation_with_initial_noise()
    test_multiple_generations()
    print("\nAll pass 3 tests passed! ✓")
