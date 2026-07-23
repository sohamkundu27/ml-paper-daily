import numpy as np
from single_image_diffusion import (
    PatchExtractor,
    ClosedFormDenoiser,
    DiffusionSampler,
    linear_noise_schedule,
    create_simple_image,
    add_gaussian_noise,
)


def test_end_to_end_generation():
    """End-to-end test: create reference, extract patches, build denoiser, generate samples."""
    print("\n=== Test: End-to-End Generation ===")

    # Step 1: Create reference image
    reference = create_simple_image(size=32)
    print(f"Reference image: mean={reference.mean():.3f}, std={reference.std():.3f}")

    # Step 2: Extract patches
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(reference)
    patches = extractor.get_patches()
    print(f"Extracted {len(patches)} patches of dimension {patches.shape[1]}")

    # Step 3: Compute statistics and build denoiser
    mean, cov = extractor.get_statistics()
    noise_level = 0.1
    denoiser = ClosedFormDenoiser(mean, cov, noise_level)
    print(f"Denoiser built with noise_level={noise_level}")

    # Step 4: Create sampler and generate
    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=30)
    generated = sampler.sample((32, 32))

    # Verify output
    assert generated.shape == (32, 32)
    assert np.all(np.isfinite(generated))
    assert np.all((generated >= 0) & (generated <= 1))
    print(f"Generated image: mean={generated.mean():.3f}, std={generated.std():.3f}")
    print("✓ End-to-end generation successful")


def test_multiple_generations():
    """Generate multiple variations from the same reference."""
    print("\n=== Test: Multiple Generations ===")

    # Create reference
    reference = create_simple_image(size=32)

    # Setup
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(reference)
    mean, cov = extractor.get_statistics()
    denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.1)
    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=20)

    # Generate multiple samples
    num_samples = 3
    samples = []
    for i in range(num_samples):
        sample = sampler.sample((32, 32))
        samples.append(sample)
        print(f"Sample {i+1}: mean={sample.mean():.3f}, std={sample.std():.3f}")

    # Verify diversity (samples should differ)
    diffs = []
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            diff = np.mean((samples[i] - samples[j]) ** 2)
            diffs.append(diff)

    mean_diff = np.mean(diffs)
    assert mean_diff > 1e-4, f"Samples too similar (MSE diff: {mean_diff})"
    print(f"Average pairwise MSE: {mean_diff:.6f}")
    print("✓ Multiple generations are diverse")


def test_application_stylization():
    """Demonstrate stylization: apply reference patch statistics to another image."""
    print("\n=== Test: Application - Stylization ===")

    # Create reference image (e.g., a specific texture)
    reference = create_simple_image(size=32)

    # Create a different target image to stylize
    y, x = np.meshgrid(np.linspace(0, 2*np.pi, 32), np.linspace(0, 2*np.pi, 32))
    target = 0.5 + 0.3 * np.sin(2*x + y)
    target = np.clip(target, 0, 1)

    print(f"Target image (before): mean={target.mean():.3f}, std={target.std():.3f}")

    # Extract patches from reference
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(reference)
    mean, cov = extractor.get_statistics()

    # Create denoiser from reference patches
    denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.15)
    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=25)

    # Add noise to target, then denoise with reference patch statistics
    noisy_target = add_gaussian_noise(target, noise_level=0.2)

    # Stylize by extracting patches from noisy target and denoising
    stylized = np.zeros_like(target)
    counts = np.zeros_like(target)

    for i in range(max(0, 32 - 5 + 1)):
        for j in range(max(0, 32 - 5 + 1)):
            patch = noisy_target[i:i+5, j:j+5].flatten()
            denoised_patch = denoiser.denoise_patch(patch)
            denoised_2d = denoised_patch.reshape(5, 5)
            stylized[i:i+5, j:j+5] += denoised_2d
            counts[i:i+5, j:j+5] += 1

    # Average overlaps
    mask = counts > 0
    stylized[mask] /= counts[mask]
    stylized = np.clip(stylized, 0, 1)

    print(f"Stylized image (after): mean={stylized.mean():.3f}, std={stylized.std():.3f}")

    # Verify stylization changed the image
    difference = np.mean((stylized - target) ** 2)
    assert difference > 1e-4, "Stylization did not modify the image"
    print(f"Image change (MSE): {difference:.6f}")
    print("✓ Stylization application successful")


def test_varying_noise_schedule():
    """Test generation with different noise schedules."""
    print("\n=== Test: Varying Noise Schedules ===")

    # Create reference
    reference = create_simple_image(size=32)
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(reference)
    mean, cov = extractor.get_statistics()
    denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.1)

    # Test different schedules
    schedules = [
        ("Short", 10, 0.5),
        ("Medium", 20, 1.0),
        ("Long", 30, 1.0),
    ]

    for name, steps, max_sigma in schedules:
        schedule = linear_noise_schedule(steps, min_sigma=0.01, max_sigma=max_sigma)
        sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=steps, noise_schedule=schedule)
        generated = sampler.sample((32, 32))

        assert generated.shape == (32, 32)
        assert np.all(np.isfinite(generated))
        print(f"{name} schedule ({steps} steps): mean={generated.mean():.3f}, std={generated.std():.3f}")

    print("✓ Multiple noise schedules work correctly")


def test_reference_comparison():
    """Compare generated samples to reference statistics."""
    print("\n=== Test: Reference vs. Generated Statistics ===")

    # Create reference
    reference = create_simple_image(size=32)
    ref_mean = reference.mean()
    ref_std = reference.std()

    # Extract and build sampler
    extractor = PatchExtractor(patch_size=5, scales=[1])
    extractor.extract(reference)
    mean, cov = extractor.get_statistics()
    denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.1)
    sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=25)

    # Generate multiple samples
    generated_samples = []
    for _ in range(5):
        gen = sampler.sample((32, 32))
        generated_samples.append(gen)

    gen_means = [g.mean() for g in generated_samples]
    gen_stds = [g.std() for g in generated_samples]

    avg_gen_mean = np.mean(gen_means)
    avg_gen_std = np.mean(gen_stds)

    print(f"Reference: mean={ref_mean:.3f}, std={ref_std:.3f}")
    print(f"Generated (avg): mean={avg_gen_mean:.3f}, std={avg_gen_std:.3f}")

    # They should be in the same ballpark (within reasonable margin)
    # Since we're generating from the patch distribution, means should be somewhat close
    print("✓ Statistics comparison complete")


def test_scalability():
    """Test with different image sizes."""
    print("\n=== Test: Scalability ===")

    for size in [16, 32]:
        reference = create_simple_image(size=size)
        extractor = PatchExtractor(patch_size=5, scales=[1])
        extractor.extract(reference)
        mean, cov = extractor.get_statistics()
        denoiser = ClosedFormDenoiser(mean, cov, noise_level=0.1)
        sampler = DiffusionSampler(denoiser, patch_size=5, num_steps=15)

        generated = sampler.sample((size, size))
        assert generated.shape == (size, size)
        assert np.all(np.isfinite(generated))
        print(f"Size {size}x{size}: Generated successfully (mean={generated.mean():.3f})")

    print("✓ Scalability test passed")


if __name__ == "__main__":
    test_end_to_end_generation()
    test_multiple_generations()
    test_application_stylization()
    test_varying_noise_schedule()
    test_reference_comparison()
    test_scalability()
    print("\n" + "="*50)
    print("All pass 4 tests passed! ✓")
    print("="*50)
