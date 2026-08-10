"""Test the end-to-end BitDance pipeline."""

import torch
from end_to_end_demo import EndToEndDemo


def test_toy_image_generation():
    """Test that toy images can be created."""
    demo = EndToEndDemo(device="cpu")
    images = demo.create_toy_images(batch_size=4, height=32, width=32)

    assert images.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {images.shape}"
    assert images.min() >= 0 and images.max() <= 1, f"Images should be in [0, 1], got [{images.min()}, {images.max()}]"

    print("✓ Toy image generation test passed")


def test_encode_decode_cycle():
    """Test encode-decode cycle on toy images."""
    demo = EndToEndDemo(device="cpu")
    images = demo.create_toy_images(batch_size=2, height=32, width=32)

    # Encode
    tokens, binary_tokens = demo.encode_images(images)

    # Check shapes
    assert tokens.shape == (2, 32, 4, 4), f"Expected (2, 32, 4, 4), got {tokens.shape}"
    assert binary_tokens.shape == (2, 32, 4, 4), f"Expected (2, 32, 4, 4), got {binary_tokens.shape}"

    # Check binary tokens are actually binary
    assert torch.all((binary_tokens == 0) | (binary_tokens == 1)), "Binary tokens should be 0 or 1"

    # Decode
    reconstructed = demo.decode_tokens(tokens)

    # Check shape
    assert reconstructed.shape == (2, 3, 32, 32), f"Expected (2, 3, 32, 32), got {reconstructed.shape}"

    # Check range
    assert reconstructed.min() >= 0 and reconstructed.max() <= 1, (
        f"Reconstructed should be in [0, 1], got [{reconstructed.min()}, {reconstructed.max()}]"
    )

    print("✓ Encode-decode cycle test passed")


def test_token_masking():
    """Test that token masking works correctly."""
    demo = EndToEndDemo(device="cpu")
    images = demo.create_toy_images(batch_size=2, height=32, width=32)
    tokens, binary_tokens = demo.encode_images(images)

    # Mask 50% of tokens
    masked_tokens, mask = demo.mask_tokens(binary_tokens, mask_fraction=0.5)

    # Check that masked positions are zero
    assert torch.all(masked_tokens[mask] == 0), "Masked positions should be zero"

    # Check that unmasked positions are preserved
    unmasked_positions = ~mask
    if unmasked_positions.any():
        assert torch.all(masked_tokens[unmasked_positions] == binary_tokens[unmasked_positions]), (
            "Unmasked positions should be preserved"
        )

    print("✓ Token masking test passed")


def test_token_regeneration():
    """Test that masked tokens can be regenerated."""
    demo = EndToEndDemo(device="cpu")
    images = demo.create_toy_images(batch_size=2, height=32, width=32)
    tokens, binary_tokens = demo.encode_images(images)

    # Mask some tokens
    masked_tokens, mask = demo.mask_tokens(binary_tokens, mask_fraction=0.5)

    # Regenerate
    regenerated = demo.regenerate_masked_tokens(masked_tokens, num_diffusion_steps=5)

    # Check shape
    assert regenerated.shape == binary_tokens.shape, (
        f"Expected shape {binary_tokens.shape}, got {regenerated.shape}"
    )

    # Check binary
    assert torch.all((regenerated == 0) | (regenerated == 1)), "Regenerated tokens should be binary"

    print("✓ Token regeneration test passed")


def test_image_completion():
    """Test end-to-end image completion pipeline."""
    demo = EndToEndDemo(device="cpu")
    images = demo.create_toy_images(batch_size=2, height=32, width=32)

    reconstructed, original, mask = demo.generate_image_completion(
        images, mask_fraction=0.5, num_diffusion_steps=5
    )

    # Check shapes
    assert reconstructed.shape == images.shape, (
        f"Expected shape {images.shape}, got {reconstructed.shape}"
    )
    assert original.shape == images.shape, (
        f"Expected shape {images.shape}, got {original.shape}"
    )

    # Check range
    assert reconstructed.min() >= 0 and reconstructed.max() <= 1, (
        f"Reconstructed should be in [0, 1], got [{reconstructed.min()}, {reconstructed.max()}]"
    )

    print("✓ Image completion test passed")


def test_metrics_computation():
    """Test that metrics can be computed."""
    demo = EndToEndDemo(device="cpu")
    images = demo.create_toy_images(batch_size=2, height=32, width=32)

    # Create slightly perturbed reconstructions
    reconstructed = images + 0.01 * torch.randn_like(images)
    reconstructed = torch.clamp(reconstructed, 0, 1)

    metrics = demo.compute_metrics(images, reconstructed)

    # Check that metrics are computed
    assert "mse" in metrics, "MSE metric should be present"
    assert "l1" in metrics, "L1 metric should be present"
    assert "psnr" in metrics, "PSNR metric should be present"

    # Check that metrics are reasonable
    assert metrics["mse"] > 0, "MSE should be positive"
    assert metrics["l1"] > 0, "L1 should be positive"
    assert metrics["psnr"] > 0, "PSNR should be positive"

    print(f"✓ Metrics computation test passed")
    print(f"  Metrics: {metrics}")


def test_full_demo():
    """Test the complete demo pipeline."""
    demo = EndToEndDemo(device="cpu")

    # Run demo
    results = demo.run_demo(batch_size=4, mask_fraction=0.5, num_diffusion_steps=5)

    # Check that results contain expected keys
    assert "original_images" in results, "Results should contain original_images"
    assert "reconstructed_images" in results, "Results should contain reconstructed_images"
    assert "mask" in results, "Results should contain mask"
    assert "metrics" in results, "Results should contain metrics"

    # Check shapes
    original = results["original_images"]
    reconstructed = results["reconstructed_images"]
    assert original.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {original.shape}"
    assert reconstructed.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {reconstructed.shape}"

    # Check metrics
    metrics = results["metrics"]
    print(f"Metrics from full demo: {metrics}")
    assert metrics["mse"] >= 0, "MSE should be non-negative"

    print("✓ Full demo test passed")


if __name__ == "__main__":
    test_toy_image_generation()
    test_encode_decode_cycle()
    test_token_masking()
    test_token_regeneration()
    test_image_completion()
    test_metrics_computation()
    test_full_demo()
    print("\n✅ All end-to-end tests passed!")
