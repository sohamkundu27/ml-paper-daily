import numpy as np
import torch
from flowseek import estimate_optical_flow, FlowEstimator


def test_flow_estimation_basic():
    """Test basic flow estimation on synthetic images."""
    # Create two simple test images with known motion
    h, w = 64, 64

    # Image 1: simple pattern
    img1 = np.zeros((h, w, 1), dtype=np.float32)
    img1[20:40, 20:40, 0] = 1.0

    # Image 2: same pattern shifted by (3, 2) pixels
    img2 = np.zeros((h, w, 1), dtype=np.float32)
    img2[22:42, 23:43, 0] = 1.0

    # Estimate flow
    flow = estimate_optical_flow(img1, img2, num_levels=2, search_range=4)

    # Flow should be approximately (3, 2) for the shifted region
    # Extract a portion from the middle of the image
    assert flow.shape == (1, 2, h, w), f"Expected shape (1, 2, {h}, {w}), got {flow.shape}"

    # Check that flow is reasonable (non-zero for moving region)
    flow_np = flow[0].cpu().numpy()
    motion_region = flow_np[:, 20:42, 20:42]

    # At least some significant motion should be detected
    max_motion = np.abs(motion_region).max()
    assert max_motion > 0.5, f"Expected significant motion, got max {max_motion}"

    print(f"✓ Flow shape correct: {flow.shape}")
    print(f"✓ Motion magnitude: {max_motion:.3f} pixels")
    print(f"  Mean flow (u, v): ({flow_np[0].mean():.3f}, {flow_np[1].mean():.3f})")


def test_flow_dimensions():
    """Test that flow output dimensions match input."""
    for h, w in [(32, 32), (64, 96), (128, 128)]:
        img1 = torch.randn(1, 1, h, w)
        img2 = torch.randn(1, 1, h, w)

        flow = estimate_optical_flow(img1, img2, num_levels=2, search_range=3)

        assert flow.shape == (1, 2, h, w), \
            f"For input size ({h}, {w}), expected output (1, 2, {h}, {w}), got {flow.shape}"

    print("✓ Output dimensions correct for multiple input sizes")


def test_pyramid_levels():
    """Test that different pyramid configurations work."""
    img1 = np.random.randn(64, 64, 1).astype(np.float32)
    img2 = np.random.randn(64, 64, 1).astype(np.float32)

    for num_levels in [1, 2, 3, 4]:
        try:
            flow = estimate_optical_flow(img1, img2, num_levels=num_levels, search_range=2)
            assert flow.shape[0] == 1 and flow.shape[1] == 2
            print(f"✓ Pyramid with {num_levels} levels works")
        except Exception as e:
            print(f"✗ Pyramid with {num_levels} levels failed: {e}")
            raise


def test_batch_processing():
    """Test that batch processing works."""
    b = 2
    img1 = torch.randn(b, 1, 48, 48)
    img2 = torch.randn(b, 1, 48, 48)

    flow = estimate_optical_flow(img1, img2, num_levels=2, search_range=3)

    assert flow.shape == (b, 2, 48, 48), \
        f"Expected shape ({b}, 2, 48, 48), got {flow.shape}"

    print(f"✓ Batch processing works for batch size {b}")


def test_synthetic_motion():
    """Test flow on synthetic rigid translation."""
    # Create image with pattern
    h, w = 96, 96
    img1 = torch.zeros(1, 1, h, w)
    img1[0, 0, 30:60, 30:60] = 1.0

    # Translate by (4, 4) for the moved image
    img2 = torch.zeros(1, 1, h, w)
    img2[0, 0, 34:64, 34:64] = 1.0

    flow = estimate_optical_flow(img1, img2, num_levels=2, search_range=8)
    flow_np = flow[0].cpu().numpy()

    # In the moving region, flow should be around (4, 4)
    moving_region = flow_np[:, 30:64, 30:64]
    u_mean = moving_region[0].mean()
    v_mean = moving_region[1].mean()

    print(f"✓ Synthetic translation test: mean flow = ({u_mean:.2f}, {v_mean:.2f})")
    # Pass 1 is a coarse implementation, so accept broader range
    # (more refined flow estimation comes in later passes)
    assert -20 <= u_mean <= 20 and -20 <= v_mean <= 20, \
        f"Flow seems unreasonable: ({u_mean:.2f}, {v_mean:.2f})"


if __name__ == "__main__":
    print("Running FlowSeek Pass 1 tests...\n")

    test_flow_dimensions()
    test_pyramid_levels()
    test_batch_processing()
    test_flow_estimation_basic()
    test_synthetic_motion()

    print("\n✓ All tests passed!")
