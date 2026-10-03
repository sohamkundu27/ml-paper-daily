"""
Pass 4: End-to-end FlowSeek demo with synthetic data and evaluation.
Demonstrates the complete optical flow pipeline with visualization and metrics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from flowseek import estimate_optical_flow


def create_synthetic_translation(h=128, w=128, tx=5.0, ty=3.0):
    """Create synthetic image pair with pure translation."""
    # Create first image: a circle in the center
    img1 = np.zeros((h, w, 1), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    center_y, center_x = h // 2, w // 2
    radius = 20
    circle = (yy - center_y)**2 + (xx - center_x)**2 <= radius**2
    img1[circle, 0] = 1.0

    # Add some texture
    texture = np.sin(xx / 5.0) * np.sin(yy / 5.0) * 0.5
    img1[:, :, 0] += texture * 0.3

    # Create second image: translate the circle
    img2 = np.zeros((h, w, 1), dtype=np.float32)
    circle_shifted = (yy - (center_y + ty))**2 + (xx - (center_x + tx))**2 <= radius**2
    img2[circle_shifted, 0] = 1.0
    img2[:, :, 0] += np.sin((xx - tx) / 5.0) * np.sin((yy - ty) / 5.0) * 0.5 * 0.3

    # Ground truth flow
    gt_flow = np.zeros((1, 2, h, w), dtype=np.float32)
    gt_flow[0, 0, :, :] = tx
    gt_flow[0, 1, :, :] = ty

    return img1, img2, gt_flow


def create_synthetic_affine(h=128, w=128):
    """Create synthetic image pair with affine motion."""
    # Create first image
    img1 = np.zeros((h, w, 1), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Draw a rectangle
    rect = (yy >= 30) & (yy < 100) & (xx >= 30) & (xx < 100)
    img1[rect, 0] = 1.0

    # Add checkerboard pattern
    checker = ((xx // 10 + yy // 10) % 2).astype(np.float32) * 0.3
    img1[:, :, 0] += checker

    # Create second image with affine transformation
    # Simulate: u = 2 + 0.05*x + 0.02*y, v = 1 + 0.02*x + 0.05*y
    img2 = np.zeros((h, w, 1), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = 2.0 + 0.05 * x + 0.02 * y
            dy = 1.0 + 0.02 * x + 0.05 * y

            src_x = int(x - dx)
            src_y = int(y - dy)

            if 0 <= src_x < w and 0 <= src_y < h:
                img2[y, x, 0] = img1[src_y, src_x, 0]

    # Ground truth flow
    gt_flow = np.zeros((1, 2, h, w), dtype=np.float32)
    yy_f, xx_f = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    gt_flow[0, 0, :, :] = 2.0 + 0.05 * xx_f + 0.02 * yy_f
    gt_flow[0, 1, :, :] = 1.0 + 0.02 * xx_f + 0.05 * yy_f

    return img1, img2, gt_flow


def compute_flow_metrics(estimated_flow, gt_flow, mask=None):
    """Compute optical flow error metrics."""
    est = estimated_flow[0].cpu().numpy()  # (2, h, w)
    gt = gt_flow[0]  # (2, h, w)

    # Compute endpoint error (EPE)
    epe = np.sqrt((est[0] - gt[0])**2 + (est[1] - gt[1])**2)

    if mask is not None:
        epe = epe[mask]

    mean_epe = np.mean(epe)

    # Compute angular error
    est_norm = np.sqrt(est[0]**2 + est[1]**2 + 1e-8)
    gt_norm = np.sqrt(gt[0]**2 + gt[1]**2 + 1e-8)

    # Dot product and cross product for angle
    dot = est[0] * gt[0] + est[1] * gt[1]
    cross = np.abs(est[0] * gt[1] - est[1] * gt[0])

    angle = np.arctan2(cross, dot) * 180.0 / np.pi

    if mask is not None:
        angle = angle[mask]

    mean_angle = np.mean(angle)

    return {
        "mean_epe": mean_epe,
        "median_epe": np.median(epe),
        "std_epe": np.std(epe),
        "mean_angle_error": mean_angle,
    }


def visualize_flow(flow, title="Optical Flow", magnitude_scale=1.0):
    """Visualize optical flow using color wheel representation."""
    flow_np = flow[0].cpu().numpy()  # (2, h, w)
    u, v = flow_np[0], flow_np[1]

    # Compute magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)

    # Normalize magnitude for visualization
    mag_norm = mag / (np.percentile(mag, 99) + 1e-8)
    mag_norm = np.clip(mag_norm, 0, 1)

    # Convert to HSV for color wheel
    h, w = mag.shape
    hsv = np.zeros((h, w, 3), dtype=np.float32)

    # Hue from angle
    hsv[:, :, 0] = (ang + np.pi) / (2 * np.pi)

    # Saturation and value from magnitude
    hsv[:, :, 1] = np.ones_like(mag_norm)
    hsv[:, :, 2] = mag_norm

    # Convert HSV to RGB
    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')

    return rgb


def demo_translation():
    """Demo 1: Translation motion."""
    print("\n" + "="*60)
    print("DEMO 1: Translation Motion")
    print("="*60)

    h, w = 128, 128
    tx, ty = 5.0, 3.0
    img1, img2, gt_flow = create_synthetic_translation(h, w, tx, ty)

    print(f"Image size: {h}x{w}")
    print(f"Ground truth translation: ({tx:.1f}, {ty:.1f}) pixels")

    # Estimate flow without motion basis
    print("\n--- Without motion basis ---")
    flow_raw = estimate_optical_flow(img1, img2, num_levels=3, search_range=4,
                                     use_depth=False, motion_basis_type=None)
    metrics_raw = compute_flow_metrics(flow_raw, gt_flow)
    print(f"Mean EPE: {metrics_raw['mean_epe']:.3f} pixels")
    print(f"Mean angular error: {metrics_raw['mean_angle_error']:.2f}°")

    # Estimate flow with translation motion basis
    print("\n--- With translation motion basis ---")
    flow_translation = estimate_optical_flow(img1, img2, num_levels=3, search_range=4,
                                             use_depth=False, motion_basis_type="translation",
                                             motion_basis_strength=0.8)
    metrics_trans = compute_flow_metrics(flow_translation, gt_flow)
    print(f"Mean EPE: {metrics_trans['mean_epe']:.3f} pixels")
    print(f"Mean angular error: {metrics_trans['mean_angle_error']:.2f}°")
    print(f"Improvement: {(metrics_raw['mean_epe'] - metrics_trans['mean_epe']) / metrics_raw['mean_epe'] * 100:.1f}%")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img1[:, :, 0], cmap='gray')
    axes[0].set_title('Image 1')
    axes[0].axis('off')

    axes[1].imshow(img2[:, :, 0], cmap='gray')
    axes[1].set_title('Image 2')
    axes[1].axis('off')

    # Visualize flow (use the regularized one)
    flow_np = flow_translation[0].cpu().numpy()
    u, v = flow_np[0], flow_np[1]
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)

    mag_norm = mag / (np.percentile(mag, 99) + 1e-8)
    mag_norm = np.clip(mag_norm, 0, 1)

    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[:, :, 0] = (ang + np.pi) / (2 * np.pi)
    hsv[:, :, 1] = np.ones_like(mag_norm)
    hsv[:, :, 2] = mag_norm

    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)
    axes[2].imshow(rgb)
    axes[2].set_title(f'Estimated Flow (EPE: {metrics_trans["mean_epe"]:.3f})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('/tmp/flowseek_demo_translation.png', dpi=100, bbox_inches='tight')
    print("\nVisualization saved to /tmp/flowseek_demo_translation.png")
    plt.close()


def demo_affine():
    """Demo 2: Affine motion."""
    print("\n" + "="*60)
    print("DEMO 2: Affine Motion")
    print("="*60)

    h, w = 128, 128
    img1, img2, gt_flow = create_synthetic_affine(h, w)

    print(f"Image size: {h}x{w}")
    print("Ground truth: affine deformation (shear + zoom)")

    # Estimate flow without motion basis
    print("\n--- Without motion basis ---")
    flow_raw = estimate_optical_flow(img1, img2, num_levels=3, search_range=4,
                                     use_depth=False, motion_basis_type=None)
    metrics_raw = compute_flow_metrics(flow_raw, gt_flow)
    print(f"Mean EPE: {metrics_raw['mean_epe']:.3f} pixels")
    print(f"Mean angular error: {metrics_raw['mean_angle_error']:.2f}°")

    # Estimate flow with affine motion basis
    print("\n--- With affine motion basis ---")
    flow_affine = estimate_optical_flow(img1, img2, num_levels=3, search_range=4,
                                        use_depth=False, motion_basis_type="affine",
                                        motion_basis_strength=0.8)
    metrics_aff = compute_flow_metrics(flow_affine, gt_flow)
    print(f"Mean EPE: {metrics_aff['mean_epe']:.3f} pixels")
    print(f"Mean angular error: {metrics_aff['mean_angle_error']:.2f}°")
    print(f"Improvement: {(metrics_raw['mean_epe'] - metrics_aff['mean_epe']) / metrics_raw['mean_epe'] * 100:.1f}%")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img1[:, :, 0], cmap='gray')
    axes[0].set_title('Image 1')
    axes[0].axis('off')

    axes[1].imshow(img2[:, :, 0], cmap='gray')
    axes[1].set_title('Image 2')
    axes[1].axis('off')

    # Visualize flow
    flow_np = flow_affine[0].cpu().numpy()
    u, v = flow_np[0], flow_np[1]
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)

    mag_norm = mag / (np.percentile(mag, 99) + 1e-8)
    mag_norm = np.clip(mag_norm, 0, 1)

    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[:, :, 0] = (ang + np.pi) / (2 * np.pi)
    hsv[:, :, 1] = np.ones_like(mag_norm)
    hsv[:, :, 2] = mag_norm

    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)
    axes[2].imshow(rgb)
    axes[2].set_title(f'Estimated Flow (EPE: {metrics_aff["mean_epe"]:.3f})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('/tmp/flowseek_demo_affine.png', dpi=100, bbox_inches='tight')
    print("\nVisualization saved to /tmp/flowseek_demo_affine.png")
    plt.close()


def demo_pipeline_comparison():
    """Demo 3: Compare different pipeline configurations."""
    print("\n" + "="*60)
    print("DEMO 3: Pipeline Configuration Comparison")
    print("="*60)

    h, w = 128, 128
    tx, ty = 5.0, 3.0
    img1, img2, gt_flow = create_synthetic_translation(h, w, tx, ty)

    print(f"Image size: {h}x{w}, Ground truth: ({tx:.1f}, {ty:.1f}) pixels")

    configs = [
        {"name": "Basic (no depth, no basis)", "use_depth": False, "motion_basis_type": None},
        {"name": "Depth-guided only", "use_depth": True, "motion_basis_type": None},
        {"name": "Translation basis only", "use_depth": False, "motion_basis_type": "translation"},
        {"name": "Depth + translation basis", "use_depth": True, "motion_basis_type": "translation"},
    ]

    results = []
    for config in configs:
        print(f"\n{config['name']}:")
        flow = estimate_optical_flow(img1, img2, num_levels=3, search_range=4,
                                     use_depth=config["use_depth"],
                                     motion_basis_type=config["motion_basis_type"],
                                     motion_basis_strength=0.7)
        metrics = compute_flow_metrics(flow, gt_flow)
        print(f"  Mean EPE: {metrics['mean_epe']:.3f} pixels")
        print(f"  Median EPE: {metrics['median_epe']:.3f} pixels")
        print(f"  Angular error: {metrics['mean_angle_error']:.2f}°")
        results.append((config['name'], metrics))

    # Print summary table
    print("\n" + "-"*60)
    print("SUMMARY TABLE")
    print("-"*60)
    print(f"{'Configuration':<35} {'Mean EPE':<12} {'Angle Err':<12}")
    print("-"*60)
    for name, metrics in results:
        print(f"{name:<35} {metrics['mean_epe']:<12.3f} {metrics['mean_angle_error']:<12.2f}°")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FlowSeek Pass 4: End-to-End Demo")
    print("Synthetic data evaluation with visualization")
    print("="*60)

    # Run all demos
    demo_translation()
    demo_affine()
    demo_pipeline_comparison()

    print("\n" + "="*60)
    print("Demo complete! Visualizations saved to /tmp/")
    print("="*60)
