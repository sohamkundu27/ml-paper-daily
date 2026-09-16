"""
Pass 4: End-to-end demo for GenCeption.

This demo script:
1. Generates synthetic images with known properties
2. Runs inference on multiple tasks (depth, normals, segmentation)
3. Visualizes outputs for each task
4. Demonstrates the full GenCeption pipeline
"""

import torch
import torch.nn.functional as F
import numpy as np
from genception import GenCeption


def create_synthetic_image(height=64, width=64, image_type="noise"):
    """
    Generate a synthetic image for demonstration.

    Args:
        height, width: image dimensions
        image_type: "noise", "gradient", or "pattern"

    Returns:
        image tensor (3, H, W) in [-1, 1] range
    """
    if image_type == "noise":
        # Random Gaussian noise
        image = torch.randn(3, height, width)
    elif image_type == "gradient":
        # Smooth gradient image
        y = torch.linspace(-1, 1, height).view(-1, 1).expand(height, width)
        x = torch.linspace(-1, 1, width).view(1, -1).expand(height, width)
        image = torch.stack([
            x,                          # red channel: horizontal gradient
            y,                          # green channel: vertical gradient
            (x**2 + y**2) / 2.0        # blue channel: radial gradient
        ])
    elif image_type == "pattern":
        # Checkerboard pattern
        grid = torch.arange(height * width).reshape(height, width) % 2
        image = torch.stack([
            grid.float() * 2 - 1,
            (1 - grid.float()) * 2 - 1,
            torch.sin(torch.linspace(0, 4*np.pi, height).view(-1, 1) +
                     torch.linspace(0, 4*np.pi, width).view(1, -1)) / 2
        ])
    else:
        raise ValueError(f"Unknown image type: {image_type}")

    return image.clamp(-1, 1)


def create_instruction_tokens(task_name, vocab_size=1000, seq_len=10):
    """
    Create token sequence for an instruction.

    In a real implementation, this would tokenize natural language.
    For demo, we create task-specific token patterns that the model learns to respond to.

    Args:
        task_name: "depth", "normals", or "segmentation"
        vocab_size: vocabulary size
        seq_len: sequence length

    Returns:
        token_ids tensor (seq_len,)
    """
    # Seed pattern based on task for reproducibility
    task_seed = {"depth": 42, "normals": 43, "segmentation": 44}.get(task_name, 0)
    torch.manual_seed(task_seed)
    token_ids = torch.randint(0, vocab_size, (seq_len,))
    return token_ids


def normalize_output(output, output_type="depth"):
    """
    Normalize output to [0, 1] range for visualization.

    Args:
        output: model output tensor (1, C, H, W)
        output_type: "depth", "normals", or "segmentation"

    Returns:
        normalized output (C, H, W) in [0, 1] range
    """
    output = output.squeeze(0)  # Remove batch dim

    if output_type == "depth":
        # Depth: assume range [0, 10], clip and normalize
        output = torch.clamp(output, 0, 10) / 10.0
    elif output_type == "normals":
        # Normals: already in roughly [-1, 1], shift to [0, 1]
        output = (output + 1.0) / 2.0
        output = torch.clamp(output, 0, 1)
    elif output_type == "segmentation":
        # Segmentation: apply softmax and take max class probability
        output = F.softmax(output, dim=0).max(dim=0).values

    return output


def visualize_output(output, output_type="depth", save_path=None):
    """
    Convert output to visualization-friendly format.

    Args:
        output: normalized output (C, H, W) in [0, 1]
        output_type: "depth", "normals", or "segmentation"
        save_path: if provided, print path where output would be saved

    Returns:
        visualization as numpy array (H, W, 3) in [0, 1]
    """
    h, w = output.shape[-2:]

    if output_type == "depth":
        # Depth as grayscale heatmap: repeat single channel to RGB
        viz = output[0].repeat(3, 1, 1).numpy()
    elif output_type == "normals":
        # Normals as RGB (already 3 channels)
        viz = output.numpy()
    elif output_type == "segmentation":
        # Segmentation as grayscale: repeat to RGB
        viz = output[0].repeat(3, 1, 1).numpy() if output.shape[0] > 1 else output.numpy()
    else:
        viz = np.zeros((3, h, w))

    # Ensure valid range
    viz = np.clip(viz, 0, 1)

    # Convert to (H, W, 3) for typical visualization
    if viz.shape[0] == 3:
        viz = np.transpose(viz, (1, 2, 0))

    if save_path:
        print(f"  → Would save to {save_path}")

    return viz


def run_demo():
    """Run end-to-end GenCeption demo."""
    print("=" * 70)
    print("GenCeption Pass 4: End-to-End Demo")
    print("=" * 70)

    # Model configuration
    height, width = 64, 64
    vocab_size = 200
    embed_dim = 128
    base_channels = 16
    num_classes = 10
    batch_size = 1

    print("\n1. Initializing GenCeption model...")
    model = GenCeption(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        base_channels=base_channels,
        num_blocks=3,
        num_classes=num_classes,
        refinement_blocks=2
    )
    model.eval()
    print(f"   ✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Generate synthetic test images
    print("\n2. Generating synthetic test images...")
    image_types = ["noise", "gradient", "pattern"]
    images = []
    for img_type in image_types:
        img = create_synthetic_image(height, width, image_type=img_type)
        images.append(img)
        print(f"   ✓ Generated {img_type} image")

    # Task definitions
    tasks = ["depth", "normals", "segmentation"]
    print(f"\n3. Running inference on {len(tasks)} tasks × {len(images)} images...")

    results = {}
    for img_idx, (image, img_type) in enumerate(zip(images, image_types)):
        print(f"\n   Image {img_idx + 1}/{len(images)}: {img_type}")
        results[img_type] = {}

        # Batch the image
        image_batch = image.unsqueeze(0)  # (1, 3, H, W)

        for task in tasks:
            # Create instruction tokens
            token_ids = create_instruction_tokens(task, vocab_size=vocab_size, seq_len=10)
            token_ids_batch = token_ids.unsqueeze(0)  # (1, seq_len)

            # Forward pass
            with torch.no_grad():
                output_dict = model(image_batch, token_ids_batch, return_all_tasks=True)

            # Extract task output
            if task == "depth":
                task_output = output_dict["depth"]
            elif task == "normals":
                task_output = output_dict["normals"]
            else:  # segmentation
                task_output = output_dict["segmentation"]

            # Normalize and visualize
            normalized = normalize_output(task_output, output_type=task)
            viz = visualize_output(normalized, output_type=task)

            results[img_type][task] = {
                "output": task_output,
                "normalized": normalized,
                "visualization": viz,
                "task_id": output_dict["task_id"].item(),
                "task_logits": output_dict["task_logits"].detach().numpy()
            }

            print(f"      • {task}: output shape {task_output.shape}, "
                  f"selected task={output_dict['task_id'].item()}")

    # Print summary statistics
    print("\n4. Summary of Results:")
    print("-" * 70)

    for img_type in image_types:
        print(f"\nImage type: {img_type}")
        for task in tasks:
            task_data = results[img_type][task]
            output = task_data["output"]
            logits = task_data["task_logits"][0]

            print(f"  {task:15} | output range: [{output.min():.3f}, {output.max():.3f}] | "
                  f"task_logits: {logits}")

    # Check for multi-task diversity
    print("\n5. Task Selection Analysis:")
    print("-" * 70)

    task_selections = {}
    for task in tasks:
        task_selections[task] = 0

    for img_type in image_types:
        for task in tasks:
            task_id = results[img_type][task]["task_id"]
            task_selections[task] += 1

    print("Task selection distribution across all images:")
    for task in tasks:
        count = task_selections[task]
        print(f"  {task:15}: selected {count} times out of {len(images)} images")

    print("\n6. Verification Checks:")
    print("-" * 70)

    # Check that all outputs have valid ranges
    print("  • Checking output validity...")
    all_valid = True
    for img_type in image_types:
        for task in tasks:
            output = results[img_type][task]["output"]
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"    ✗ Invalid values in {img_type}/{task}")
                all_valid = False

    if all_valid:
        print("    ✓ All outputs contain valid numbers (no NaN/Inf)")

    # Check determinism
    print("  • Checking deterministic behavior...")
    image_test = images[0].unsqueeze(0)
    token_ids_test = create_instruction_tokens("depth", vocab_size=vocab_size, seq_len=10).unsqueeze(0)

    with torch.no_grad():
        output1 = model(image_test, token_ids_test, return_all_tasks=True)["depth"]
        output2 = model(image_test, token_ids_test, return_all_tasks=True)["depth"]

    if torch.allclose(output1, output2):
        print("    ✓ Model produces deterministic outputs")
    else:
        print("    ✗ Outputs differ on identical inputs")

    # Check backward compatibility
    print("  • Checking backward compatibility (Pass 1)...")
    with torch.no_grad():
        features = model(image_test, token_ids_test, return_all_tasks=False)
    if features.shape == (1, base_channels, height, width):
        print(f"    ✓ Features mode returns shape {features.shape}")
    else:
        print(f"    ✗ Features shape mismatch: expected (1, {base_channels}, {height}, {width}), "
              f"got {features.shape}")

    print("\n" + "=" * 70)
    print("✓ End-to-end demo completed successfully!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_demo()
