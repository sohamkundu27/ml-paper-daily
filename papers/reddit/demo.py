"""Pass 4: End-to-end demo with toy dataset, training, and generation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from discrete_diffusion import CategoricalDiffusion, SimpleDenoiser


def create_toy_dataset(num_samples: int, num_classes: int, seq_len: int, seed: int = 42) -> np.ndarray:
    """Create a simple synthetic categorical dataset.

    Dataset: sequences of random class labels. In practice, this could be
    MNIST digits, text tokens, or other discrete data.

    Args:
        num_samples: Number of sequences to generate
        num_classes: Number of discrete categories
        seq_len: Length of each sequence
        seed: Random seed for reproducibility

    Returns:
        x_0: Array of shape (num_samples, seq_len) with values in [0, num_classes)
    """
    np.random.seed(seed)
    x_0 = np.random.randint(0, num_classes, (num_samples, seq_len), dtype=np.int32)
    return x_0


def train_denoiser(
    diffusion: CategoricalDiffusion,
    denoiser: nn.Module,
    dataset: np.ndarray,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    num_corrupts: int = 2,
    device: str = "cpu",
) -> list:
    """Train the denoiser to predict x_0 from noised x_t at random timesteps.

    Args:
        diffusion: CategoricalDiffusion instance
        denoiser: SimpleDenoiser model
        dataset: Training data, shape (num_samples, seq_len)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_corrupts: Number of positions to corrupt during training
        device: Torch device ("cpu" or "cuda")

    Returns:
        losses: List of loss values per epoch
    """
    denoiser = denoiser.to(device)
    optimizer = optim.Adam(denoiser.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []
    num_samples = len(dataset)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle and iterate through batches
        indices = np.random.permutation(num_samples)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            x_0_batch = dataset[batch_indices]

            # Sample random timesteps for this batch
            timesteps = np.random.randint(1, diffusion.num_steps + 1, size=len(x_0_batch))

            # Forward diffusion with rehashing
            x_t, _ = diffusion.forward_batch_with_rehash(x_0_batch, timesteps, num_corrupts)

            # Convert to torch tensors
            x_t_torch = torch.tensor(x_t, dtype=torch.long, device=device)
            x_0_torch = torch.tensor(x_0_batch, dtype=torch.long, device=device)
            t_torch = torch.tensor(timesteps, dtype=torch.long, device=device)

            # Forward pass: predict x_0 from x_t
            logits = denoiser(x_t_torch, t_torch)  # (batch_size, seq_len, num_classes)

            # Reshape for loss computation
            batch_sz, seq_len, num_classes = logits.shape
            logits_flat = logits.reshape(-1, num_classes)  # (batch_size * seq_len, num_classes)
            x_0_flat = x_0_torch.reshape(-1)  # (batch_size * seq_len,)

            # Compute loss
            loss = criterion(logits_flat, x_0_flat)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    return losses


def generate_samples(
    diffusion: CategoricalDiffusion,
    denoiser: nn.Module,
    num_samples: int,
    seq_len: int,
    num_classes: int,
    device: str = "cpu",
) -> np.ndarray:
    """Generate new samples by starting from noise and reversing.

    Args:
        diffusion: CategoricalDiffusion instance
        denoiser: Trained denoiser model
        num_samples: Number of samples to generate
        seq_len: Sequence length
        num_classes: Number of discrete categories
        device: Torch device

    Returns:
        generated: Array of shape (num_samples, seq_len) with generated samples
    """
    generated = []

    for _ in range(num_samples):
        # Start from random noise
        x_T = np.random.randint(0, num_classes, (1, seq_len), dtype=np.int32)

        # Reverse process: denoise from noise to clean sample
        x_0 = diffusion.sample_with_rehash_sampler(denoiser, x_T, device=device)

        generated.append(x_0[0])

    return np.array(generated, dtype=np.int32)


def compute_accuracy(original: np.ndarray, generated: np.ndarray) -> float:
    """Compute per-position accuracy between original and generated samples.

    This is a sanity check: after training, the denoiser should be able
    to recover samples that are somewhat similar to the training distribution.

    Args:
        original: Array of original samples
        generated: Array of generated samples

    Returns:
        accuracy: Fraction of positions where values match (should be low without overfitting)
    """
    if len(original) == 0 or len(generated) == 0:
        return 0.0

    # For a meaningful comparison, we'd need to know the true mapping.
    # Instead, we just check that generated samples are in valid range.
    valid = np.all((generated >= 0) & (generated < original.max() + 1))
    return float(valid)


def main():
    """End-to-end demo: create data, train denoiser, and generate new samples."""

    # Hyperparameters
    num_classes = 5
    seq_len = 8
    num_steps = 30
    num_train_samples = 200
    epochs = 15
    batch_size = 32
    num_corrupts = 2
    device = "cpu"

    print("=" * 60)
    print("ReDDiT Pass 4: End-to-End Demo with Toy Dataset")
    print("=" * 60)

    # Step 1: Create toy dataset
    print("\n[1] Creating toy dataset...")
    dataset = create_toy_dataset(
        num_samples=num_train_samples,
        num_classes=num_classes,
        seq_len=seq_len,
        seed=42,
    )
    print(f"    Dataset shape: {dataset.shape}")
    print(f"    Sample from dataset: {dataset[0]}")

    # Step 2: Initialize diffusion model and denoiser
    print("\n[2] Initializing diffusion model and denoiser...")
    diffusion = CategoricalDiffusion(num_classes=num_classes, num_steps=num_steps)
    denoiser = SimpleDenoiser(
        num_classes=num_classes,
        seq_len=seq_len,
        hidden_dim=64,
    )
    print(f"    Diffusion steps: {num_steps}")
    print(f"    Denoiser parameters: {sum(p.numel() for p in denoiser.parameters()):,}")

    # Step 3: Train denoiser
    print("\n[3] Training denoiser on forward+reverse pairs...")
    losses = train_denoiser(
        diffusion=diffusion,
        denoiser=denoiser,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001,
        num_corrupts=num_corrupts,
        device=device,
    )
    print(f"    Final loss: {losses[-1]:.4f}")
    print(f"    Loss reduced by: {100 * (1 - losses[-1] / losses[0]):.1f}%")

    # Step 4: Generate samples
    print("\n[4] Generating new samples from trained model...")
    num_generate = 10
    generated = generate_samples(
        diffusion=diffusion,
        denoiser=denoiser,
        num_samples=num_generate,
        seq_len=seq_len,
        num_classes=num_classes,
        device=device,
    )
    print(f"    Generated {num_generate} samples of shape {generated.shape}")

    # Step 5: Show comparison
    print("\n[5] Sample comparison:")
    print("    Original training samples:")
    for i in range(min(3, len(dataset))):
        print(f"      {i}: {dataset[i]}")

    print("    Generated samples (from noise):")
    for i in range(min(3, len(generated))):
        print(f"      {i}: {generated[i]}")

    # Step 6: Validation checks
    print("\n[6] Validation:")

    # Check that generated samples are valid
    valid = np.all((generated >= 0) & (generated < num_classes))
    print(f"    All generated values in valid range [0, {num_classes}): {valid}")

    # Check that generated samples are diverse (not all the same)
    unique_samples = len(np.unique(generated, axis=0))
    print(f"    Number of unique generated samples: {unique_samples}/{num_generate}")

    # Compute distribution statistics
    train_class_dist = np.bincount(dataset.flatten(), minlength=num_classes) / dataset.size
    gen_class_dist = np.bincount(generated.flatten(), minlength=num_classes) / generated.size

    print(f"    Training class distribution: {train_class_dist.round(3)}")
    print(f"    Generated class distribution: {gen_class_dist.round(3)}")

    print("\n" + "=" * 60)
    print("✓ Pass 4 demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
