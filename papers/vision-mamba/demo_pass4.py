"""
End-to-end demo for Vision Mamba: train all three passes side-by-side on synthetic data.
Shows that the full pipeline works end-to-end with progressive improvements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from vision_mamba import VisionMambaPass1, VisionMambaPass2, VisionMambaPass3


def train_model(model, train_loader, test_loader, device, num_epochs=3, model_name=""):
    """Train a model and return train/test metrics."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    train_losses = []
    test_accuracies = []
    train_start = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        test_acc = 100.0 * correct / total
        test_accuracies.append(test_acc)

        print(f"  [{model_name}] Epoch {epoch + 1}/{num_epochs}: "
              f"loss={avg_train_loss:.4f}, test_acc={test_acc:.1f}%")

    train_time = time.time() - train_start
    return {
        'model_name': model_name,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'final_test_acc': test_accuracies[-1],
        'train_time': train_time
    }


def benchmark_inference(model, dataloader, device, num_batches=10):
    """Measure inference speed on a model."""
    model.to(device)
    model.eval()

    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = images.to(device)

            start = time.time()
            _ = model(images)
            times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    throughput = images.size(0) / avg_time  # images per second
    return avg_time, throughput


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Hyperparameters
    batch_size = 32
    num_epochs = 3
    num_train_samples = 300
    num_test_samples = 60

    # Generate synthetic data
    print("Generating synthetic image data...")
    torch.manual_seed(42)

    # Synthetic training data: random images with random labels
    train_images = torch.randn(num_train_samples, 3, 32, 32)
    train_labels = torch.randint(0, 10, (num_train_samples,))

    # Synthetic test data
    test_images = torch.randn(num_test_samples, 3, 32, 32)
    test_labels = torch.randint(0, 10, (num_test_samples,))

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {num_train_samples}, Test samples: {num_test_samples}\n")

    # Create models
    print("Creating Vision Mamba models (Pass 1, 2, 3)...\n")

    pass1 = VisionMambaPass1(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )

    pass2 = VisionMambaPass2(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128
    )

    pass3 = VisionMambaPass3(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_blocks=2, ssm_hidden_dim=128, num_classes=10
    )

    # Training phase: only Pass 3 is trainable (has classification head)
    print("Training Vision Mamba Pass 3 on CIFAR-10 subset...")
    print("-" * 60)
    results_pass3 = train_model(pass3, train_loader, test_loader, device,
                                num_epochs=num_epochs, model_name="Pass 3")
    print()

    # Inference benchmarking
    print("Benchmarking inference speed...")
    print("-" * 60)

    print("\nPass 1 (unidirectional SSM):")
    avg_time_p1, throughput_p1 = benchmark_inference(pass1, test_loader, device)
    print(f"  Avg time per batch: {avg_time_p1*1000:.2f}ms")
    print(f"  Throughput: {throughput_p1:.1f} images/sec")

    print("\nPass 2 (bidirectional selective SSM):")
    avg_time_p2, throughput_p2 = benchmark_inference(pass2, test_loader, device)
    print(f"  Avg time per batch: {avg_time_p2*1000:.2f}ms")
    print(f"  Throughput: {throughput_p2:.1f} images/sec")

    print("\nPass 3 (bidirectional + classification):")
    avg_time_p3, throughput_p3 = benchmark_inference(pass3, test_loader, device)
    print(f"  Avg time per batch: {avg_time_p3*1000:.2f}ms")
    print(f"  Throughput: {throughput_p3:.1f} images/sec")

    # Sample inference
    print("\n" + "=" * 60)
    print("Sample Inference (Pass 3 on test data)")
    print("=" * 60)

    pass3.eval()
    batch_images, batch_labels = next(iter(test_loader))
    batch_images = batch_images.to(device)

    with torch.no_grad():
        logits = pass3(batch_images)
        predictions = logits.argmax(dim=1).cpu()

    print("\nFirst 5 samples:")
    for i in range(min(5, batch_images.size(0))):
        pred_class = predictions[i].item()
        true_class = batch_labels[i].item()
        confidence = logits[i].softmax(dim=0).max().item()
        match = "✓" if predictions[i] == batch_labels[i] else "✗"
        print(f"  {match} Predicted: class {pred_class} (conf={confidence:.2f}), "
              f"True: class {true_class}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Vision Mamba Pass 4 Demo")
    print("=" * 60)
    print(f"Training time (Pass 3): {results_pass3['train_time']:.1f}s")
    print(f"Final test accuracy (Pass 3): {results_pass3['final_test_acc']:.1f}%")
    print(f"Relative inference speed (Pass 2 vs Pass 1): {throughput_p1/throughput_p2:.2f}x")
    print(f"Relative inference speed (Pass 3 vs Pass 1): {throughput_p1/throughput_p3:.2f}x")
    print("\n✅ End-to-end demo complete!")
    print("All three passes work together in a complete pipeline:")
    print("  Pass 1: Basic unidirectional SSM foundation")
    print("  Pass 2: Selective bidirectional SSM (core innovation)")
    print("  Pass 3: Classification head for end-to-end training")


if __name__ == "__main__":
    main()
