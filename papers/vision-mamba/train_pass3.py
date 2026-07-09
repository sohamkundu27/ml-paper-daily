"""Training script for Vision Mamba Pass 3 on CIFAR-10."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from vision_mamba import VisionMambaPass3


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate on a dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 5
    subset_size = 500  # Small subset for demo

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    print("Loading CIFAR-10...")
    train_dataset = datasets.CIFAR10(root='/tmp/cifar10', train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='/tmp/cifar10', train=False,
                                    download=True, transform=transform)

    # Use small subsets for faster training
    train_subset = Subset(train_dataset, list(range(min(subset_size, len(train_dataset)))))
    test_subset = Subset(test_dataset, list(range(min(subset_size // 5, len(test_dataset)))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_subset)}, Test samples: {len(test_subset)}")

    # Model
    print("Creating VisionMambaPass3...")
    model = VisionMambaPass3(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
        num_blocks=2,
        ssm_hidden_dim=128,
        num_classes=10
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
