import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from detection_head import YOLOv10DetectorPass3


class SyntheticObjectDataset(Dataset):
    """
    Synthetic dataset for training. Generates random images with random bounding boxes.
    Useful for testing training loop without needing real data.
    """
    def __init__(self, num_samples=100, img_size=320, num_classes=80, max_objects=5, augment=True):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.augment = augment

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image
        image = torch.rand(3, self.img_size, self.img_size)

        # Generate objects with centers in middle region where dense predictions exist
        # This ensures better target-prediction overlap for training
        num_objects = random.randint(1, self.max_objects)
        boxes = []
        classes = []

        # Place boxes in the center portion of the image (this is where many predictions exist)
        center_start = self.img_size // 4
        center_end = (3 * self.img_size) // 4

        for _ in range(num_objects):
            cx = random.uniform(center_start, center_end)
            cy = random.uniform(center_start, center_end)
            w = random.uniform(40, 80)
            h = random.uniform(40, 80)

            x1 = max(0, cx - w / 2)
            y1 = max(0, cy - h / 2)
            x2 = min(self.img_size, cx + w / 2)
            y2 = min(self.img_size, cy + h / 2)

            boxes.append([x1, y1, x2, y2])
            classes.append(random.randint(0, self.num_classes - 1))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)

        if self.augment:
            image = self._augment_image(image)

        return {
            'image': image,
            'boxes': boxes,
            'classes': classes,
        }

    def _augment_image(self, image):
        """Apply random augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[2])

        # Random brightness/contrast
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = image * brightness_factor
            image = torch.clamp(image, 0, 1)

        # Random color jitter
        if random.random() > 0.5:
            image[0] = image[0] * random.uniform(0.8, 1.2)  # Red
            image[1] = image[1] * random.uniform(0.8, 1.2)  # Green
            image[2] = image[2] * random.uniform(0.8, 1.2)  # Blue
            image = torch.clamp(image, 0, 1)

        return image


class TrainingUtils:
    """Utilities for training Pass 3 models."""

    @staticmethod
    def collate_fn(batch):
        """Collate batch with variable-length boxes."""
        images = torch.stack([item['image'] for item in batch])
        max_boxes = max(len(item['boxes']) for item in batch)

        batch_boxes = torch.full(
            (len(batch), max_boxes, 4),
            0.0,
            dtype=torch.float32
        )
        batch_classes = torch.full(
            (len(batch), max_boxes),
            -1,
            dtype=torch.long
        )

        for i, item in enumerate(batch):
            num_boxes = len(item['boxes'])
            batch_boxes[i, :num_boxes] = item['boxes']
            batch_classes[i, :num_boxes] = item['classes']

        return {
            'images': images,
            'boxes': batch_boxes,
            'classes': batch_classes,
        }

    @staticmethod
    def train_one_epoch(model, dataloader, optimizer, device, epoch=0):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(device)
            targets = {
                'boxes': batch['boxes'].to(device),
                'classes': batch['classes'].to(device),
            }

            optimizer.zero_grad()
            outputs = model(images, targets=targets)
            loss = outputs['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    @staticmethod
    def validate(model, dataloader, device):
        """Validate model on dataset."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(device)
                targets = {
                    'boxes': batch['boxes'].to(device),
                    'classes': batch['classes'].to(device),
                }

                outputs = model(images, targets=targets)
                loss = outputs['loss']

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss


def train_model(
    num_epochs=5,
    batch_size=8,
    learning_rate=0.001,
    num_train_samples=100,
    num_val_samples=20,
    img_size=320,
    num_classes=80,
    device=None,
):
    """
    Full training pipeline for Pass 3 model.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_train_samples: Number of synthetic training samples
        num_val_samples: Number of synthetic validation samples
        img_size: Image size for training
        num_classes: Number of object classes
        device: torch device

    Returns:
        model: Trained YOLOv10DetectorPass3 model
        history: Training history dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Create model
    model = YOLOv10DetectorPass3(num_classes=num_classes, lr=learning_rate)
    model.to(device)

    # Create datasets
    train_dataset = SyntheticObjectDataset(
        num_samples=num_train_samples,
        img_size=img_size,
        num_classes=num_classes,
        max_objects=5,
        augment=True
    )

    val_dataset = SyntheticObjectDataset(
        num_samples=num_val_samples,
        img_size=img_size,
        num_classes=num_classes,
        max_objects=5,
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TrainingUtils.collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TrainingUtils.collate_fn,
        num_workers=0
    )

    # Create optimizer
    optimizer = model.get_optimizer()

    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
    }

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Train samples: {num_train_samples}, Val samples: {num_val_samples}")
    print()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = TrainingUtils.train_one_epoch(model, train_loader, optimizer, device, epoch)
        history['train_loss'].append(train_loss)
        print(f"  Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = TrainingUtils.validate(model, val_loader, device)
        history['val_loss'].append(val_loss)
        print(f"  Val Loss: {val_loss:.4f}")
        print()

    print("Training complete!")
    return model, history


if __name__ == '__main__':
    # Train a small model for testing
    model, history = train_model(
        num_epochs=3,
        batch_size=4,
        learning_rate=0.001,
        num_train_samples=20,
        num_val_samples=10,
        img_size=320,
        num_classes=80,
    )

    print("\nTraining history:")
    for epoch, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
