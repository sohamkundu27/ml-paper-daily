"""
Pass 3: Training loop and loss computation.

This implements the training pipeline for VAR:
- Cross-entropy loss for next-scale token prediction
- Training loop with optimizer and gradient updates
- Synthetic toy dataset for verification that the model learns
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from var_pass2 import VARPass2


class ToyImageDataset(Dataset):
    """
    Synthetic dataset of random images for training VAR.

    Since we're not using actual images, we generate random tensors.
    In a real scenario, these would be actual images with ground-truth
    token labels from a pre-trained tokenizer.
    """

    def __init__(self, num_samples=100, img_size=64, num_channels=3):
        """
        Args:
            num_samples: Number of synthetic images to generate
            img_size: Height and width of images (must be power of 2 for hierarchical tokenization)
            num_channels: Number of color channels (default 3 for RGB)
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_channels = num_channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image tensor normalized to [0, 1]
        image = torch.randn(self.num_channels, self.img_size, self.img_size)
        return image


class VARTrainer:
    """
    Trainer for Visual AutoRegressive Modeling.

    Handles the training loop, loss computation, and gradient updates.
    The loss is cross-entropy over predicted tokens at each scale,
    with the model learning to predict finer scales from coarser ones.
    """

    def __init__(self, model, device="cpu", learning_rate=1e-3):
        """
        Args:
            model: VARPass2 model instance
            device: Device to train on (cpu or cuda)
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_losses = []

    def compute_target_tokens(self, token_maps, num_scales):
        """
        Convert token maps into target token indices for loss computation.

        For each scale i, the target is the quantized token from scale i.
        This simulates having a learned codebook (which is simplified compared to
        the full VAR paper that uses VQ-VAE).

        Args:
            token_maps: List of token maps from the tokenizer
            num_scales: Number of scales

        Returns:
            target_tokens: Tensor of shape (B, N) where N is total number of tokens
        """
        targets = []
        for scale_idx, token_map in enumerate(token_maps):
            B, D, H, W = token_map.shape
            # Simple quantization: hash the features to discrete token indices
            # This is a crude approximation; real VAR uses a learned VQ-VAE codebook
            tokens = token_map.permute(0, 2, 3, 1)  # (B, H, W, D)

            # Quantize: compute a pseudo-token index from the feature vector
            # Use sum over feature dimensions to create integer indices
            token_indices = (tokens.sum(dim=-1) * 100).long() % 4096  # Hash to vocab size
            token_indices = token_indices.reshape(B, H * W)
            targets.append(token_indices)

        # Concatenate across scales
        target_tokens = torch.cat(targets, dim=1)  # (B, N)
        return target_tokens

    def train_step(self, batch):
        """
        Single training step: forward pass, loss computation, backward pass.

        Args:
            batch: Batch of images from the dataset

        Returns:
            loss: Scalar loss value
        """
        images = batch.to(self.device)

        # Forward pass: get logits for all tokens across all scales
        logits, token_maps = self.model(images)

        # Compute target tokens from the tokenizer output
        target_tokens = self.compute_target_tokens(token_maps, self.model.num_scales)
        target_tokens = target_tokens.to(self.device)

        # Reshape for loss computation
        # logits: (B, N, vocab_size) -> (B*N, vocab_size)
        # targets: (B, N) -> (B*N,)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = target_tokens.reshape(-1)

        # Compute cross-entropy loss
        loss = self.loss_fn(logits_flat, targets_flat)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader providing batches of images

        Returns:
            avg_loss: Average loss over the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, dataloader):
        """
        Compute validation loss (no gradient updates).

        Args:
            dataloader: DataLoader for validation data

        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch.to(self.device)
                logits, token_maps = self.model(images)

                target_tokens = self.compute_target_tokens(token_maps, self.model.num_scales)
                target_tokens = target_tokens.to(self.device)

                logits_flat = logits.reshape(-1, logits.shape[-1])
                targets_flat = target_tokens.reshape(-1)

                loss = self.loss_fn(logits_flat, targets_flat)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
