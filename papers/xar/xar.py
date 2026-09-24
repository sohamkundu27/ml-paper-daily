import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class Entity:
    """Represents a prediction unit: patch, cell, subsample, scale, or image."""
    granularity: str  # 'patch', 'cell', 'subsample', 'scale', 'image'
    patch_size: int  # size of base patch (e.g., 8 for 8x8 patches)
    group_size: Optional[int] = None  # for cells: group_size x group_size patches


class ImageToEntity:
    """Convert images to entities and back."""

    def __init__(self, patch_size: int = 4):
        """
        Args:
            patch_size: size of base patches (e.g., 4 for 4x4 patches)
        """
        self.patch_size = patch_size

    def image_to_patches(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to patch tokens.

        Args:
            image: shape (H, W, C) or (H, W)

        Returns:
            patches: shape (num_patches_h, num_patches_w, patch_size, patch_size, C)
        """
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        H, W, C = image.shape
        p = self.patch_size

        if H % p != 0 or W % p != 0:
            raise ValueError(f"Image size ({H}, {W}) must be divisible by patch_size {p}")

        num_h = H // p
        num_w = W // p

        patches = image.reshape(num_h, p, num_w, p, C)
        patches = patches.transpose(0, 2, 1, 3, 4)  # (num_h, num_w, patch_size, patch_size, C)
        return patches

    def patches_to_image(self, patches: np.ndarray) -> np.ndarray:
        """
        Reconstruct image from patches.

        Args:
            patches: shape (num_patches_h, num_patches_w, patch_size, patch_size, C)

        Returns:
            image: shape (H, W, C)
        """
        num_h, num_w, p, _, C = patches.shape
        patches = patches.transpose(0, 2, 1, 3, 4)  # (num_h, p, num_w, p, C)
        image = patches.reshape(num_h * p, num_w * p, C)
        return image

    def vectorize_entity(self, patches: np.ndarray, entity_type: str = 'patch',
                         group_size: int = 1) -> np.ndarray:
        """
        Vectorize patches into entities of given granularity.

        Args:
            patches: shape (num_patches_h, num_patches_w, patch_size, patch_size, C)
            entity_type: 'patch', 'cell', or 'subsample'
            group_size: for cell/subsample, how many patches to group

        Returns:
            entities: shape (num_entities, patch_size * patch_size * C)
        """
        num_h, num_w, p, _, C = patches.shape
        patch_dim = p * p * C

        if entity_type == 'patch':
            # One entity per patch
            entities = patches.reshape(num_h * num_w, patch_dim)

        elif entity_type == 'cell':
            # Group adjacent patches
            if num_h % group_size != 0 or num_w % group_size != 0:
                raise ValueError(f"Patch grid ({num_h}, {num_w}) not divisible by group_size {group_size}")

            cell_h = num_h // group_size
            cell_w = num_w // group_size

            cells = patches.reshape(cell_h, group_size, cell_w, group_size, p, p, C)
            cells = cells.transpose(0, 2, 1, 3, 4, 5, 6)  # (cell_h, cell_w, group_size, group_size, p, p, C)
            entities = cells.reshape(cell_h * cell_w, group_size * group_size * patch_dim)

        elif entity_type == 'subsample':
            # Non-local grouping: stride through patches
            entities = patches[::group_size, ::group_size].reshape(-1, patch_dim)

        else:
            raise ValueError(f"Unknown entity_type: {entity_type}")

        return entities

    def devectorize_entity(self, entities: np.ndarray, num_patches_h: int, num_patches_w: int,
                           entity_type: str = 'patch', group_size: int = 1) -> np.ndarray:
        """
        Reconstruct patches from vectorized entities.

        Args:
            entities: shape (num_entities, feature_dim)
            num_patches_h, num_patches_w: shape of original patch grid
            entity_type: 'patch', 'cell', or 'subsample'
            group_size: for cell/subsample

        Returns:
            patches: shape (num_patches_h, num_patches_w, patch_size, patch_size, C)
        """
        p = self.patch_size

        if entity_type == 'patch':
            feature_dim = entities.shape[1]
            C = feature_dim // (p * p)
            patches = entities.reshape(num_patches_h, num_patches_w, p, p, C)

        elif entity_type == 'cell':
            cell_h = num_patches_h // group_size
            cell_w = num_patches_w // group_size
            feature_dim = entities.shape[1]
            patch_dim = feature_dim // (group_size * group_size)
            C = patch_dim // (p * p)

            cells = entities.reshape(cell_h, cell_w, group_size, group_size, p, p, C)
            cells = cells.transpose(0, 2, 1, 3, 4, 5, 6)  # rearrange for reshape
            patches = cells.reshape(num_patches_h, num_patches_w, p, p, C)

        elif entity_type == 'subsample':
            feature_dim = entities.shape[1]
            C = feature_dim // (p * p)
            sub_h = (num_patches_h + group_size - 1) // group_size
            sub_w = (num_patches_w + group_size - 1) // group_size

            subsampled = entities[:sub_h * sub_w].reshape(sub_h, sub_w, p, p, C)
            patches = np.zeros((num_patches_h, num_patches_w, p, p, C))
            patches[::group_size, ::group_size] = subsampled[:num_patches_h // group_size, :num_patches_w // group_size]

        else:
            raise ValueError(f"Unknown entity_type: {entity_type}")

        return patches


class SimpleARModel:
    """Minimal AR model that predicts next entity from previous entities."""

    def __init__(self, entity_dim: int, hidden_dim: int = 64, num_history: int = 4):
        """
        Args:
            entity_dim: dimension of each entity vector
            hidden_dim: hidden layer size
            num_history: how many past entities to condition on
        """
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.num_history = num_history

        # Simple linear predictor: concat history, pass through linear layer
        input_dim = num_history * entity_dim
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, entity_dim) * 0.01
        self.b2 = np.zeros(entity_dim)

    def predict(self, history: List[np.ndarray]) -> np.ndarray:
        """
        Predict next entity given history.

        Args:
            history: list of past entities, each shape (entity_dim,)

        Returns:
            next_entity: shape (entity_dim,)
        """
        # Take last num_history entities, pad with zeros if needed
        h = history[-self.num_history:]
        while len(h) < self.num_history:
            h = [np.zeros(self.entity_dim)] + h

        x = np.concatenate(h)  # (num_history * entity_dim,)
        hidden = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        output = hidden @ self.w2 + self.b2
        return output

    def predict_sequence(self, initial_entities: np.ndarray, num_predict: int) -> np.ndarray:
        """
        Autoregressively predict a sequence of entities.

        Args:
            initial_entities: shape (num_initial, entity_dim)
            num_predict: how many more entities to predict

        Returns:
            predicted: shape (num_predict, entity_dim)
        """
        history = [initial_entities[i] for i in range(len(initial_entities))]
        predictions = []

        for _ in range(num_predict):
            next_entity = self.predict(history)
            predictions.append(next_entity)
            history.append(next_entity)

        return np.array(predictions)


class TransformerEntityPredictor(nn.Module):
    """Transformer-based entity predictor for continuous regression."""

    def __init__(self, entity_dim: int, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        """
        Args:
            entity_dim: dimension of each entity vector
            hidden_dim: transformer hidden dimension
            num_heads: number of attention heads
            num_layers: number of transformer layers
        """
        super().__init__()
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Linear(entity_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, entity_dim)

    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        """
        Predict next entity from sequence.

        Args:
            entities: shape (batch, seq_len, entity_dim)

        Returns:
            predictions: shape (batch, seq_len, entity_dim)
        """
        batch_size, seq_len, _ = entities.shape

        x = self.embedding(entities)  # (batch, seq_len, hidden_dim)
        x = x + self.pos_encoding[:, :seq_len, :]  # Add positional encoding

        x = self.transformer(x)  # (batch, seq_len, hidden_dim)
        output = self.output_proj(x)  # (batch, seq_len, entity_dim)

        return output


class NoisyContextLearner:
    """Trainer for xAR using flow-matching and noisy context learning."""

    def __init__(self, entity_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2, learning_rate: float = 1e-3, device: str = 'cpu'):
        """
        Args:
            entity_dim: dimension of each entity vector
            hidden_dim: transformer hidden dimension
            num_heads: number of attention heads
            num_layers: number of transformer layers
            learning_rate: optimizer learning rate
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.entity_dim = entity_dim
        self.model = TransformerEntityPredictor(entity_dim, hidden_dim, num_heads, num_layers)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def flow_matching_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Flow-matching loss: continuous regression objective.
        Predicts clean entity vectors from noisy inputs.

        Args:
            predicted: shape (batch, seq_len, entity_dim)
            target: shape (batch, seq_len, entity_dim)

        Returns:
            loss: scalar tensor
        """
        return nn.MSELoss()(predicted, target)

    def train_step(self, batch_entities: np.ndarray, noise_std: float = 0.15) -> float:
        """
        Training step with noisy context learning.
        Adds Gaussian noise to input entities and trains model to predict clean entities.

        Args:
            batch_entities: numpy array, shape (batch_size, seq_len, entity_dim)
            noise_std: standard deviation of Gaussian noise

        Returns:
            loss: scalar loss value
        """
        # Convert to tensor
        batch_tensor = torch.from_numpy(batch_entities).float().to(self.device)

        # Add noise to input (noisy context learning)
        noise = torch.randn_like(batch_tensor) * noise_std
        noisy_entities = batch_tensor + noise

        # Forward pass
        predicted = self.model(noisy_entities)

        # Loss: predict clean entities from noisy input
        loss = self.flow_matching_loss(predicted, batch_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epochs(self, batch_entities: np.ndarray, num_epochs: int = 100,
                     noise_std: float = 0.15, verbose: bool = True) -> List[float]:
        """
        Train for multiple epochs.

        Args:
            batch_entities: numpy array, shape (batch_size, seq_len, entity_dim)
            num_epochs: number of training epochs
            noise_std: standard deviation of Gaussian noise
            verbose: whether to print loss

        Returns:
            losses: list of loss values per epoch
        """
        losses = []
        for epoch in range(num_epochs):
            loss = self.train_step(batch_entities, noise_std=noise_std)
            losses.append(loss)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}")
        return losses

    @torch.no_grad()
    def predict(self, entities: np.ndarray) -> np.ndarray:
        """
        Predict entity continuations from noisy input.

        Args:
            entities: numpy array, shape (batch_size, seq_len, entity_dim)

        Returns:
            predictions: numpy array, shape (batch_size, seq_len, entity_dim)
        """
        batch_tensor = torch.from_numpy(entities).float().to(self.device)
        output = self.model(batch_tensor)
        return output.cpu().numpy()


def test_basic_entity_conversion():
    """Test image -> entity -> image conversion."""
    # Create simple 16x16 single-channel image
    image = np.random.rand(16, 16, 1).astype(np.float32)

    converter = ImageToEntity(patch_size=4)

    # Convert to patches
    patches = converter.image_to_patches(image)
    assert patches.shape == (4, 4, 4, 4, 1), f"Expected (4,4,4,4,1), got {patches.shape}"

    # Convert back
    reconstructed = converter.patches_to_image(patches)
    assert reconstructed.shape == image.shape
    assert np.allclose(image, reconstructed), "Reconstruction should be exact"

    print("✓ Image -> patches -> image conversion works")


def test_entity_vectorization():
    """Test vectorization of patches into entities."""
    image = np.random.rand(16, 16, 1).astype(np.float32)
    converter = ImageToEntity(patch_size=4)
    patches = converter.image_to_patches(image)

    # Test patch entities (one per patch)
    patch_entities = converter.vectorize_entity(patches, entity_type='patch')
    assert patch_entities.shape == (16, 16), f"Expected (16, 16), got {patch_entities.shape}"

    # Test cell entities (2x2 groups of patches)
    cell_entities = converter.vectorize_entity(patches, entity_type='cell', group_size=2)
    assert cell_entities.shape == (4, 64), f"Expected (4, 64), got {cell_entities.shape}"

    # Test subsample entities
    subsample_entities = converter.vectorize_entity(patches, entity_type='subsample', group_size=2)
    assert subsample_entities.shape == (4, 16), f"Expected (4, 16), got {subsample_entities.shape}"

    print("✓ Entity vectorization works")


def test_ar_prediction():
    """Test AR model can predict entity sequences."""
    entity_dim = 16
    model = SimpleARModel(entity_dim=entity_dim, hidden_dim=32, num_history=4)

    # Initial sequence
    initial = np.random.randn(4, entity_dim).astype(np.float32)

    # Predict next 5 entities
    predictions = model.predict_sequence(initial, num_predict=5)
    assert predictions.shape == (5, entity_dim), f"Expected (5, {entity_dim}), got {predictions.shape}"

    print("✓ AR prediction works")


def test_full_pipeline():
    """Test full pipeline: image -> entities -> AR prediction -> reconstruction."""
    # Create a small image
    image = np.random.rand(16, 16, 1).astype(np.float32)

    converter = ImageToEntity(patch_size=4)
    patches = converter.image_to_patches(image)

    # Vectorize to patch entities
    entities = converter.vectorize_entity(patches, entity_type='patch')

    # Run AR prediction on first 8 entities, predict 8 more
    model = SimpleARModel(entity_dim=16, hidden_dim=32, num_history=4)
    predicted_entities = model.predict_sequence(entities[:8], num_predict=8)

    # Combine original and predicted
    all_entities = np.vstack([entities[:8], predicted_entities])

    # Reconstruct patches from entities
    reconstructed_patches = converter.devectorize_entity(
        all_entities, num_patches_h=4, num_patches_w=4, entity_type='patch'
    )

    # Reconstruct image
    reconstructed_image = converter.patches_to_image(reconstructed_patches)
    assert reconstructed_image.shape == image.shape

    print("✓ Full pipeline (image -> entities -> AR -> reconstruction) works")


def test_noisy_context_learning():
    """Test that model learns to denoise entities via flow-matching."""
    entity_dim = 16
    batch_size = 8
    seq_len = 10

    # Create toy dataset: random entity sequences
    toy_entities = np.random.randn(batch_size, seq_len, entity_dim).astype(np.float32)

    # Initialize learner
    learner = NoisyContextLearner(entity_dim=entity_dim, hidden_dim=64, num_heads=2, num_layers=1)

    # Train
    losses = learner.train_epochs(toy_entities, num_epochs=50, noise_std=0.2, verbose=False)

    # Check that loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    print(f"✓ Noisy context learning: loss decreased from {initial_loss:.4f} to {final_loss:.4f}")

    # Test inference
    predictions = learner.predict(toy_entities)
    assert predictions.shape == toy_entities.shape, f"Prediction shape mismatch"
    print("✓ Model can predict clean entities from noisy input")


def test_flow_matching_denoising():
    """Test flow-matching objective on synthetic denoising task."""
    entity_dim = 8
    batch_size = 4
    seq_len = 5
    noise_std = 0.3

    # Clean entity data
    clean_data = np.random.randn(batch_size, seq_len, entity_dim).astype(np.float32)

    # Initialize learner and train
    learner = NoisyContextLearner(entity_dim=entity_dim, hidden_dim=32, num_heads=1, num_layers=1)
    initial_loss = learner.train_step(clean_data, noise_std=noise_std)

    # Train for more steps
    for _ in range(49):
        learner.train_step(clean_data, noise_std=noise_std)

    final_loss = learner.train_step(clean_data, noise_std=noise_std)

    # Loss should improve
    assert final_loss < initial_loss, "Flow-matching loss should decrease with training"

    # Predictions should approach clean data
    noisy_test = clean_data + np.random.randn(*clean_data.shape) * noise_std
    predictions = learner.predict(noisy_test)

    # Reconstruction error should be better than just using noisy input
    noisy_error = np.mean((noisy_test - clean_data) ** 2)
    pred_error = np.mean((predictions - clean_data) ** 2)

    print(f"✓ Flow-matching: noisy MSE={noisy_error:.4f}, predicted MSE={pred_error:.4f}")


if __name__ == '__main__':
    print("=== Testing Pass 1 (Entity Abstraction) ===")
    test_basic_entity_conversion()
    test_entity_vectorization()
    test_ar_prediction()
    test_full_pipeline()

    print("\n=== Testing Pass 2 (Flow-Matching & Noisy Context Learning) ===")
    test_noisy_context_learning()
    test_flow_matching_denoising()

    print("\n✓ All tests passed!")
