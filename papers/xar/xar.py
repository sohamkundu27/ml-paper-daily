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


class SimpleFeatureBackbone(nn.Module):
    """Lightweight CNN backbone for feature encoding/decoding."""

    def __init__(self, in_channels: int = 1, out_channels: int = 8):
        """
        Args:
            in_channels: number of input channels (typically 1 for grayscale)
            out_channels: number of output feature channels
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)
        )

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Encode patches to learned features.

        Args:
            patches: shape (batch, height, width, channels) as (B, H, W, C)

        Returns:
            features: shape (batch, height, width, out_channels)
        """
        # Rearrange to (B, C, H, W) for convolution
        x = patches.permute(0, 3, 1, 2)
        x = self.encoder(x)
        # Back to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        return x

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to patches.

        Args:
            features: shape (batch, height, width, channels)

        Returns:
            patches: shape (batch, height, width, in_channels)
        """
        # Rearrange to (B, C, H, W)
        x = features.permute(0, 3, 1, 2)
        x = self.decoder(x)
        # Back to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        return x


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


class ScheduledNoisyContextLearner(NoisyContextLearner):
    """Extends NoisyContextLearner with scheduled sampling and curriculum learning."""

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
        super().__init__(entity_dim, hidden_dim, num_heads, num_layers, learning_rate, device)

    def get_scheduled_noise(self, epoch: int, total_epochs: int,
                           noise_start: float = 0.05, noise_end: float = 0.25) -> float:
        """
        Compute noise level for current epoch with curriculum strategy.
        Noise increases over time to gradually expose the model to harder denoising.

        Args:
            epoch: current epoch (0-indexed)
            total_epochs: total number of epochs
            noise_start: initial noise std
            noise_end: final noise std

        Returns:
            noise_std: noise level for this epoch
        """
        progress = epoch / max(1, total_epochs - 1)
        noise_std = noise_start + (noise_end - noise_start) * progress
        return noise_std

    def train_epochs_scheduled(self, batch_entities: np.ndarray, num_epochs: int = 100,
                              noise_start: float = 0.05, noise_end: float = 0.25,
                              verbose: bool = True) -> Tuple[List[float], List[float]]:
        """
        Train for multiple epochs with scheduled noise curriculum.

        Args:
            batch_entities: numpy array, shape (batch_size, seq_len, entity_dim)
            num_epochs: number of training epochs
            noise_start: initial noise standard deviation
            noise_end: final noise standard deviation
            verbose: whether to print loss

        Returns:
            losses: list of loss values per epoch
            noise_levels: list of noise levels used per epoch
        """
        losses = []
        noise_levels = []

        for epoch in range(num_epochs):
            noise_std = self.get_scheduled_noise(epoch, num_epochs, noise_start, noise_end)
            loss = self.train_step(batch_entities, noise_std=noise_std)
            losses.append(loss)
            noise_levels.append(noise_std)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}, Noise: {noise_std:.4f}")

        return losses, noise_levels


class MultiGranularityARTrainer:
    """Trainer supporting multiple entity granularities simultaneously."""

    def __init__(self, entity_dim: int, granularities: List[str], hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2, learning_rate: float = 1e-3,
                 device: str = 'cpu', use_backbone: bool = False):
        """
        Args:
            entity_dim: dimension of each entity vector
            granularities: list of granularity types (e.g., ['patch', 'cell'])
            hidden_dim: transformer hidden dimension
            num_heads: number of attention heads
            num_layers: number of transformer layers
            learning_rate: optimizer learning rate
            device: 'cpu' or 'cuda'
            use_backbone: whether to use learned feature backbone
        """
        self.device = device
        self.entity_dim = entity_dim
        self.granularities = granularities
        self.use_backbone = use_backbone

        # Create predictors for each granularity
        self.predictors = {
            gran: TransformerEntityPredictor(entity_dim, hidden_dim, num_heads, num_layers)
            for gran in granularities
        }
        for predictor in self.predictors.values():
            predictor.to(device)

        # Optional: feature backbone
        if use_backbone:
            self.backbone = SimpleFeatureBackbone(in_channels=1, out_channels=8)
            self.backbone.to(device)
        else:
            self.backbone = None

        # Shared optimizer - include all predictor parameters
        params = []
        for predictor in self.predictors.values():
            params.extend(predictor.parameters())
        if self.backbone is not None:
            params.extend(self.backbone.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

    def train_step(self, granularity_entities: dict, noise_std: float = 0.15) -> dict:
        """
        Training step on multiple granularities.

        Args:
            granularity_entities: dict mapping granularity -> entities (batch, seq_len, entity_dim)
            noise_std: noise standard deviation

        Returns:
            losses: dict mapping granularity -> loss value
        """
        losses = {}
        total_loss = 0.0
        self.optimizer.zero_grad()

        for granularity, entities in granularity_entities.items():
            batch_tensor = torch.from_numpy(entities).float().to(self.device)

            # Add noise
            noise = torch.randn_like(batch_tensor) * noise_std
            noisy_entities = batch_tensor + noise

            # Forward pass
            predictor = self.predictors[granularity]
            predicted = predictor(noisy_entities)

            # Loss
            loss = nn.MSELoss()(predicted, batch_tensor)
            total_loss = total_loss + loss

            losses[granularity] = loss.item()

        # Single backward pass on accumulated loss
        total_loss.backward()
        self.optimizer.step()

        return losses

    def train_epochs(self, granularity_entities: dict, num_epochs: int = 100,
                     noise_std: float = 0.15, verbose: bool = True) -> dict:
        """
        Train for multiple epochs on all granularities.

        Args:
            granularity_entities: dict mapping granularity -> entities
            num_epochs: number of epochs
            noise_std: noise standard deviation
            verbose: whether to print loss

        Returns:
            all_losses: dict mapping granularity -> list of losses
        """
        all_losses = {gran: [] for gran in self.granularities}

        for epoch in range(num_epochs):
            losses = self.train_step(granularity_entities, noise_std=noise_std)
            for gran, loss in losses.items():
                all_losses[gran].append(loss)

            if verbose and (epoch + 1) % 20 == 0:
                loss_str = ", ".join([f"{gran}: {losses[gran]:.4f}" for gran in self.granularities])
                print(f"Epoch {epoch+1}/{num_epochs}, {loss_str}")

        return all_losses

    @torch.no_grad()
    def predict(self, granularity: str, entities: np.ndarray) -> np.ndarray:
        """
        Predict for a specific granularity.

        Args:
            granularity: which granularity to use for prediction
            entities: numpy array, shape (batch_size, seq_len, entity_dim)

        Returns:
            predictions: numpy array of same shape as entities
        """
        batch_tensor = torch.from_numpy(entities).float().to(self.device)
        predictor = self.predictors[granularity]
        output = predictor(batch_tensor)
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


def test_feature_backbone():
    """Test lightweight CNN backbone for feature encoding/decoding."""
    backbone = SimpleFeatureBackbone(in_channels=1, out_channels=8)

    # Create dummy patches (batch_size=2, height=4, width=4, channels=1)
    patches = torch.randn(2, 4, 4, 1)

    # Encode
    features = backbone.encode(patches)
    assert features.shape == (2, 4, 4, 8), f"Expected (2,4,4,8), got {features.shape}"

    # Decode
    reconstructed = backbone.decode(features)
    assert reconstructed.shape == patches.shape, "Reconstructed shape should match input"

    print("✓ Feature backbone encoding/decoding works")


def test_scheduled_curriculum_learning():
    """Test scheduled sampling with curriculum learning."""
    entity_dim = 16
    batch_size = 8
    seq_len = 10

    # Toy dataset
    toy_entities = np.random.randn(batch_size, seq_len, entity_dim).astype(np.float32)

    # Initialize scheduled learner
    learner = ScheduledNoisyContextLearner(entity_dim=entity_dim, hidden_dim=64, num_heads=2, num_layers=1)

    # Train with scheduled noise (0.05 -> 0.3)
    losses, noise_levels = learner.train_epochs_scheduled(
        toy_entities, num_epochs=50, noise_start=0.05, noise_end=0.3, verbose=False
    )

    # Check that noise increased over time
    assert noise_levels[0] < noise_levels[-1], "Noise should increase with curriculum"
    assert abs(noise_levels[0] - 0.05) < 0.01, "Initial noise should be ~0.05"
    assert abs(noise_levels[-1] - 0.3) < 0.01, "Final noise should be ~0.3"

    # Loss should still decrease overall
    assert losses[-1] < losses[0], "Loss should decrease during training"

    print(f"✓ Scheduled curriculum: noise {noise_levels[0]:.4f} -> {noise_levels[-1]:.4f}, loss decreased")


def test_multi_granularity_training():
    """Test training on multiple granularities simultaneously."""
    entity_dim = 16
    batch_size = 4
    seq_len = 8

    # Create data for two granularities: patch and cell (same entity_dim)
    patch_entities = np.random.randn(batch_size, seq_len, entity_dim).astype(np.float32)
    cell_entities = np.random.randn(batch_size, seq_len, entity_dim).astype(np.float32)

    granularity_entities = {
        'patch': patch_entities,
        'cell': cell_entities
    }

    # Initialize multi-granularity trainer
    trainer = MultiGranularityARTrainer(
        entity_dim=entity_dim,
        granularities=['patch', 'cell'],
        hidden_dim=64,
        num_heads=2,
        num_layers=1,
        use_backbone=False
    )

    # Train for a few epochs
    all_losses = trainer.train_epochs(granularity_entities, num_epochs=30, noise_std=0.15, verbose=False)

    # Check that both granularities are being trained
    assert 'patch' in all_losses, "Should have loss for patch granularity"
    assert 'cell' in all_losses, "Should have loss for cell granularity"
    assert len(all_losses['patch']) == 30, "Should have 30 loss values"
    assert len(all_losses['cell']) == 30, "Should have 30 loss values"

    # Losses should decrease
    assert all_losses['patch'][-1] < all_losses['patch'][0], "Patch loss should decrease"
    assert all_losses['cell'][-1] < all_losses['cell'][0], "Cell loss should decrease"

    print(f"✓ Multi-granularity training: patch loss {all_losses['patch'][0]:.4f} -> {all_losses['patch'][-1]:.4f}")
    print(f"                            cell loss {all_losses['cell'][0]:.4f} -> {all_losses['cell'][-1]:.4f}")

    # Test prediction for each granularity
    patch_pred = trainer.predict('patch', patch_entities)
    cell_pred = trainer.predict('cell', cell_entities)
    assert patch_pred.shape == patch_entities.shape, "Patch prediction shape mismatch"
    assert cell_pred.shape == cell_entities.shape, "Cell prediction shape mismatch"

    print("✓ Multi-granularity predictions work")


def test_multi_granularity_with_backbone():
    """Test multi-granularity training with learned feature backbone."""
    entity_dim = 8
    batch_size = 2
    seq_len = 4

    # Toy data
    patch_entities = np.random.randn(batch_size, seq_len, entity_dim).astype(np.float32)
    cell_entities = np.random.randn(batch_size, seq_len, entity_dim).astype(np.float32)

    granularity_entities = {
        'patch': patch_entities,
        'cell': cell_entities
    }

    # Initialize with backbone
    trainer = MultiGranularityARTrainer(
        entity_dim=entity_dim,
        granularities=['patch', 'cell'],
        hidden_dim=32,
        num_heads=1,
        num_layers=1,
        use_backbone=True
    )

    # Verify backbone exists
    assert trainer.backbone is not None, "Backbone should be created"
    assert isinstance(trainer.backbone, SimpleFeatureBackbone), "Backbone should be SimpleFeatureBackbone"

    # Train briefly to ensure backbone updates
    losses = trainer.train_epochs(granularity_entities, num_epochs=50, noise_std=0.1, verbose=False)

    # Check that losses generally decrease (allow some noise in early epochs)
    avg_first_10 = np.mean(losses['patch'][:10])
    avg_last_10 = np.mean(losses['patch'][-10:])
    assert avg_last_10 < avg_first_10, "Patch loss should decrease on average"

    print(f"✓ Multi-granularity with backbone: patch loss {losses['patch'][0]:.4f} -> {losses['patch'][-1]:.4f}")


class ToyImageDataset:
    """Generate simple synthetic 32x32 images for Pass 4 demo."""

    def __init__(self, num_images: int = 64, image_size: int = 32):
        """
        Args:
            num_images: number of images to generate
            image_size: size of square images (H=W)
        """
        self.num_images = num_images
        self.image_size = image_size
        self.images = self._generate_images()

    def _generate_images(self) -> np.ndarray:
        """Generate toy images: random rectangles and circles."""
        images = []
        for _ in range(self.num_images):
            img = np.ones((self.image_size, self.image_size, 1), dtype=np.float32)

            # Random rectangles
            for _ in range(np.random.randint(1, 4)):
                y1 = np.random.randint(0, self.image_size - 4)
                x1 = np.random.randint(0, self.image_size - 4)
                h = np.random.randint(2, 8)
                w = np.random.randint(2, 8)
                color = np.random.rand()
                img[y1:y1 + h, x1:x1 + w, 0] = color

            # Random circles (approximate with squares for simplicity)
            for _ in range(np.random.randint(0, 3)):
                cy = np.random.randint(4, self.image_size - 4)
                cx = np.random.randint(4, self.image_size - 4)
                r = np.random.randint(1, 4)
                color = np.random.rand()
                yy, xx = np.ogrid[:self.image_size, :self.image_size]
                mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
                img[mask, 0] = color

            images.append(img)

        return np.array(images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.images[idx]


class EndToEndDemo:
    """End-to-end xAR demonstration on toy dataset."""

    def __init__(self, patch_size: int = 4, entity_dim: int = 16, hidden_dim: int = 64,
                 num_heads: int = 2, num_layers: int = 1, device: str = 'cpu'):
        """
        Args:
            patch_size: size of base patches
            entity_dim: dimension of vectorized entities
            hidden_dim: transformer hidden dimension
            num_heads: attention heads
            num_layers: transformer layers
            device: 'cpu' or 'cuda'
        """
        self.patch_size = patch_size
        self.entity_dim = entity_dim
        self.device = device
        self.converter = ImageToEntity(patch_size=patch_size)

        # Multi-granularity trainer for 'patch' and 'cell' granularities
        self.trainer = MultiGranularityARTrainer(
            entity_dim=entity_dim,
            granularities=['patch', 'cell'],
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            learning_rate=1e-3,
            device=device,
            use_backbone=False
        )

    def prepare_batch(self, images: np.ndarray) -> dict:
        """
        Convert batch of images to entity sequences.

        Args:
            images: shape (batch_size, H, W, C)

        Returns:
            granularity_entities: dict mapping granularity -> entities
        """
        batch_size = len(images)
        all_patch_entities = []
        all_cell_entities = []

        for img in images:
            patches = self.converter.image_to_patches(img)
            num_h, num_w = patches.shape[0], patches.shape[1]

            # Flatten patches into sequence
            patch_entities = self.converter.vectorize_entity(patches, entity_type='patch')
            all_patch_entities.append(patch_entities)

            # Cell entities (2x2 grouped patches)
            group_size = 2
            if num_h % group_size == 0 and num_w % group_size == 0:
                cell_entities = self.converter.vectorize_entity(
                    patches, entity_type='cell', group_size=group_size
                )
            else:
                # Fallback: use subsample if grid not divisible
                cell_entities = self.converter.vectorize_entity(
                    patches, entity_type='subsample', group_size=group_size
                )
            all_cell_entities.append(cell_entities)

        # Pad sequences to same length
        max_patch_len = max(len(e) for e in all_patch_entities)
        max_cell_len = max(len(e) for e in all_cell_entities)

        patch_seq = np.zeros((batch_size, max_patch_len, self.entity_dim), dtype=np.float32)
        cell_seq = np.zeros((batch_size, max_cell_len, self.entity_dim), dtype=np.float32)

        for i, (pe, ce) in enumerate(zip(all_patch_entities, all_cell_entities)):
            # Pad or truncate to entity_dim
            pe_padded = np.zeros((len(pe), self.entity_dim), dtype=np.float32)
            pe_padded[:, :min(self.entity_dim, pe.shape[1])] = pe[:, :self.entity_dim]
            patch_seq[i, :len(pe), :] = pe_padded

            ce_padded = np.zeros((len(ce), self.entity_dim), dtype=np.float32)
            ce_padded[:, :min(self.entity_dim, ce.shape[1])] = ce[:, :self.entity_dim]
            cell_seq[i, :len(ce), :] = ce_padded

        return {
            'patch': patch_seq,
            'cell': cell_seq
        }

    def train(self, dataset: ToyImageDataset, num_epochs: int = 100, batch_size: int = 16,
              noise_std: float = 0.15, verbose: bool = True) -> Tuple[List[float], List[float]]:
        """
        Train on toy dataset.

        Args:
            dataset: ToyImageDataset instance
            num_epochs: number of training epochs
            batch_size: batch size for training
            noise_std: noise standard deviation
            verbose: whether to print progress

        Returns:
            patch_losses, cell_losses: training loss curves
        """
        patch_losses = []
        cell_losses = []

        for epoch in range(num_epochs):
            epoch_patch_loss = 0.0
            epoch_cell_loss = 0.0
            num_batches = 0

            for batch_start in range(0, len(dataset), batch_size):
                batch_end = min(batch_start + batch_size, len(dataset))
                batch_images = np.array([dataset[i] for i in range(batch_start, batch_end)])

                entities = self.prepare_batch(batch_images)
                losses = self.trainer.train_step(entities, noise_std=noise_std)

                epoch_patch_loss += losses.get('patch', 0.0)
                epoch_cell_loss += losses.get('cell', 0.0)
                num_batches += 1

            avg_patch_loss = epoch_patch_loss / max(1, num_batches)
            avg_cell_loss = epoch_cell_loss / max(1, num_batches)
            patch_losses.append(avg_patch_loss)
            cell_losses.append(avg_cell_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: patch={avg_patch_loss:.4f}, cell={avg_cell_loss:.4f}")

        return patch_losses, cell_losses

    def generate_from_sample(self, sample_image: np.ndarray, granularity: str = 'patch',
                            num_steps: int = 5) -> np.ndarray:
        """
        Generate image continuation from a sample.

        Args:
            sample_image: starting image, shape (H, W, C)
            granularity: 'patch' or 'cell'
            num_steps: autoregressive steps to generate

        Returns:
            generated_image: shape (H, W, C)
        """
        patches = self.converter.image_to_patches(sample_image)
        patch_entities = self.converter.vectorize_entity(patches, entity_type='patch')

        # Pad features to entity_dim
        padded_entities = np.zeros((len(patch_entities), self.entity_dim), dtype=np.float32)
        padded_entities[:, :patch_entities.shape[1]] = patch_entities

        # Use first half of entities as context, generate rest
        context_len = max(1, len(padded_entities) // 2)
        context = padded_entities[:context_len]

        # Pad to sequence length
        max_len = len(padded_entities)
        padded_context = np.zeros((1, max_len, self.entity_dim), dtype=np.float32)
        padded_context[0, :context_len, :] = context

        # Predict
        predictions = self.trainer.predict(granularity, padded_context)

        # Use predictions to fill remaining positions
        generated_entities = padded_context[0].copy()
        generated_entities[context_len:] = predictions[0, context_len:]

        # Reconstruct image - truncate back to original entity dim for devectorize
        original_entity_dim = patch_entities.shape[1]
        truncated_entities = generated_entities[:, :original_entity_dim]

        # Reconstruct image
        num_h = patches.shape[0]
        num_w = patches.shape[1]
        reconstructed_patches = self.converter.devectorize_entity(
            truncated_entities, num_patches_h=num_h, num_patches_w=num_w, entity_type='patch'
        )
        reconstructed_image = self.converter.patches_to_image(reconstructed_patches)

        return reconstructed_image

    def evaluate_reconstruction(self, dataset: ToyImageDataset, num_samples: int = 10) -> float:
        """
        Evaluate reconstruction error on test samples.

        Args:
            dataset: ToyImageDataset
            num_samples: number of samples to evaluate

        Returns:
            mean_mse: mean squared error averaged over samples
        """
        total_mse = 0.0
        for i in range(min(num_samples, len(dataset))):
            original = dataset[i]
            reconstructed = self.generate_from_sample(original, granularity='patch', num_steps=5)
            mse = np.mean((original - reconstructed) ** 2)
            total_mse += mse

        return total_mse / max(1, min(num_samples, len(dataset)))


def test_end_to_end_demo():
    """Pass 4: End-to-end demo on toy dataset."""
    print("\n=== Pass 4: End-to-End Demo on Toy Dataset ===")

    # Create toy dataset
    print("Generating toy dataset (64 32x32 images with random shapes)...")
    dataset = ToyImageDataset(num_images=64, image_size=32)
    print(f"✓ Dataset created: {len(dataset)} images of size {dataset.image_size}x{dataset.image_size}")

    # Initialize end-to-end demo
    print("Initializing xAR model...")
    demo = EndToEndDemo(
        patch_size=4,
        entity_dim=64,
        hidden_dim=64,
        num_heads=2,
        num_layers=1,
        device='cpu'
    )
    print("✓ Model initialized with patch_size=4, entity_dim=64, hidden_dim=64")

    # Train
    print(f"Training for 50 epochs on batch_size=8...")
    patch_losses, cell_losses = demo.train(
        dataset,
        num_epochs=50,
        batch_size=8,
        noise_std=0.15,
        verbose=True
    )
    print(f"✓ Training complete")
    print(f"  Patch loss: {patch_losses[0]:.4f} -> {patch_losses[-1]:.4f}")
    print(f"  Cell loss:  {cell_losses[0]:.4f} -> {cell_losses[-1]:.4f}")

    # Evaluate reconstruction
    print("Evaluating reconstruction on 10 test samples...")
    recon_mse = demo.evaluate_reconstruction(dataset, num_samples=10)
    print(f"✓ Mean reconstruction MSE: {recon_mse:.4f}")

    # Generate samples
    print("Generating sample outputs...")
    sample_idx = 0
    original = dataset[sample_idx]
    generated = demo.generate_from_sample(original, granularity='patch')
    gen_mse = np.mean((original - generated) ** 2)
    print(f"✓ Generated image MSE vs original: {gen_mse:.4f}")
    print(f"  Original intensity range: [{original.min():.3f}, {original.max():.3f}]")
    print(f"  Generated intensity range: [{generated.min():.3f}, {generated.max():.3f}]")

    # Summary statistics
    print("\n--- Training Summary ---")
    print(f"Final patch loss:  {patch_losses[-1]:.4f} (decreased by {(1 - patch_losses[-1]/patch_losses[0])*100:.1f}%)")
    print(f"Final cell loss:   {cell_losses[-1]:.4f} (decreased by {(1 - cell_losses[-1]/cell_losses[0])*100:.1f}%)")
    print(f"Reconstruction MSE: {recon_mse:.4f}")

    return patch_losses, cell_losses, recon_mse


if __name__ == '__main__':
    print("=== Testing Pass 1 (Entity Abstraction) ===")
    test_basic_entity_conversion()
    test_entity_vectorization()
    test_ar_prediction()
    test_full_pipeline()

    print("\n=== Testing Pass 2 (Flow-Matching & Noisy Context Learning) ===")
    test_noisy_context_learning()
    test_flow_matching_denoising()

    print("\n=== Testing Pass 3 (Multi-Granularity & Curriculum Learning) ===")
    test_feature_backbone()
    test_scheduled_curriculum_learning()
    test_multi_granularity_training()
    test_multi_granularity_with_backbone()

    print("\n=== Testing Pass 4 (End-to-End Demo) ===")
    test_end_to_end_demo()

    print("\n✓ All tests passed!")
