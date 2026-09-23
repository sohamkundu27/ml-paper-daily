import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


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


if __name__ == '__main__':
    test_basic_entity_conversion()
    test_entity_vectorization()
    test_ar_prediction()
    test_full_pipeline()
    print("\n✓ All tests passed!")
