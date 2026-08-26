import torch
import torch.nn as nn
import torch.optim as optim
from voxel_advection import VoxelGrid, FeatureAdvection


class ActionConditionedFlowPredictor(nn.Module):
    """
    Predicts 3D velocity fields conditioned on both voxel features and action inputs.
    This is the key extension in Pass 3: velocity is no longer position-only, but
    also depends on what action is being applied.
    """

    def __init__(self, grid_size, feature_dim, action_dim=3):
        """
        Args:
            grid_size: size of the cubic voxel grid
            feature_dim: dimension of features stored in voxels
            action_dim: dimension of action vectors (default: 3D force/direction)
        """
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        # Encode features into a latent space
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Encode action into latent space
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # Encode position (normalized coordinates)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # Combine all information to predict velocity
        # latent_dim = 32 (features) + 16 (action) + 16 (position) = 64
        self.velocity_head = nn.Sequential(
            nn.Linear(32 + 16 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, coords, voxel_features, actions):
        """
        Predict 3D velocity at given coordinates, conditioned on voxel features and actions.

        Args:
            coords: (B, N, 3) coordinates in [0, grid_size-1]
            voxel_features: (B, N, feature_dim) features sampled at those coordinates
            actions: (B, action_dim) action vectors for each batch item

        Returns:
            velocity: (B, N, 3) predicted velocity vectors
        """
        B, N, _ = coords.shape

        # Normalize coordinates to [0, 1]
        coords_normalized = coords / (self.grid_size - 1)

        # Encode each component
        feat_encoded = self.feature_encoder(voxel_features)  # (B, N, 32)
        pos_encoded = self.position_encoder(coords_normalized)  # (B, N, 16)

        # Expand action to match all points in the batch
        action_encoded = self.action_encoder(actions)  # (B, 16)
        action_expanded = action_encoded.unsqueeze(1).expand(B, N, -1)  # (B, N, 16)

        # Concatenate all encoded information
        combined = torch.cat([feat_encoded, action_expanded, pos_encoded], dim=-1)  # (B, N, 64)

        # Reshape for MLP
        combined_flat = combined.reshape(B * N, -1)
        velocity = self.velocity_head(combined_flat)
        velocity = velocity.reshape(B, N, 3)

        return velocity


class ActionConditionedAdvection(nn.Module):
    """
    Feature advection that is conditioned on actions.
    Extends FeatureAdvection to use ActionConditionedFlowPredictor instead of
    a simple position-based velocity predictor.
    """

    def __init__(self, grid_size, feature_dim, action_dim=3):
        """
        Args:
            grid_size: size of the cubic voxel grid
            feature_dim: dimension of voxel features
            action_dim: dimension of action vectors
        """
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.voxel_grid = VoxelGrid(grid_size, feature_dim)
        self.flow_predictor = ActionConditionedFlowPredictor(grid_size, feature_dim, action_dim)

    def forward(self, coords, actions, dt=0.1):
        """
        Advect features forward in time conditioned on actions.

        Args:
            coords: (B, N, 3) current coordinates in voxel space
            actions: (B, action_dim) action vectors for each batch item
            dt: timestep size

        Returns:
            advected_features: (B, N, feature_dim) features at new locations
            new_coords: (B, N, 3) new coordinates after advection
        """
        # Sample current features at coordinates
        current_features = self.voxel_grid(coords)  # (B, N, feature_dim)

        # Predict action-conditioned velocity
        velocity = self.flow_predictor(coords, current_features, actions)  # (B, N, 3)

        # Semi-implicit Euler step
        new_coords = coords + dt * velocity

        # Clamp to valid range
        new_coords = torch.clamp(new_coords, 0, self.grid_size - 1)

        # Sample features at new locations
        advected_features = self.voxel_grid(new_coords)

        return advected_features, new_coords, velocity


class SyntheticVideoSequenceGenerator:
    """
    Generates synthetic video sequences for self-supervised training.
    Each sequence has a blob that moves according to an action.
    """

    def __init__(self, grid_size=32, feature_dim=1, num_frames=10, sequence_length=5):
        """
        Args:
            grid_size: size of voxel grid
            feature_dim: dimension of voxel features
            num_frames: number of sequences to generate
            sequence_length: number of frames per sequence
        """
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.sequence_length = sequence_length

    def create_gaussian_blob(self, center, std=1.0):
        """Create a 3D Gaussian blob at a given center position."""
        device = torch.device('cpu')
        features = torch.zeros(1, self.feature_dim, self.grid_size, self.grid_size, self.grid_size, device=device)

        z, y, x = torch.meshgrid(
            torch.arange(self.grid_size, dtype=torch.float32, device=device),
            torch.arange(self.grid_size, dtype=torch.float32, device=device),
            torch.arange(self.grid_size, dtype=torch.float32, device=device),
            indexing='ij'
        )

        dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        gaussian = torch.exp(-dist_sq / (2 * std ** 2))

        for c in range(self.feature_dim):
            features[0, c, :, :, :] = gaussian

        return features

    def generate_sequence(self):
        """
        Generate a single synthetic video sequence.

        Returns:
            frames: list of (1, feature_dim, grid_size, grid_size, grid_size) tensors
            actions: (sequence_length-1, 3) action vectors applied between frames
        """
        frames = []
        actions = []

        # Random starting position
        start_pos = torch.tensor([
            self.grid_size / 2.0 + torch.randint(-3, 4, (1,)).item(),
            self.grid_size / 2.0 + torch.randint(-3, 4, (1,)).item(),
            self.grid_size / 2.0 + torch.randint(-3, 4, (1,)).item(),
        ], dtype=torch.float32)

        # Random initial action direction
        current_action = torch.randn(3)
        current_action = current_action / (current_action.norm() + 1e-6) * 0.5  # Normalize and scale

        # Generate frames
        current_pos = start_pos.clone()

        for frame_idx in range(self.sequence_length):
            # Create blob at current position
            blob = self.create_gaussian_blob(current_pos.tolist(), std=1.5)
            frames.append(blob)

            if frame_idx < self.sequence_length - 1:
                # Add slight random noise to action for variety
                current_action = current_action + torch.randn(3) * 0.1
                current_action = current_action / (current_action.norm() + 1e-6) * 0.5
                actions.append(current_action.clone())

                # Move position based on action
                current_pos = current_pos + current_action * 1.0
                # Clamp to keep blob in grid
                current_pos = torch.clamp(current_pos, 2.0, self.grid_size - 2.0)

        actions = torch.stack(actions, dim=0)  # (sequence_length-1, 3)
        return frames, actions

    def generate_batch(self, batch_size):
        """
        Generate a batch of sequences.

        Returns:
            frames_list: list of sequences, each with (B, feature_dim, grid_size, grid_size, grid_size)
            actions_list: list of action sequences, each with (B, sequence_length-1, 3)
        """
        frames_batch = []
        actions_batch = []

        for _ in range(batch_size):
            frames, actions = self.generate_sequence()
            frames_stacked = torch.cat(frames, dim=0)  # Stack into batch of 1
            frames_batch.append(frames_stacked)
            actions_batch.append(actions)

        # Stack into batches
        # frames_batch: (batch_size * sequence_length, 1, feature_dim, grid_size, grid_size, grid_size)
        frames_batch = torch.cat(frames_batch, dim=0)
        # Reshape to separate batch and sequence dims
        frames_batch = frames_batch.reshape(batch_size, self.sequence_length, self.feature_dim, self.grid_size, self.grid_size, self.grid_size)

        actions_batch = torch.stack(actions_batch, dim=0)  # (batch_size, sequence_length-1, 3)

        return frames_batch, actions_batch


def train_action_conditioned_model(num_epochs=10, batch_size=4, learning_rate=0.001):
    """
    Train the action-conditioned feature advection model on synthetic video sequences.

    Args:
        num_epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer

    Returns:
        model: trained ActionConditionedAdvection model
        losses: list of loss values per epoch
    """
    print("=" * 70)
    print("Neural Voxel Dynamics - Pass 3: Action-Conditioned Flow Training")
    print("=" * 70)

    device = torch.device('cpu')

    # Initialize model
    grid_size = 32
    feature_dim = 1
    action_dim = 3

    model = ActionConditionedAdvection(grid_size, feature_dim, action_dim)
    model.to(device)

    # Create synthetic data generator
    data_generator = SyntheticVideoSequenceGenerator(
        grid_size=grid_size,
        feature_dim=feature_dim,
        sequence_length=5
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    losses = []
    print(f"\nConfiguration:")
    print(f"  Grid size: {grid_size}³")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Epochs: {num_epochs}, Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    # Training loop
    print(f"\nTraining for {num_epochs} epochs:")
    print(f"{'Epoch':>6} | {'Loss':>10} | {'Progress':>30}")
    print("-" * 50)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Generate data for this epoch
        for batch_idx in range(2):  # 2 batches per epoch for variety
            # Generate sequences: (batch_size, sequence_length, feature_dim, grid_size, grid_size, grid_size)
            frames, actions = data_generator.generate_batch(batch_size)
            frames = frames.to(device)
            actions = actions.to(device)

            # Process sequence: predict next frame from current frame + action
            for seq_idx in range(frames.shape[1] - 1):
                current_frame = frames[:, seq_idx]  # (batch_size, feature_dim, grid_size, grid_size, grid_size)
                next_frame = frames[:, seq_idx + 1]  # (batch_size, feature_dim, grid_size, grid_size, grid_size)
                action = actions[:, seq_idx]  # (batch_size, 3)

                # Sample random points from current frame
                # For simplicity, we track points at the center of each blob
                center_coord = torch.tensor([grid_size / 2.0, grid_size / 2.0, grid_size / 2.0], device=device)
                coords = center_coord.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)  # (batch_size, 1, 3)

                # Advect features using action-conditioned model
                # But we need to inject current frame features into the voxel grid
                model.voxel_grid.features.data = current_frame

                predicted_features, new_coords, velocity = model(coords, action, dt=1.0)  # (batch_size, 1, feature_dim)

                # Get ground truth features at new coordinates
                model.voxel_grid.features.data = next_frame
                target_features = model.voxel_grid(new_coords)  # (batch_size, 1, feature_dim)

                # Compute loss
                loss = mse_loss(predicted_features, target_features)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        progress_bar = "▓" * int(20 * (epoch + 1) / num_epochs) + "░" * (20 - int(20 * (epoch + 1) / num_epochs))
        print(f"{epoch + 1:6d} | {avg_loss:10.6f} | [{progress_bar}]")

    print("\n" + "=" * 70)
    print(f"Training completed! Final loss: {losses[-1]:.6f}")
    print(f"Loss evolution: {losses[0]:.6f} → {losses[-1]:.6f}")
    print("=" * 70)

    return model, losses


if __name__ == '__main__':
    model, losses = train_action_conditioned_model(
        num_epochs=10,
        batch_size=4,
        learning_rate=0.001
    )
