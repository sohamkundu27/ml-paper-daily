"""Fast3R Pass 1-2: Pairwise relative pose estimation and multi-view Transformer refinement."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PoseHead(nn.Module):
    """Predicts 6D relative pose from two concatenated image embeddings."""

    def __init__(self, embedding_dim=768, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.pose_out = nn.Linear(hidden_dim, 6)  # 6D rotation + 3D translation

    def forward(self, emb1, emb2):
        """
        Args:
            emb1: (batch, embedding_dim) - first image embedding
            emb2: (batch, embedding_dim) - second image embedding

        Returns:
            pose_6d: (batch, 6) - 6D relative pose representation
        """
        x = torch.cat([emb1, emb2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pose_6d = self.pose_out(x)
        return pose_6d


def rotation_6d_to_matrix(rot_6d):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Uses Gram-Schmidt orthogonalization on the first two columns.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    """
    batch_size = rot_6d.shape[0]
    x_raw = rot_6d[:, :3]
    y_raw = rot_6d[:, 3:6]

    x = F.normalize(x_raw, p=2, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, p=2, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1)  # (batch, 3, 3)
    return matrix


def pose_6d_to_matrix(pose_6d):
    """Convert 6D pose (rotation + translation) to 3x4 transformation matrix.

    Args:
        pose_6d: (batch, 6) - first 3 elements are rotation in 6D form, last 3 are translation

    Returns:
        (batch, 3, 4) - transformation matrix [R | t]
    """
    rot_6d = pose_6d[:, :6]
    trans = pose_6d[:, 6:9]

    # Handle case where pose_6d is (batch, 6) - just use rotation
    if pose_6d.shape[1] == 6:
        rot_6d = pose_6d
        trans = torch.zeros(pose_6d.shape[0], 3, device=pose_6d.device)

    rot_matrix = rotation_6d_to_matrix(rot_6d)
    matrix = torch.cat([rot_matrix, trans.unsqueeze(-1)], dim=-1)  # (batch, 3, 4)
    return matrix


class PairwisePoseModel(nn.Module):
    """Fast3R Pass 1: predicts relative poses between image pairs."""

    def __init__(self, embedding_dim=768, hidden_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pose_head = PoseHead(embedding_dim, hidden_dim)

    def forward(self, embeddings, pair_indices):
        """Predict relative poses for given image pairs.

        Args:
            embeddings: (num_images, embedding_dim)
            pair_indices: (num_pairs, 2) - pairs of image indices

        Returns:
            poses_6d: (num_pairs, 6) - 6D relative pose for each pair
            pose_matrices: (num_pairs, 3, 4) - transformation matrices
        """
        i_indices, j_indices = pair_indices[:, 0], pair_indices[:, 1]
        emb_i = embeddings[i_indices]
        emb_j = embeddings[j_indices]

        poses_6d = self.pose_head(emb_i, emb_j)
        pose_matrices = pose_6d_to_matrix(poses_6d)

        return poses_6d, pose_matrices


class PoseRefinementTransformer(nn.Module):
    """Pass 2: Transformer encoder that refines pairwise pose predictions through multi-view aggregation."""

    def __init__(self, pose_feat_dim=32, num_heads=8, num_layers=2, hidden_dim=128):
        """
        Args:
            pose_feat_dim: dimension of pose feature embeddings
            num_heads: number of attention heads
            num_layers: number of transformer layers
            hidden_dim: hidden dimension in transformer MLP
        """
        super().__init__()
        self.pose_feat_dim = pose_feat_dim

        # Project 6D pose to higher-dimensional feature space for transformer
        self.pose_encoder = nn.Linear(6, pose_feat_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pose_feat_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project refined features back to 6D pose space
        self.pose_decoder = nn.Linear(pose_feat_dim, 6)

    def forward(self, poses_6d):
        """Refine pairwise pose predictions using multi-view attention.

        Args:
            poses_6d: (num_pairs, 6) - initial 6D pose predictions from Pass 1

        Returns:
            refined_poses_6d: (num_pairs, 6) - refined pose predictions
        """
        # Encode poses to feature space
        pose_features = self.pose_encoder(poses_6d)  # (num_pairs, pose_feat_dim)

        # Apply transformer to aggregate pose information across all pairs
        # Attention learns which pose predictions should influence each other
        refined_features = self.transformer(pose_features.unsqueeze(0))  # (1, num_pairs, pose_feat_dim)
        refined_features = refined_features.squeeze(0)

        # Decode back to 6D pose space
        refined_poses_6d = self.pose_decoder(refined_features)  # (num_pairs, 6)

        # Add residual connection to maintain pose information from Pass 1
        refined_poses_6d = refined_poses_6d + poses_6d

        return refined_poses_6d


class Fast3RMultiView(nn.Module):
    """Fast3R Pass 1+2: Combines pairwise pose estimation with multi-view Transformer refinement."""

    def __init__(self, embedding_dim=768, pose_hidden_dim=512, pose_feat_dim=32,
                 num_transformer_heads=8, num_transformer_layers=2):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Pass 1: Pairwise pose estimation
        self.pairwise_model = PairwisePoseModel(embedding_dim, pose_hidden_dim)

        # Pass 2: Multi-view Transformer refinement
        self.pose_refiner = PoseRefinementTransformer(
            pose_feat_dim=pose_feat_dim,
            num_heads=num_transformer_heads,
            num_layers=num_transformer_layers
        )

    def forward(self, embeddings, pair_indices):
        """Predict and refine relative poses for image pairs.

        Args:
            embeddings: (num_images, embedding_dim)
            pair_indices: (num_pairs, 2) - pairs of image indices

        Returns:
            initial_poses_6d: (num_pairs, 6) - initial pose predictions from Pass 1
            refined_poses_6d: (num_pairs, 6) - refined pose predictions from Pass 2
            refined_pose_matrices: (num_pairs, 3, 4) - transformation matrices from refined poses
        """
        # Pass 1: Get initial pairwise pose estimates
        initial_poses_6d, _ = self.pairwise_model(embeddings, pair_indices)

        # Pass 2: Refine poses through Transformer aggregation
        refined_poses_6d = self.pose_refiner(initial_poses_6d)

        # Convert refined 6D poses to 3x4 matrices
        refined_pose_matrices = pose_6d_to_matrix(refined_poses_6d)

        return initial_poses_6d, refined_poses_6d, refined_pose_matrices
