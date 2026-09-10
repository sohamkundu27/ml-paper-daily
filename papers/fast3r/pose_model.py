"""Fast3R Pass 1: Pairwise relative pose estimation from image embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
