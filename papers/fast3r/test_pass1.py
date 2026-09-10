"""Tests for Fast3R Pass 1: pairwise pose estimation."""
import torch
import numpy as np
from pose_model import PairwisePoseModel, rotation_6d_to_matrix, pose_6d_to_matrix


def test_rotation_6d_to_matrix():
    """Test that 6D rotation conversion produces valid rotation matrices."""
    batch_size = 4
    rot_6d = torch.randn(batch_size, 6)

    rot_matrix = rotation_6d_to_matrix(rot_6d)

    # Check shape
    assert rot_matrix.shape == (batch_size, 3, 3), f"Expected (4, 3, 3), got {rot_matrix.shape}"

    # Check orthogonality: R @ R.T = I
    for i in range(batch_size):
        R = rot_matrix[i]
        identity = R @ R.T
        expected_identity = torch.eye(3)
        error = torch.norm(identity - expected_identity).item()
        assert error < 0.1, f"Rotation matrix {i} is not orthogonal, error: {error}"

    # Check determinant ≈ 1 (proper rotation, not reflection)
    for i in range(batch_size):
        det = torch.det(rot_matrix[i]).item()
        assert abs(det - 1.0) < 0.1, f"Rotation matrix {i} has det {det}, expected ≈1"

    print("✓ test_rotation_6d_to_matrix passed")


def test_pose_6d_to_matrix():
    """Test that 6D pose converts to 3x4 transformation matrices."""
    batch_size = 8
    pose_6d = torch.randn(batch_size, 6)

    pose_matrix = pose_6d_to_matrix(pose_6d)

    # Check shape
    assert pose_matrix.shape == (batch_size, 3, 4), f"Expected (8, 3, 4), got {pose_matrix.shape}"

    # Extract rotation and check it's orthogonal
    R = pose_matrix[:, :3, :3]
    for i in range(batch_size):
        identity = R[i] @ R[i].T
        error = torch.norm(identity - torch.eye(3)).item()
        assert error < 0.1, f"Rotation in pose {i} is not orthogonal"

    print("✓ test_pose_6d_to_matrix passed")


def test_pairwise_pose_model():
    """Test the full pairwise pose model."""
    embedding_dim = 768
    num_images = 5
    num_pairs = 6

    # Create random image embeddings
    embeddings = torch.randn(num_images, embedding_dim)

    # Create a few image pairs
    pair_indices = torch.tensor([
        [0, 1], [0, 2], [1, 2],
        [2, 3], [3, 4], [1, 4]
    ])

    model = PairwisePoseModel(embedding_dim=embedding_dim)
    poses_6d, pose_matrices = model(embeddings, pair_indices)

    # Check shapes
    assert poses_6d.shape == (num_pairs, 6), f"Expected (6, 6), got {poses_6d.shape}"
    assert pose_matrices.shape == (num_pairs, 3, 4), f"Expected (6, 3, 4), got {pose_matrices.shape}"

    # Verify rotation parts are orthogonal
    R = pose_matrices[:, :3, :3]
    for i in range(num_pairs):
        identity = R[i] @ R[i].T
        error = torch.norm(identity - torch.eye(3)).item()
        assert error < 0.2, f"Rotation in pair {i} is not orthogonal, error: {error}"

    # Verify translation is reasonable (not NaN/Inf)
    t = pose_matrices[:, :3, 3]
    assert not torch.any(torch.isnan(t)), "Translations contain NaN"
    assert not torch.any(torch.isinf(t)), "Translations contain Inf"

    print("✓ test_pairwise_pose_model passed")


def test_forward_backward():
    """Test that gradients flow correctly through the model."""
    embedding_dim = 256
    num_images = 3
    batch_size = 2

    embeddings = torch.randn(num_images, embedding_dim, requires_grad=True)
    pair_indices = torch.tensor([[0, 1], [1, 2]])

    model = PairwisePoseModel(embedding_dim=embedding_dim)
    poses_6d, pose_matrices = model(embeddings, pair_indices)

    # Compute a simple loss and backprop
    loss = poses_6d.norm()
    loss.backward()

    # Check that gradients were computed
    assert embeddings.grad is not None, "Embeddings have no gradients"
    assert not torch.all(embeddings.grad == 0), "Embeddings have zero gradients"

    # Check model parameters have gradients
    for param in model.parameters():
        assert param.grad is not None, "Some model parameters have no gradients"

    print("✓ test_forward_backward passed")


def test_batched_processing():
    """Test that model can handle different batch sizes."""
    embedding_dim = 512
    model = PairwisePoseModel(embedding_dim=embedding_dim)

    for num_images in [2, 5, 10]:
        embeddings = torch.randn(num_images, embedding_dim)
        num_pairs = num_images * (num_images - 1) // 2
        pair_indices = []
        for i in range(num_images):
            for j in range(i + 1, num_images):
                pair_indices.append([i, j])
        pair_indices = torch.tensor(pair_indices)

        poses_6d, pose_matrices = model(embeddings, pair_indices)

        assert poses_6d.shape[0] == len(pair_indices), "Wrong number of pose predictions"
        assert pose_matrices.shape == (len(pair_indices), 3, 4), "Wrong pose matrix shape"

    print("✓ test_batched_processing passed")


if __name__ == "__main__":
    test_rotation_6d_to_matrix()
    test_pose_6d_to_matrix()
    test_pairwise_pose_model()
    test_forward_backward()
    test_batched_processing()
    print("\nAll Pass 1 tests passed! ✓")
