"""Tests for Fast3R Pass 2: multi-view Transformer refinement."""
import torch
import numpy as np
from pose_model import (
    PoseRefinementTransformer, Fast3RMultiView, pose_6d_to_matrix, rotation_6d_to_matrix
)


def test_pose_refinement_transformer():
    """Test that PoseRefinementTransformer refines pose predictions correctly."""
    num_pairs = 10
    pose_feat_dim = 32
    num_heads = 8
    num_layers = 2

    # Create random initial pose predictions
    initial_poses = torch.randn(num_pairs, 6)

    refiner = PoseRefinementTransformer(
        pose_feat_dim=pose_feat_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )

    refined_poses = refiner(initial_poses)

    # Check shape
    assert refined_poses.shape == (num_pairs, 6), f"Expected ({num_pairs}, 6), got {refined_poses.shape}"

    # Refined poses should be different from initial (due to transformer layers)
    diff = torch.norm(refined_poses - initial_poses).item()
    assert diff > 0.0, "Refined poses should differ from initial poses"

    print("✓ test_pose_refinement_transformer passed")


def test_pose_refiner_with_few_images():
    """Test transformer refinement with 3-5 images (small multi-view case)."""
    for num_images in [3, 4, 5]:
        # Create all possible pairs
        pair_indices = []
        for i in range(num_images):
            for j in range(i + 1, num_images):
                pair_indices.append([i, j])
        pair_indices = torch.tensor(pair_indices)
        num_pairs = len(pair_indices)

        # Create random poses for each pair
        poses_6d = torch.randn(num_pairs, 6)

        refiner = PoseRefinementTransformer(pose_feat_dim=32, num_heads=4, num_layers=2)
        refined_poses = refiner(poses_6d)

        # Check shape consistency
        assert refined_poses.shape == (num_pairs, 6)

        # Check that poses are valid (can be converted to rotation matrices)
        pose_matrices = pose_6d_to_matrix(refined_poses)
        assert pose_matrices.shape == (num_pairs, 3, 4)

        # Verify rotation parts are orthogonal
        R = pose_matrices[:, :3, :3]
        for k in range(num_pairs):
            identity = R[k] @ R[k].T
            error = torch.norm(identity - torch.eye(3)).item()
            assert error < 0.2, f"Rotation in refined pair {k} is not orthogonal"

    print("✓ test_pose_refiner_with_few_images passed")


def test_gradient_flow_through_refiner():
    """Test that gradients flow through the pose refiner."""
    num_pairs = 8
    poses_6d = torch.randn(num_pairs, 6, requires_grad=True)

    refiner = PoseRefinementTransformer(pose_feat_dim=32, num_heads=4, num_layers=2)
    refined_poses = refiner(poses_6d)

    # Compute a simple loss and backprop
    loss = refined_poses.norm()
    loss.backward()

    # Check that gradients were computed
    assert poses_6d.grad is not None, "Input poses have no gradients"
    assert not torch.all(poses_6d.grad == 0), "Input poses have zero gradients"

    # Check model parameters have gradients
    for param in refiner.parameters():
        assert param.grad is not None, "Some model parameters have no gradients"

    print("✓ test_gradient_flow_through_refiner passed")


def test_fast3r_multiview_full_pipeline():
    """Test the complete Fast3R Pass 1+2 pipeline."""
    embedding_dim = 768
    num_images = 5

    # Create random image embeddings
    embeddings = torch.randn(num_images, embedding_dim)

    # Create all possible image pairs
    pair_indices = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            pair_indices.append([i, j])
    pair_indices = torch.tensor(pair_indices)

    model = Fast3RMultiView(
        embedding_dim=embedding_dim,
        pose_feat_dim=32,
        num_transformer_heads=8,
        num_transformer_layers=2
    )

    # Forward pass through Pass 1 and Pass 2
    initial_poses_6d, refined_poses_6d, refined_pose_matrices = model(embeddings, pair_indices)

    # Check shapes
    num_pairs = len(pair_indices)
    assert initial_poses_6d.shape == (num_pairs, 6), f"Initial poses shape mismatch"
    assert refined_poses_6d.shape == (num_pairs, 6), f"Refined poses shape mismatch"
    assert refined_pose_matrices.shape == (num_pairs, 3, 4), f"Pose matrices shape mismatch"

    # Verify rotation parts of refined poses are orthogonal
    R = refined_pose_matrices[:, :3, :3]
    for k in range(num_pairs):
        identity = R[k] @ R[k].T
        error = torch.norm(identity - torch.eye(3)).item()
        assert error < 0.2, f"Rotation in refined pose {k} is not orthogonal"

    print("✓ test_fast3r_multiview_full_pipeline passed")


def test_refinement_with_different_sizes():
    """Test that refinement handles various numbers of images."""
    embedding_dim = 256

    for num_images in [2, 3, 5, 10]:
        embeddings = torch.randn(num_images, embedding_dim)

        # Create all possible pairs
        pair_indices = []
        for i in range(num_images):
            for j in range(i + 1, num_images):
                pair_indices.append([i, j])
        pair_indices = torch.tensor(pair_indices)

        model = Fast3RMultiView(embedding_dim=embedding_dim, pose_feat_dim=16, num_transformer_heads=4)
        initial_poses, refined_poses, pose_mats = model(embeddings, pair_indices)

        assert refined_poses.shape[0] == len(pair_indices)
        assert pose_mats.shape == (len(pair_indices), 3, 4)

    print("✓ test_refinement_with_different_sizes passed")


def test_backward_pass_full_model():
    """Test that gradients flow through the complete model."""
    embedding_dim = 512
    num_images = 4

    embeddings = torch.randn(num_images, embedding_dim, requires_grad=True)
    pair_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]])

    model = Fast3RMultiView(embedding_dim=embedding_dim)
    initial_poses, refined_poses, pose_mats = model(embeddings, pair_indices)

    # Compute a loss and backprop
    loss = refined_poses.norm() + pose_mats.norm()
    loss.backward()

    # Check gradients
    assert embeddings.grad is not None
    assert not torch.all(embeddings.grad == 0)

    # Check model parameters
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

    print("✓ test_backward_pass_full_model passed")


def test_pose_consistency_after_refinement():
    """Test that refined poses are numerically valid for reconstruction."""
    num_pairs = 6
    poses_6d = torch.randn(num_pairs, 6)

    refiner = PoseRefinementTransformer(pose_feat_dim=32, num_heads=4, num_layers=2)
    refined_poses = refiner(poses_6d)

    # Convert to matrices
    matrices = pose_6d_to_matrix(refined_poses)

    # Check for NaN/Inf
    assert not torch.any(torch.isnan(matrices)), "Refined poses produce NaN in matrices"
    assert not torch.any(torch.isinf(matrices)), "Refined poses produce Inf in matrices"

    # Extract and verify rotations are orthogonal
    R = matrices[:, :3, :3]
    dets = torch.det(R)
    assert torch.all(torch.abs(dets - 1.0) < 0.15), "Some determinants not close to 1"

    print("✓ test_pose_consistency_after_refinement passed")


if __name__ == "__main__":
    test_pose_refinement_transformer()
    test_pose_refiner_with_few_images()
    test_gradient_flow_through_refiner()
    test_fast3r_multiview_full_pipeline()
    test_refinement_with_different_sizes()
    test_backward_pass_full_model()
    test_pose_consistency_after_refinement()
    print("\nAll Pass 2 tests passed! ✓")
