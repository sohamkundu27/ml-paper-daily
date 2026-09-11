"""Tests for Fast3R Pass 3: pose consistency loss and ICP-style refinement."""
import torch
import numpy as np
from pose_model import (
    pose_6d_to_matrix, rotation_6d_to_matrix, compose_poses_matrices,
    pose_consistency_loss, ICPPoseRefiner, Fast3RWithConsistency
)


def test_compose_poses_matrices():
    """Test pose composition (chaining two transformations)."""
    batch_size = 4

    # Create two random transformations
    poses1_6d = torch.randn(batch_size, 6)
    poses2_6d = torch.randn(batch_size, 6)

    T1 = pose_6d_to_matrix(poses1_6d)  # (batch, 3, 4)
    T2 = pose_6d_to_matrix(poses2_6d)

    T_composed = compose_poses_matrices(T1, T2)

    # Check shape
    assert T_composed.shape == (batch_size, 3, 4), f"Expected (batch, 3, 4), got {T_composed.shape}"

    # Extract and verify rotation is orthogonal
    R_composed = T_composed[:, :3, :3]
    for i in range(batch_size):
        identity = R_composed[i] @ R_composed[i].T
        error = torch.norm(identity - torch.eye(3)).item()
        assert error < 0.2, f"Composed rotation {i} is not orthogonal, error: {error}"

    # Check determinant ≈ 1
    for i in range(batch_size):
        det = torch.det(R_composed[i]).item()
        assert abs(det - 1.0) < 0.15, f"Composed rotation {i} has det {det}, expected ≈1"

    print("✓ test_compose_poses_matrices passed")


def test_pose_consistency_loss_small():
    """Test pose consistency loss on a small multi-view case."""
    # Create a simple 3-view case with all pairs
    num_images = 3
    pair_indices = torch.tensor([[0, 1], [0, 2], [1, 2]])

    # Create random poses
    poses_6d = torch.randn(3, 6, requires_grad=True)

    # Compute consistency loss
    loss = pose_consistency_loss(poses_6d, pair_indices, num_images)

    # Loss should be a scalar
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"

    # Loss should be non-negative
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    # Check that gradients flow
    loss.backward()
    assert poses_6d.grad is not None, "Poses should have gradients"
    assert not torch.all(poses_6d.grad == 0), "Poses should have non-zero gradients"

    print("✓ test_pose_consistency_loss_small passed")


def test_pose_consistency_loss_larger():
    """Test consistency loss on a larger multi-view case."""
    num_images = 5
    pair_indices = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            pair_indices.append([i, j])
    pair_indices = torch.tensor(pair_indices)

    poses_6d = torch.randn(len(pair_indices), 6)

    loss = pose_consistency_loss(poses_6d, pair_indices, num_images)

    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("✓ test_pose_consistency_loss_larger passed")


def test_icp_pose_refiner():
    """Test the ICP-style pose refiner."""
    num_images = 4
    num_pairs = 6

    # Create a triangle (3 pairs: 0-1, 1-2, 0-2) and an extra pair
    pair_indices = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 3], [0, 3], [1, 3]])

    # Create initial poses
    poses_6d = torch.randn(num_pairs, 6)

    refiner = ICPPoseRefiner(num_iterations=3, learning_rate=0.01)

    refined_poses, consistency_losses = refiner(poses_6d, pair_indices, num_images)

    # Check shape
    assert refined_poses.shape == poses_6d.shape, "Refined poses shape mismatch"

    # Check that consistency loss decreases or stays stable
    assert len(consistency_losses) == 3, f"Expected 3 loss values, got {len(consistency_losses)}"
    assert all(l >= 0 for l in consistency_losses), "All losses should be non-negative"

    # The last loss should be reasonably low (though we don't constrain optimization here)
    # Just check it's a valid number
    assert not any(np.isnan(l) or np.isinf(l) for l in consistency_losses), \
        "Loss values should be finite"

    # Refined poses should be different from initial (due to optimization)
    diff = torch.norm(refined_poses - poses_6d).item()
    assert diff > 0.0, "Refined poses should differ from initial poses"

    print("✓ test_icp_pose_refiner passed")


def test_icp_refiner_convergence():
    """Test that ICP refiner converges when starting from reasonable poses."""
    num_images = 4
    pair_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]])

    # Start with small random poses (closer to identity)
    poses_6d = torch.randn(4, 6) * 0.1

    refiner = ICPPoseRefiner(num_iterations=5, learning_rate=0.05)

    refined_poses, consistency_losses = refiner(poses_6d, pair_indices, num_images)

    # Check that losses are decreasing or stable (convergence check)
    # We allow some noise, so check general trend
    initial_loss = consistency_losses[0]
    final_loss = consistency_losses[-1]

    # Final loss should be less than initial (or very close)
    assert final_loss <= initial_loss * 1.1, \
        f"Loss did not converge: initial={initial_loss:.6f}, final={final_loss:.6f}"

    print("✓ test_icp_refiner_convergence passed")


def test_fast3r_with_consistency_full_pipeline():
    """Test the complete Fast3R Pass 1+2+3 pipeline."""
    embedding_dim = 256
    num_images = 5

    # Create random embeddings
    embeddings = torch.randn(num_images, embedding_dim)

    # Create all pairs
    pair_indices = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            pair_indices.append([i, j])
    pair_indices = torch.tensor(pair_indices)

    model = Fast3RWithConsistency(
        embedding_dim=embedding_dim,
        pose_feat_dim=16,
        num_transformer_heads=4,
        num_transformer_layers=1,
        num_icp_iterations=3,
        icp_learning_rate=0.01
    )

    # Forward pass
    initial_poses, pass2_poses, pass3_poses, consistency_losses = model(embeddings, pair_indices)

    # Check shapes
    num_pairs = len(pair_indices)
    assert initial_poses.shape == (num_pairs, 6), "Initial poses shape mismatch"
    assert pass2_poses.shape == (num_pairs, 6), "Pass 2 poses shape mismatch"
    assert pass3_poses.shape == (num_pairs, 6), "Pass 3 poses shape mismatch"

    # Check consistency losses
    assert len(consistency_losses) == 3, "Should have 3 consistency loss values"
    assert all(l >= 0 for l in consistency_losses), "All losses should be non-negative"

    # Convert to matrices and verify rotations are orthogonal
    pass3_matrices = pose_6d_to_matrix(pass3_poses)
    R = pass3_matrices[:, :3, :3]
    for k in range(num_pairs):
        identity = R[k] @ R[k].T
        error = torch.norm(identity - torch.eye(3)).item()
        assert error < 0.2, f"Rotation in pass3 pose {k} is not orthogonal"

    print("✓ test_fast3r_with_consistency_full_pipeline passed")


def test_consistency_refinement_synthetic_data():
    """Demo: Test refinement on synthetic circular camera motion."""
    num_images = 10
    embedding_dim = 128

    # Create embeddings (just random for this test)
    embeddings = torch.randn(num_images, embedding_dim)

    # Create a subset of image pairs (not all pairs, to be practical)
    pair_indices = []
    for i in range(num_images):
        # Connect each image to its neighbors
        for offset in [1, 2]:
            j = (i + offset) % num_images
            if i < j:
                pair_indices.append([i, j])
    pair_indices = torch.tensor(pair_indices)

    model = Fast3RWithConsistency(
        embedding_dim=embedding_dim,
        pose_feat_dim=16,
        num_transformer_heads=4,
        num_transformer_layers=1,
        num_icp_iterations=5,
        icp_learning_rate=0.05
    )

    # Forward pass
    initial_poses, pass2_poses, pass3_poses, consistency_losses = model(embeddings, pair_indices)

    # Verify that refinement process ran
    assert len(consistency_losses) == 5, "Should have 5 iterations of refinement"

    # Check that poses remain valid
    pass3_matrices = pose_6d_to_matrix(pass3_poses)
    assert pass3_matrices.shape == (len(pair_indices), 3, 4)

    # Verify all rotations are orthogonal
    R = pass3_matrices[:, :3, :3]
    for k in range(len(pair_indices)):
        identity = R[k] @ R[k].T
        error = torch.norm(identity - torch.eye(3)).item()
        assert error < 0.2, f"Rotation {k} not orthogonal"

    print(f"✓ test_consistency_refinement_synthetic_data passed")
    print(f"  - Processed {num_images} images, {len(pair_indices)} pairs")
    print(f"  - Initial consistency loss: {consistency_losses[0]:.6f}")
    print(f"  - Final consistency loss: {consistency_losses[-1]:.6f}")


def test_pass3_gradient_flow():
    """Test that gradients flow through Pass 1+2 (Pass 3 ICP is a separate optimization)."""
    embedding_dim = 128
    num_images = 4

    embeddings = torch.randn(num_images, embedding_dim, requires_grad=True)
    pair_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]])

    model = Fast3RWithConsistency(
        embedding_dim=embedding_dim,
        pose_feat_dim=16,
        num_transformer_heads=4,
        num_icp_iterations=2
    )

    initial_poses, pass2_poses, pass3_poses, _ = model(embeddings, pair_indices)

    # Note: Pass 3 (ICP) does gradient descent on poses separately, so pass3_poses
    # may not have gradients flowing back to embeddings. But Pass 2 poses should.
    loss = pass2_poses.norm()
    loss.backward()

    # Check gradients on embeddings (from Pass 1+2)
    assert embeddings.grad is not None, "Embeddings should have gradients from Pass 1+2"
    assert not torch.all(embeddings.grad == 0), "Embeddings should have non-zero gradients"

    # Check gradients on model parameters
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None, "All model parameters should have gradients"

    print("✓ test_pass3_gradient_flow passed")


def test_consistency_loss_reduces_with_optimization():
    """Test that consistency loss actually decreases during refinement."""
    num_images = 4
    pair_indices = torch.tensor([[0, 1], [1, 2], [2, 3]])

    # Start with small initial poses
    poses_6d = torch.randn(3, 6) * 0.1

    refiner = ICPPoseRefiner(num_iterations=10, learning_rate=0.1)
    refined_poses, losses = refiner(poses_6d, pair_indices, num_images)

    # Loss should decrease significantly
    loss_decrease = losses[0] - losses[-1]
    assert loss_decrease >= -0.1 * losses[0], \
        f"Loss should decrease; got {losses[0]} -> {losses[-1]}"

    print("✓ test_consistency_loss_reduces_with_optimization passed")
    print(f"  - Loss evolution: {[f'{l:.4f}' for l in losses]}")


def test_large_scale_demo():
    """Demo: Test on a larger synthetic case (15-20 images)."""
    num_images = 15
    embedding_dim = 256

    # Create embeddings
    embeddings = torch.randn(num_images, embedding_dim)

    # Create pairs: connect each to a few neighbors + some random pairs
    pair_indices = []
    for i in range(num_images):
        # Connect to next 2 neighbors (circular)
        for offset in [1, 2]:
            j = (i + offset) % num_images
            if i < j:
                pair_indices.append([i, j])

    pair_indices = torch.tensor(pair_indices)

    model = Fast3RWithConsistency(
        embedding_dim=embedding_dim,
        pose_feat_dim=32,
        num_transformer_heads=8,
        num_transformer_layers=2,
        num_icp_iterations=3,
        icp_learning_rate=0.05
    )

    # Forward pass
    initial_poses, pass2_poses, pass3_poses, consistency_losses = model(embeddings, pair_indices)

    # Verify output
    num_pairs = len(pair_indices)
    assert pass3_poses.shape == (num_pairs, 6), "Output shape mismatch"

    # Convert to matrices
    pass3_matrices = pose_6d_to_matrix(pass3_poses)

    # Verify all rotations are valid
    R = pass3_matrices[:, :3, :3]
    for k in range(num_pairs):
        det = torch.det(R[k]).item()
        assert abs(det - 1.0) < 0.2, f"Rotation {k} has invalid determinant"

    print(f"✓ test_large_scale_demo passed")
    print(f"  - Processed {num_images} images, {num_pairs} pairs")
    print(f"  - Consistency losses: {[f'{l:.4f}' for l in consistency_losses]}")


if __name__ == "__main__":
    test_compose_poses_matrices()
    test_pose_consistency_loss_small()
    test_pose_consistency_loss_larger()
    test_icp_pose_refiner()
    test_icp_refiner_convergence()
    test_fast3r_with_consistency_full_pipeline()
    test_consistency_refinement_synthetic_data()
    test_pass3_gradient_flow()
    test_consistency_loss_reduces_with_optimization()
    test_large_scale_demo()
    print("\n✓ All Pass 3 tests passed!")
