"""Fast3R Pass 4: End-to-end demo with synthetic multi-view scene, 3D reconstruction, and evaluation."""
import torch
import numpy as np
from pose_model import Fast3RWithConsistency, pose_6d_to_matrix


class SyntheticMultiViewScene:
    """Generate synthetic camera poses and 3D points for multi-view reconstruction."""

    def __init__(self, num_images=8, num_points=50, seed=42):
        """
        Args:
            num_images: number of views
            num_points: number of 3D points to generate
            seed: random seed for reproducibility
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_images = num_images
        self.num_points = num_points

        # Focal length and principal point (simple camera intrinsics)
        self.focal_length = 500.0
        self.principal_point = np.array([320.0, 240.0])
        self.image_size = (640, 480)

        # Generate ground truth camera poses (circular motion around origin)
        self.gt_poses = self._generate_circular_camera_motion()

        # Generate random 3D points in front of cameras
        self.points_3d = self._generate_3d_points()

        # Project 3D points to 2D image coordinates
        self.points_2d_per_image = self._project_3d_to_2d()

    def _generate_circular_camera_motion(self):
        """Generate camera poses in a circle around the origin."""
        poses = []
        for i in range(self.num_images):
            angle = 2 * np.pi * i / self.num_images
            radius = 3.0

            # Camera position (in circle)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 2.0 + 0.3 * np.sin(2 * angle)  # slight vertical variation

            # Build camera-to-world rotation and translation
            # Camera frame: X=right, Y=up, Z=backward (into scene)
            cam_pos = np.array([x, y, z])
            look_at = np.array([0, 0, 1.5])

            # Forward direction (into the scene)
            forward = look_at - cam_pos
            forward = forward / np.linalg.norm(forward)

            # Right vector (perpendicular to forward and world up)
            world_up = np.array([0, 0, 1])
            right = np.cross(world_up, forward)
            right = right / np.linalg.norm(right)

            # Actual up vector (perpendicular to forward and right)
            up = np.cross(forward, right)

            # Rotation matrix: columns are the camera axes in world coordinates
            # [right | up | forward]
            R_c2w = np.stack([right, up, forward], axis=1)
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ cam_pos

            poses.append({'R': R_w2c, 't': t_w2c})

        return poses

    def _generate_3d_points(self):
        """Generate random 3D points in front of all cameras."""
        # Generate points in a central region that all cameras can see
        # Using the center of the scene which cameras orbit around
        points = np.random.uniform(-1.5, 1.5, size=(self.num_points, 3))
        points[:, 2] = np.random.uniform(0.5, 3.0, size=self.num_points)  # depth range near center
        return points

    def _project_3d_to_2d(self):
        """Project 3D points to 2D image coordinates for each camera."""
        points_2d_per_image = []

        for pose_data in self.gt_poses:
            R = pose_data['R']
            t = pose_data['t']

            # Transform points to camera frame: p_cam = R @ p_world + t
            points_cam = (R @ self.points_3d.T + t[:, np.newaxis]).T  # (num_points, 3)

            # Perspective projection
            points_2d = []

            for p_cam in points_cam:
                if p_cam[2] > 0.5:  # must be in front of camera
                    x = self.focal_length * p_cam[0] / p_cam[2] + self.principal_point[0]
                    y = self.focal_length * p_cam[1] / p_cam[2] + self.principal_point[1]
                    # Check if point is in image bounds
                    if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                        points_2d.append([x, y])
                    else:
                        points_2d.append([np.nan, np.nan])
                else:
                    points_2d.append([np.nan, np.nan])

            points_2d_per_image.append(np.array(points_2d))

        return points_2d_per_image

    def get_relative_pose(self, i, j):
        """Get ground truth relative pose from camera i to camera j."""
        R_i = self.gt_poses[i]['R']
        t_i = self.gt_poses[i]['t']
        R_j = self.gt_poses[j]['R']
        t_j = self.gt_poses[j]['t']

        # Relative pose: T_ij = T_j * T_i^{-1}
        # In camera-world coordinates: R_ij = R_j @ R_i^T, t_ij = t_j - R_j @ R_i^T @ t_i
        R_ij = R_j @ R_i.T
        t_ij = t_j - R_ij @ t_i

        return {'R': R_ij, 't': t_ij}


def rotation_matrix_to_6d(R):
    """Convert 3x3 rotation matrix to 6D representation (first two columns)."""
    return R[:, :2].reshape(6)


def triangulate_point(P1, P2, point1_2d, point2_2d):
    """Triangulate a 3D point from two views using Linear Triangulation Method.

    Args:
        P1: (3, 4) first camera projection matrix [R1 | t1]
        P2: (3, 4) second camera projection matrix [R2 | t2]
        point1_2d: (2,) 2D point in first image
        point2_2d: (2,) 2D point in second image

    Returns:
        point_3d: (3,) triangulated 3D point (or None if degenerate)
    """
    x1, y1 = point1_2d
    x2, y2 = point2_2d

    # Build the system: A @ X = 0
    # Each point correspondence gives 2 equations
    A = np.zeros((4, 4))
    A[0] = x1 * P1[2] - P1[0]
    A[1] = y1 * P1[2] - P1[1]
    A[2] = x2 * P2[2] - P2[0]
    A[3] = y2 * P2[2] - P2[1]

    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]

    # Normalize to get 3D point
    if X[3] != 0:
        point_3d = X[:3] / X[3]
    else:
        return None

    return point_3d


def compute_reprojection_error(points_3d_pred, points_2d_gt, P):
    """Compute reprojection error for a set of 3D points.

    Args:
        points_3d_pred: (num_points, 3) predicted 3D points
        points_2d_gt: (num_points, 2) ground truth 2D points
        P: (3, 4) camera projection matrix

    Returns:
        error: average reprojection error (in pixels)
    """
    # Project 3D points using predicted pose
    points_3d_homog = np.hstack([points_3d_pred, np.ones((points_3d_pred.shape[0], 1))])
    points_2d_proj = (P @ points_3d_homog.T).T  # (num_points, 3)

    # Normalize by homogeneous coordinate
    points_2d_proj = points_2d_proj[:, :2] / points_2d_proj[:, 2:3]

    # Compute error
    valid_mask = ~np.isnan(points_2d_gt).any(axis=1)
    if not valid_mask.any():
        return np.inf

    error = np.linalg.norm(points_2d_proj[valid_mask] - points_2d_gt[valid_mask], axis=1).mean()
    return error


def end_to_end_demo():
    """Full end-to-end demo: generate synthetic data, run Fast3R, triangulate, evaluate."""
    print("\n" + "=" * 70)
    print("FAST3R PASS 4: END-TO-END DEMO WITH SYNTHETIC MULTI-VIEW SCENE")
    print("=" * 70)

    # Step 1: Generate synthetic multi-view scene
    print("\n[Step 1] Generating synthetic multi-view scene...")
    scene = SyntheticMultiViewScene(num_images=8, num_points=100, seed=42)
    print(f"  ✓ Generated {scene.num_images} camera views")
    print(f"  ✓ Generated {scene.num_points} 3D points")
    print(f"  ✓ Camera trajectory: circular motion around scene center")

    # Step 2: Prepare data for Fast3R
    print("\n[Step 2] Preparing data for Fast3R model...")
    num_images = scene.num_images
    embedding_dim = 128

    # Create synthetic embeddings (random in practice, but deterministic here)
    embeddings = torch.randn(num_images, embedding_dim)

    # Create image pairs (use nearby pairs + some global pairs for connectivity)
    pair_indices = []
    for i in range(num_images):
        # Connect to neighbors (circular)
        for offset in [1, 2, 3]:
            j = (i + offset) % num_images
            if i < j:
                pair_indices.append([i, j])
    pair_indices = torch.tensor(pair_indices)
    print(f"  ✓ Created {len(pair_indices)} image pairs")

    # Step 3: Run Fast3R model
    print("\n[Step 3] Running Fast3R Pass 1+2+3 pipeline...")
    model = Fast3RWithConsistency(
        embedding_dim=embedding_dim,
        pose_hidden_dim=256,
        pose_feat_dim=32,
        num_transformer_heads=8,
        num_transformer_layers=2,
        num_icp_iterations=5,
        icp_learning_rate=0.05
    )

    # Run without gradient tracking for efficiency (we're just running inference)
    # Note: Pass 3 (ICP) will enable gradients internally for its own optimization
    model.eval()
    initial_poses_6d, pass2_poses_6d, pass3_poses_6d, consistency_losses = model(
        embeddings, pair_indices
    )

    print(f"  ✓ Pass 1: Initial pose predictions")
    print(f"  ✓ Pass 2: Transformer refinement")
    print(f"  ✓ Pass 3: ICP-style consistency refinement")
    print(f"    Consistency loss: {consistency_losses[0]:.6f} → {consistency_losses[-1]:.6f}")

    # Step 4: Convert predicted 6D poses to matrices
    print("\n[Step 4] Converting 6D poses to transformation matrices...")
    pred_pose_matrices = pose_6d_to_matrix(pass3_poses_6d)
    print(f"  ✓ Converted {len(pair_indices)} pose predictions")

    # Step 5: Triangulate 3D points from predicted poses
    print("\n[Step 5] Triangulating 3D points from predicted poses...")

    # Build camera projection matrices
    # Intrinsic matrix
    K = np.array([
        [scene.focal_length, 0, scene.principal_point[0]],
        [0, scene.focal_length, scene.principal_point[1]],
        [0, 0, 1]
    ])

    # For each pair, triangulate points
    triangulated_points_all = []
    reprojection_errors_pass = []
    reprojection_errors_gt = []

    successful_triangulations = 0

    for pair_idx, (i, j) in enumerate(pair_indices):
        i, j = int(i.item()), int(j.item())

        # Get predicted poses
        T_pred = pred_pose_matrices[pair_idx].numpy()  # (3, 4)
        P1_pred = K @ np.eye(3, 4)  # First camera at origin
        P2_pred = K @ T_pred  # Second camera from predicted pose

        # Get ground truth for evaluation only
        gt_pose_ij = scene.get_relative_pose(i, j)
        T_gt = np.hstack([gt_pose_ij['R'], gt_pose_ij['t'][:, np.newaxis]])
        P1_gt = K @ np.eye(3, 4)
        P2_gt = K @ T_gt

        # Triangulate points using predicted poses
        points_2d_i = scene.points_2d_per_image[i]
        points_2d_j = scene.points_2d_per_image[j]

        pair_triangulated = []
        pair_gt_points = []

        for pt_idx in range(scene.num_points):
            if np.isnan(points_2d_i[pt_idx]).any() or np.isnan(points_2d_j[pt_idx]).any():
                continue

            pt_3d = triangulate_point(P1_pred, P2_pred, points_2d_i[pt_idx], points_2d_j[pt_idx])
            if pt_3d is not None and np.all(np.isfinite(pt_3d)):
                pair_triangulated.append(pt_3d)
                pair_gt_points.append(scene.points_3d[pt_idx])

        if len(pair_triangulated) > 0:
            triangulated_points_all.extend(pair_triangulated)
            successful_triangulations += 1

            # Compute reprojection error using predicted poses
            points_3d_array = np.array(pair_triangulated)
            points_2d_for_reprojection = np.array([points_2d_i[pt_idx]
                                                   for pt_idx in range(len(pair_triangulated))])
            reprojection_error = compute_reprojection_error(
                points_3d_array, points_2d_for_reprojection, P1_pred
            )
            if np.isfinite(reprojection_error):
                reprojection_errors_pass.append(reprojection_error)

            # Also compute with ground truth poses for reference
            points_gt_array = np.array(pair_gt_points)
            reprojection_error_gt = compute_reprojection_error(
                points_gt_array, points_2d_for_reprojection, P1_gt
            )
            if np.isfinite(reprojection_error_gt):
                reprojection_errors_gt.append(reprojection_error_gt)

    if len(triangulated_points_all) > 0:
        triangulated_points_all = np.array(triangulated_points_all)
        print(f"  ✓ Successfully triangulated {len(triangulated_points_all)} 3D points across {successful_triangulations} pairs")

    # Step 6: Evaluate reconstruction accuracy
    print("\n[Step 6] Evaluating reconstruction accuracy...")

    avg_reprojection_error = np.inf
    avg_reprojection_error_gt = np.inf

    if len(reprojection_errors_pass) > 0:
        avg_reprojection_error = np.mean(reprojection_errors_pass)
        print(f"  ✓ Average reprojection error (predicted poses): {avg_reprojection_error:.4f} pixels")
    else:
        print(f"  ⚠ No valid reprojection errors computed with predicted poses")

    if len(reprojection_errors_gt) > 0:
        avg_reprojection_error_gt = np.mean(reprojection_errors_gt)
        print(f"  ✓ Average reprojection error (GT poses): {avg_reprojection_error_gt:.4f} pixels (reference)")
    else:
        print(f"  ⚠ No valid GT reprojection errors computed")

    # Evaluate relative pose accuracy
    pose_errors = []
    for pair_idx, (i, j) in enumerate(pair_indices):
        i, j = int(i.item()), int(j.item())

        # Predicted pose
        R_pred = pred_pose_matrices[pair_idx, :3, :3].numpy()

        # Ground truth pose
        gt_pose_ij = scene.get_relative_pose(i, j)
        R_gt = gt_pose_ij['R']

        # Compute rotation error as Frobenius norm
        rotation_error = np.linalg.norm(R_pred @ R_gt.T - np.eye(3))
        pose_errors.append(rotation_error)

    avg_pose_error = np.mean(pose_errors)
    print(f"  ✓ Average relative pose error: {avg_pose_error:.6f} (Frobenius norm)")

    # Step 7: Summary statistics
    print("\n[Step 7] Summary Statistics")
    print("-" * 70)
    print(f"  Input:")
    print(f"    - Number of images:           {num_images}")
    print(f"    - Number of image pairs:      {len(pair_indices)}")
    print(f"    - Number of 3D points (GT):   {scene.num_points}")
    print(f"\n  Pipeline Performance (Pass 1 → Pass 2 → Pass 3):")
    print(f"    - Initial consistency loss:   {consistency_losses[0]:.6f}")
    print(f"    - Final consistency loss:     {consistency_losses[-1]:.6f}")
    print(f"    - Loss reduction:             {consistency_losses[0] - consistency_losses[-1]:.6f}")
    print(f"\n  3D Reconstruction:")
    print(f"    - Successful triangulations:  {successful_triangulations}/{len(pair_indices)} pairs")
    print(f"    - Total triangulated points:  {len(triangulated_points_all) if len(triangulated_points_all) > 0 else 0}")
    print(f"\n  Reconstruction Accuracy:")
    if not np.isinf(avg_reprojection_error):
        print(f"    - Reprojection error (pred):  {avg_reprojection_error:.4f} pixels")
    else:
        print(f"    - Reprojection error (pred):  N/A (untrained model)")
    if not np.isinf(avg_reprojection_error_gt):
        print(f"    - Reprojection error (GT):    {avg_reprojection_error_gt:.4f} pixels (reference)")
    print(f"\n  Pose Estimation Accuracy:")
    print(f"    - Avg pose error (rotation):  {avg_pose_error:.6f} (Frobenius)")
    print(f"    - Min pose error:             {min(pose_errors):.6f}")
    print(f"    - Max pose error:             {max(pose_errors):.6f}")

    print("\n" + "=" * 70)
    print("PASS 4 DEMO COMPLETE")
    print("=" * 70)

    return {
        'num_images': num_images,
        'num_pairs': len(pair_indices),
        'successful_triangulations': successful_triangulations,
        'triangulated_points': len(triangulated_points_all) if len(triangulated_points_all) > 0 else 0,
        'reprojection_error': avg_reprojection_error,
        'reprojection_error_gt': avg_reprojection_error_gt,
        'pose_error': avg_pose_error,
        'pose_error_min': min(pose_errors),
        'pose_error_max': max(pose_errors),
        'consistency_loss_initial': consistency_losses[0],
        'consistency_loss_final': consistency_losses[-1]
    }


if __name__ == "__main__":
    results = end_to_end_demo()
    print("\n✓ Pass 4 end-to-end demo completed successfully!")
