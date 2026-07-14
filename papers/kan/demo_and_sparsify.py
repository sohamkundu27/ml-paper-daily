import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from kan_network import KANNetwork


def create_regression_data(num_samples=200, seed=42, problem_type='polynomial'):
    """Create toy regression dataset.

    Args:
        num_samples: int, number of samples
        seed: int, random seed
        problem_type: str, 'polynomial', 'sine', or 'mixed'

    Returns:
        X: tensor of shape (num_samples, input_dim)
        y: tensor of shape (num_samples, 1)
    """
    torch.manual_seed(seed)

    if problem_type == 'polynomial':
        # y = x^3 - 2*x + 0.5
        x = torch.linspace(-2, 2, num_samples).unsqueeze(1)
        y = x**3 - 2*x + 0.5
        y += torch.randn_like(y) * 0.2

    elif problem_type == 'sine':
        # y = sin(x) + 0.1*x
        x = torch.linspace(-3*np.pi, 3*np.pi, num_samples).unsqueeze(1)
        y = torch.sin(x) + 0.1*x
        y += torch.randn_like(y) * 0.1

    elif problem_type == 'mixed':
        # y = sin(x) * cos(x) (more complex)
        x = torch.linspace(-2*np.pi, 2*np.pi, num_samples).unsqueeze(1)
        y = torch.sin(x) * torch.cos(x)
        y += torch.randn_like(y) * 0.1

    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    # Normalize x to [-1, 1]
    x = (x - x.mean()) / (x.std() + 1e-8)

    return x, y


def train_regressor(network, X_train, y_train, num_epochs=200, lr=0.01,
                    batch_size=32, verbose=True):
    """Train KAN network for regression.

    Args:
        network: KANNetwork instance
        X_train: tensor of shape (num_samples, input_dim)
        y_train: tensor of shape (num_samples, 1)
        num_epochs: int, number of training epochs
        lr: float, learning rate
        batch_size: int, batch size for training
        verbose: bool, whether to print progress

    Returns:
        losses: list of training losses
    """
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_samples = X_train.shape[0]
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        perm = torch.randperm(num_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            y_pred = network(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return losses


def evaluate_regressor(network, X_test, y_test):
    """Evaluate regression network on test data.

    Args:
        network: KANNetwork instance
        X_test: tensor of shape (num_samples, input_dim)
        y_test: tensor of shape (num_samples, 1)

    Returns:
        mse: float, mean squared error
        rmse: float, root mean squared error
    """
    with torch.no_grad():
        y_pred = network(X_test)
        mse = nn.MSELoss()(y_pred, y_test).item()
        rmse = np.sqrt(mse)
    return mse, rmse


def compute_edge_importance(network):
    """Compute importance of each edge based on control point magnitudes.

    The importance of an edge is the L2 norm of its control points,
    normalized by layer. Edges with small importance can be pruned.

    Args:
        network: KANNetwork instance

    Returns:
        importance_by_layer: list of (layer_idx, in_idx, out_idx, importance)
    """
    importance_list = []

    for layer_idx, layer in enumerate(network.layers):
        # Get control points: shape (out_features, in_features, num_basis)
        ctrl_pts = layer.control_points.data

        for out_idx in range(layer.out_features):
            for in_idx in range(layer.in_features):
                # L2 norm of control points for this edge
                edge_importance = torch.norm(ctrl_pts[out_idx, in_idx]).item()
                importance_list.append({
                    'layer': layer_idx,
                    'in_idx': in_idx,
                    'out_idx': out_idx,
                    'importance': edge_importance
                })

    return importance_list


def prune_edges(network, threshold=0.1):
    """Zero out edges with importance below threshold.

    This is a soft sparsification: we zero out control points for weak edges,
    effectively removing their contribution to the network.

    Args:
        network: KANNetwork instance
        threshold: float, importance threshold for pruning

    Returns:
        pruned_count: int, number of edges pruned
        importance_list: list of importance values
    """
    importance_list = compute_edge_importance(network)

    # Find threshold value
    importances = [item['importance'] for item in importance_list]
    threshold_value = np.percentile(importances, threshold * 100)

    pruned_count = 0
    with torch.no_grad():
        for layer_idx, layer in enumerate(network.layers):
            for out_idx in range(layer.out_features):
                for in_idx in range(layer.in_features):
                    edge_importance = layer.control_points[out_idx, in_idx].norm().item()
                    if edge_importance < threshold_value:
                        layer.control_points[out_idx, in_idx].zero_()
                        pruned_count += 1

    return pruned_count, importance_list


def run_regression_demo(problem_type='polynomial', verbose=True):
    """Run end-to-end regression demo with KAN network.

    Args:
        problem_type: str, 'polynomial', 'sine', or 'mixed'
        verbose: bool, whether to print detailed output

    Returns:
        results: dict with performance metrics and pruning stats
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"KAN Regression Demo: {problem_type}")
        print(f"{'='*60}\n")

    # Create data
    X_train, y_train = create_regression_data(
        num_samples=200,
        seed=42,
        problem_type=problem_type
    )
    X_test, y_test = create_regression_data(
        num_samples=100,
        seed=43,
        problem_type=problem_type
    )

    if verbose:
        print(f"Data shapes: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"              X_test {X_test.shape}, y_test {y_test.shape}\n")

    # Create network
    network = KANNetwork(
        layer_sizes=[1, 16, 8, 1],
        grid_size=8,
        use_activation=True,
        grid_range=(-1.0, 1.0)
    )

    if verbose:
        print("Network architecture: [1] -> [16] -> [8] -> [1]\n")
        print("Training...")

    # Train
    losses = train_regressor(
        network,
        X_train, y_train,
        num_epochs=150,
        lr=0.05,
        batch_size=32,
        verbose=verbose
    )

    # Evaluate before pruning
    mse_before, rmse_before = evaluate_regressor(network, X_test, y_test)
    if verbose:
        print(f"\nBefore pruning:")
        print(f"  Test MSE: {mse_before:.6f}")
        print(f"  Test RMSE: {rmse_before:.6f}")

    # Compute importance
    importance_list = compute_edge_importance(network)
    importances = [item['importance'] for item in importance_list]
    if verbose:
        print(f"\nEdge importance statistics:")
        print(f"  Min: {min(importances):.6f}")
        print(f"  Max: {max(importances):.6f}")
        print(f"  Mean: {np.mean(importances):.6f}")
        print(f"  Median: {np.median(importances):.6f}")

    # Prune edges (remove bottom 30% by importance)
    pruned_count, _ = prune_edges(network, threshold=0.30)
    if verbose:
        print(f"\nPruning edges with importance in bottom 30%...")
        print(f"  Edges pruned: {pruned_count}")

    # Evaluate after pruning
    mse_after, rmse_after = evaluate_regressor(network, X_test, y_test)
    if verbose:
        print(f"\nAfter pruning:")
        print(f"  Test MSE: {mse_after:.6f}")
        print(f"  Test RMSE: {rmse_after:.6f}")
        print(f"  MSE increase: {(mse_after - mse_before) / mse_before * 100:.2f}%")

    results = {
        'problem_type': problem_type,
        'mse_before_pruning': mse_before,
        'rmse_before_pruning': rmse_before,
        'mse_after_pruning': mse_after,
        'rmse_after_pruning': rmse_after,
        'edges_pruned': pruned_count,
        'total_edges': len(importance_list),
        'sparsity_ratio': pruned_count / len(importance_list),
        'losses': losses
    }

    if verbose:
        print(f"\nSummary:")
        print(f"  Sparsity: {results['sparsity_ratio']*100:.1f}% of edges pruned")
        print(f"  Total edges: {results['total_edges']}")

    return results


def main():
    """Run complete end-to-end demo on all problem types."""
    print("\n" + "="*60)
    print("KAN: End-to-End Regression Demo with Sparsification")
    print("="*60)

    results_list = []
    for problem_type in ['polynomial', 'sine', 'mixed']:
        results = run_regression_demo(problem_type, verbose=True)
        results_list.append(results)

    # Print summary
    print("\n" + "="*60)
    print("Summary across all problems:")
    print("="*60 + "\n")
    print(f"{'Problem':<12} {'MSE (before)':<15} {'MSE (after)':<15} {'Sparsity':<10}")
    print("-" * 55)
    for results in results_list:
        print(f"{results['problem_type']:<12} "
              f"{results['mse_before_pruning']:<15.6f} "
              f"{results['mse_after_pruning']:<15.6f} "
              f"{results['sparsity_ratio']*100:<9.1f}%")

    print("\nAll demos completed successfully!")
    return results_list


if __name__ == "__main__":
    results = main()
