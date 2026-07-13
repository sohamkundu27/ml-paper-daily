import torch
import torch.nn as nn
import torch.optim as optim
from kan_layer import KANLayer


def test_kan_forward():
    """Test basic forward pass."""
    batch_size = 4
    in_features = 2
    out_features = 3

    layer = KANLayer(in_features, out_features, grid_size=5)
    x = torch.randn(batch_size, in_features)
    y = layer(x)

    assert y.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {y.shape}"
    print("✓ Forward pass test passed")


def test_kan_gradient():
    """Test that gradients flow through KAN layer."""
    batch_size = 4
    in_features = 2
    out_features = 1

    layer = KANLayer(in_features, out_features, grid_size=5)
    x = torch.randn(batch_size, in_features, requires_grad=True)
    y = layer(x)

    # Compute loss and backprop
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Gradient not computed for input"
    assert layer.control_points.grad is not None, "Gradient not computed for control points"
    print("✓ Gradient test passed")


def test_kan_learn_identity():
    """Test that KAN can learn the identity function (x -> x) on a single input."""
    torch.manual_seed(42)

    # Create a KAN layer that learns a simple function: input -> output
    layer = KANLayer(in_features=1, out_features=1, grid_size=10, degree=3)
    optimizer = optim.Adam(layer.parameters(), lr=0.01)

    # Generate training data: y = x (identity)
    x_train = torch.linspace(-1, 1, 100).unsqueeze(1)
    y_train = x_train.clone()

    # Train for a few iterations
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = layer(x_train)
        loss = nn.MSELoss()(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # Test that the learned function is close to identity
    with torch.no_grad():
        y_pred = layer(x_train)
        mse = nn.MSELoss()(y_pred, y_train).item()

    print(f"  MSE after training: {mse:.6f}")
    assert mse < 0.1, f"MSE too high: {mse}. Network failed to learn identity."
    print("✓ Learning test passed")


def test_kan_learn_nonlinear():
    """Test that KAN can learn a nonlinear function: y = x^2."""
    torch.manual_seed(42)

    layer = KANLayer(in_features=1, out_features=1, grid_size=15, degree=3)
    optimizer = optim.Adam(layer.parameters(), lr=0.01)

    # Generate training data: y = x^2
    x_train = torch.linspace(-1, 1, 100).unsqueeze(1)
    y_train = x_train ** 2

    # Train
    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = layer(x_train)
        loss = nn.MSELoss()(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # Check the fit
    with torch.no_grad():
        y_pred = layer(x_train)
        mse = nn.MSELoss()(y_pred, y_train).item()

    print(f"  MSE after training: {mse:.6f}")
    assert mse < 0.15, f"MSE too high: {mse}. Network failed to learn x^2."
    print("✓ Nonlinear learning test passed")


def test_kan_multi_feature():
    """Test KAN with multiple input and output features."""
    torch.manual_seed(42)

    layer = KANLayer(in_features=3, out_features=2, grid_size=10)
    optimizer = optim.Adam(layer.parameters(), lr=0.01)

    # Generate training data: y1 = x1 + x2, y2 = x3
    x_train = torch.randn(50, 3)
    y_train = torch.cat([
        (x_train[:, 0:1] + x_train[:, 1:2]),
        x_train[:, 2:3]
    ], dim=1)

    # Train
    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = layer(x_train)
        loss = nn.MSELoss()(y_pred, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = layer(x_train)
        mse = nn.MSELoss()(y_pred, y_train).item()

    print(f"  MSE after training: {mse:.6f}")
    assert mse < 0.25, f"MSE too high: {mse}."
    print("✓ Multi-feature learning test passed")


if __name__ == "__main__":
    print("Running KAN layer tests...\n")
    test_kan_forward()
    test_kan_gradient()
    test_kan_learn_identity()
    test_kan_learn_nonlinear()
    test_kan_multi_feature()
    print("\n✓ All tests passed!")
