import torch
import torch.nn as nn
import torch.optim as optim
from kan_layer import KANLayer
from kan_network import KANNetwork
from kan_classifier import KANClassifier, create_toy_classification_data, train_classifier, evaluate_classifier


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


def test_kan_network_forward():
    """Test KANNetwork forward pass with multiple layers."""
    batch_size = 4
    layer_sizes = [2, 4, 3, 1]

    network = KANNetwork(layer_sizes, grid_size=5, use_activation=True)
    x = torch.randn(batch_size, 2)
    y = network(x)

    assert y.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {y.shape}"
    print("✓ Network forward pass test passed")


def test_kan_network_gradient():
    """Test that gradients flow through entire KANNetwork."""
    batch_size = 4
    layer_sizes = [2, 3, 1]

    network = KANNetwork(layer_sizes, grid_size=5)
    x = torch.randn(batch_size, 2, requires_grad=True)
    y = network(x)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Gradient not computed for input"
    for i, layer in enumerate(network.layers):
        assert layer.control_points.grad is not None, f"Gradient not computed for layer {i}"
    print("✓ Network gradient test passed")


def test_kan_network_learn():
    """Test that KANNetwork can learn a nonlinear function."""
    torch.manual_seed(42)

    # Create a 2-layer network
    network = KANNetwork(layer_sizes=[1, 8, 1], grid_size=10, use_activation=True)
    optimizer = optim.Adam(network.parameters(), lr=0.01)

    # Generate training data: y = x^2
    x_train = torch.linspace(-1, 1, 100).unsqueeze(1)
    y_train = x_train ** 2

    # Train
    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = network(x_train)
        loss = nn.MSELoss()(y_pred, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = network(x_train)
        mse = nn.MSELoss()(y_pred, y_train).item()

    print(f"  MSE after training: {mse:.6f}")
    assert mse < 0.2, f"MSE too high: {mse}. Network failed to learn x^2."
    print("✓ Network learning test passed")


def test_kan_grid_refinement():
    """Test grid refinement functionality."""
    torch.manual_seed(42)

    # Create network with small grid
    network = KANNetwork(layer_sizes=[1, 4, 1], grid_size=5, use_activation=False)

    x = torch.randn(10, 1)
    with torch.no_grad():
        y_before = network(x).clone()

    # Get initial grid info
    info_before = network.get_grid_info()
    assert info_before['grid_size'] == 5

    # Refine grid on layer 0
    network.refine_grid(layer_idx=0, new_grid_size=10)

    info_after = network.get_grid_info()
    assert info_after['grid_size'] == 10

    # Forward pass should still work
    with torch.no_grad():
        y_after = network(x)
    assert y_after.shape == y_before.shape
    print("✓ Grid refinement test passed")


def test_kan_network_with_activation():
    """Test KANNetwork with activation functions between layers."""
    batch_size = 8
    layer_sizes = [2, 4, 4, 1]

    # Network with activation
    network_act = KANNetwork(layer_sizes, grid_size=5, use_activation=True)
    # Network without activation
    network_no_act = KANNetwork(layer_sizes, grid_size=5, use_activation=False)

    x = torch.randn(batch_size, 2)
    y_act = network_act(x)
    y_no_act = network_no_act(x)

    assert y_act.shape == (batch_size, 1)
    assert y_no_act.shape == (batch_size, 1)
    print("✓ Network with/without activation test passed")


def test_kan_classifier_forward():
    """Test KANClassifier forward pass."""
    batch_size = 8
    input_dim = 2
    num_classes = 2

    classifier = KANClassifier(
        input_dim=input_dim,
        hidden_dims=[8, 4],
        num_classes=num_classes,
        grid_size=5,
        use_activation=True
    )

    x = torch.randn(batch_size, input_dim)
    logits = classifier(x)

    assert logits.shape == (batch_size, num_classes), \
        f"Expected shape {(batch_size, num_classes)}, got {logits.shape}"
    print("✓ Classifier forward pass test passed")


def test_kan_classifier_predict():
    """Test KANClassifier predict and predict_proba methods."""
    batch_size = 8
    input_dim = 2
    num_classes = 3

    classifier = KANClassifier(
        input_dim=input_dim,
        hidden_dims=[6],
        num_classes=num_classes,
        grid_size=5
    )

    x = torch.randn(batch_size, input_dim)

    # Test predict_proba
    probs = classifier.predict_proba(x)
    assert probs.shape == (batch_size, num_classes)
    assert torch.all(probs >= 0) and torch.all(probs <= 1), "Probabilities out of [0, 1] range"
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size)), "Probabilities don't sum to 1"

    # Test predict
    predictions = classifier.predict(x)
    assert predictions.shape == (batch_size,)
    assert torch.all(predictions >= 0) and torch.all(predictions < num_classes), \
        "Predictions out of valid class range"

    print("✓ Classifier predict methods test passed")


def test_toy_data_creation():
    """Test toy classification data creation."""
    for problem_type in ['moons', 'circles', 'xor']:
        X, y = create_toy_classification_data(
            num_samples=100,
            num_classes=2,
            input_dim=2,
            seed=42,
            problem_type=problem_type
        )

        assert X.shape == (100, 2), f"Expected X shape (100, 2), got {X.shape}"
        assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
        assert y.min() >= 0 and y.max() < 2, "Class labels out of range"

    print("✓ Toy data creation test passed")


def test_kan_classifier_training():
    """Test that KAN classifier can train on moons dataset."""
    torch.manual_seed(42)

    # Create toy data
    X_train, y_train = create_toy_classification_data(
        num_samples=200,
        num_classes=2,
        input_dim=2,
        seed=42,
        problem_type='moons'
    )

    # Create classifier
    classifier = KANClassifier(
        input_dim=2,
        hidden_dims=[16, 8],
        num_classes=2,
        grid_size=8,
        use_activation=True
    )

    # Train
    losses = train_classifier(
        classifier,
        X_train, y_train,
        num_epochs=150,
        lr=0.05,
        batch_size=32,
        verbose=False
    )

    # Check that loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"  Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

    # Check training accuracy
    with torch.no_grad():
        train_acc = evaluate_classifier(classifier, X_train, y_train)
    print(f"  Training accuracy: {train_acc:.4f}")
    assert train_acc > 0.6, f"Training accuracy too low: {train_acc}"

    print("✓ Classifier training test passed")


def test_kan_classifier_circles():
    """Test KAN classifier on circles problem (nonlinear separability)."""
    torch.manual_seed(42)

    # Create circles data (concentric circles - harder than moons)
    X_train, y_train = create_toy_classification_data(
        num_samples=200,
        num_classes=2,
        input_dim=2,
        seed=42,
        problem_type='circles'
    )

    # Create classifier
    classifier = KANClassifier(
        input_dim=2,
        hidden_dims=[16, 8],
        num_classes=2,
        grid_size=8,
        use_activation=True
    )

    # Train
    losses = train_classifier(
        classifier,
        X_train, y_train,
        num_epochs=200,
        lr=0.05,
        batch_size=32,
        verbose=False
    )

    # Evaluate
    with torch.no_grad():
        train_acc = evaluate_classifier(classifier, X_train, y_train)
    print(f"  Circles training accuracy: {train_acc:.4f}")
    assert train_acc > 0.6, f"Circles accuracy too low: {train_acc}"

    print("✓ Classifier circles test passed")


if __name__ == "__main__":
    print("Running KAN layer tests...\n")
    test_kan_forward()
    test_kan_gradient()
    test_kan_learn_identity()
    test_kan_learn_nonlinear()
    test_kan_multi_feature()

    print("\nRunning KAN network tests...\n")
    test_kan_network_forward()
    test_kan_network_gradient()
    test_kan_network_learn()
    test_kan_grid_refinement()
    test_kan_network_with_activation()

    print("\nRunning KAN classifier tests...\n")
    test_kan_classifier_forward()
    test_kan_classifier_predict()
    test_toy_data_creation()
    test_kan_classifier_training()
    test_kan_classifier_circles()

    print("\n✓ All tests passed!")
