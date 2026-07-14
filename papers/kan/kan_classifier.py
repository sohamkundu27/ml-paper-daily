import torch
import torch.nn as nn
import torch.optim as optim
from kan_network import KANNetwork


class KANClassifier(nn.Module):
    """KAN-based classifier for binary or multi-class classification."""

    def __init__(self, input_dim, hidden_dims, num_classes, grid_size=5,
                 use_activation=True, grid_range=(-1.0, 1.0)):
        """Initialize KAN classifier.

        Args:
            input_dim: int, number of input features
            hidden_dims: list of ints, sizes of hidden layers
            num_classes: int, number of output classes
            grid_size: int, number of grid points per input dimension
            use_activation: bool, whether to use ReLU between KAN layers
            grid_range: tuple, (min, max) for grid normalization
        """
        super().__init__()

        # Build layer architecture: input -> hidden -> output
        layer_sizes = [input_dim] + hidden_dims + [num_classes]

        self.network = KANNetwork(
            layer_sizes=layer_sizes,
            grid_size=grid_size,
            use_activation=use_activation,
            grid_range=grid_range
        )
        self.num_classes = num_classes

    def forward(self, x):
        """Forward pass through classifier.

        Args:
            x: tensor of shape (batch_size, input_dim)

        Returns:
            logits: tensor of shape (batch_size, num_classes)
        """
        return self.network(x)

    def predict_proba(self, x):
        """Return class probabilities.

        Args:
            x: tensor of shape (batch_size, input_dim)

        Returns:
            probs: tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x):
        """Return predicted class labels.

        Args:
            x: tensor of shape (batch_size, input_dim)

        Returns:
            predictions: tensor of shape (batch_size,)
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


def create_toy_classification_data(num_samples=200, num_classes=2, input_dim=2,
                                   seed=42, problem_type='moons'):
    """Create toy classification dataset.

    Args:
        num_samples: int, total number of samples
        num_classes: int, number of classes (2 or 3)
        input_dim: int, input dimensionality
        seed: int, random seed
        problem_type: str, 'moons', 'circles', or 'xor'

    Returns:
        X: tensor of shape (num_samples, input_dim)
        y: tensor of shape (num_samples,) with class labels
    """
    torch.manual_seed(seed)

    if problem_type == 'moons':
        # Two interleaving moons
        samples_per_class = num_samples // 2
        t = torch.linspace(0, torch.pi, samples_per_class)

        class_0 = torch.stack([
            torch.cos(t),
            torch.sin(t)
        ], dim=1)

        class_1 = torch.stack([
            1 - torch.cos(t),
            0.5 - torch.sin(t)
        ], dim=1)

        X = torch.cat([class_0, class_1], dim=0)
        y = torch.cat([
            torch.zeros(samples_per_class, dtype=torch.long),
            torch.ones(samples_per_class, dtype=torch.long)
        ], dim=0)

        # Add noise
        X += torch.randn_like(X) * 0.1

    elif problem_type == 'circles':
        # Concentric circles
        samples_per_class = num_samples // 2
        theta = torch.linspace(0, 2 * torch.pi, samples_per_class)

        class_0 = torch.stack([
            0.5 * torch.cos(theta),
            0.5 * torch.sin(theta)
        ], dim=1)

        class_1 = torch.stack([
            1.0 * torch.cos(theta),
            1.0 * torch.sin(theta)
        ], dim=1)

        X = torch.cat([class_0, class_1], dim=0)
        y = torch.cat([
            torch.zeros(samples_per_class, dtype=torch.long),
            torch.ones(samples_per_class, dtype=torch.long)
        ], dim=0)

        # Add noise
        X += torch.randn_like(X) * 0.05

    elif problem_type == 'xor':
        # XOR problem
        samples_per_quad = num_samples // 4
        q1 = torch.randn(samples_per_quad, 2) + torch.tensor([1.0, 1.0])
        q2 = torch.randn(samples_per_quad, 2) + torch.tensor([-1.0, -1.0])
        q3 = torch.randn(samples_per_quad, 2) + torch.tensor([1.0, -1.0])
        q4 = torch.randn(samples_per_quad, 2) + torch.tensor([-1.0, 1.0])

        X = torch.cat([q1, q2, q3, q4], dim=0)
        y = torch.cat([
            torch.zeros(samples_per_quad * 2, dtype=torch.long),
            torch.ones(samples_per_quad * 2, dtype=torch.long)
        ], dim=0)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    # Normalize to [-1, 1] range
    X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)

    return X, y


def train_classifier(classifier, X_train, y_train, num_epochs=200, lr=0.01,
                     batch_size=32, verbose=True):
    """Train KAN classifier.

    Args:
        classifier: KANClassifier instance
        X_train: tensor of shape (num_samples, input_dim)
        y_train: tensor of shape (num_samples,) with class labels
        num_epochs: int, number of training epochs
        lr: float, learning rate
        batch_size: int, batch size for training
        verbose: bool, whether to print progress

    Returns:
        losses: list of training losses
    """
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_samples = X_train.shape[0]
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle data
        perm = torch.randperm(num_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            logits = classifier(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return losses


def evaluate_classifier(classifier, X_test, y_test):
    """Evaluate classifier on test data.

    Args:
        classifier: KANClassifier instance
        X_test: tensor of shape (num_samples, input_dim)
        y_test: tensor of shape (num_samples,) with class labels

    Returns:
        accuracy: float, accuracy on test set
    """
    with torch.no_grad():
        predictions = classifier.predict(X_test)
        accuracy = (predictions == y_test).float().mean().item()
    return accuracy
