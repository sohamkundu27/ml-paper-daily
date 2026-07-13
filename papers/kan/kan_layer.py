import torch
import torch.nn as nn


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network layer.

    This layer replaces a traditional linear layer + activation with learnable
    B-spline univariate functions on each edge.
    """

    def __init__(self, in_features, out_features, grid_size=5, degree=1,
                 grid_range=(-1.0, 1.0)):
        """Initialize KAN layer.

        Args:
            in_features: int, number of input features
            out_features: int, number of output features
            grid_size: int, number of grid points per input dimension
            degree: int, degree of B-spline (default 1 = linear)
            grid_range: tuple, (min, max) for grid normalization
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.degree = degree
        self.grid_min, self.grid_max = grid_range

        # Number of basis functions: for linear splines, it's grid_size
        self.num_basis = grid_size

        # Control points for each output dimension for each input
        # Shape: (out_features, in_features, num_basis)
        self.control_points = nn.Parameter(
            torch.randn(out_features, in_features, self.num_basis) * 0.1
        )

        # Bias term for each output
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        """Forward pass through KAN layer.

        Args:
            x: tensor of shape (batch_size, in_features)

        Returns:
            tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        output = torch.zeros(batch_size, self.out_features, device=device, dtype=dtype)

        # For each input-output feature pair
        for out_idx in range(self.out_features):
            for in_idx in range(self.in_features):
                # Get input values
                x_in = x[:, in_idx]  # (batch_size,)

                # Compute B-spline basis (fully vectorized)
                basis = self._compute_linear_basis(x_in)  # (batch_size, num_basis)

                # Get control points
                ctrl = self.control_points[out_idx, in_idx]  # (num_basis,)

                # Compute weighted sum
                contribution = torch.matmul(basis, ctrl)  # (batch_size,)
                output[:, out_idx] = output[:, out_idx] + contribution

        # Add bias
        output = output + self.bias.unsqueeze(0)
        return output

    def _compute_linear_basis(self, x):
        """Compute piecewise linear B-spline basis (fully differentiable).

        Uses only tensor operations, no loops or .item() calls.

        Args:
            x: tensor of shape (batch_size,)

        Returns:
            tensor of shape (batch_size, num_basis)
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Normalize x to [0, num_basis - 1]
        x_norm = (x - self.grid_min) / (self.grid_max - self.grid_min) * (self.num_basis - 1)
        x_norm = torch.clamp(x_norm, 0, self.num_basis - 1 - 1e-7)

        # Create grid points
        grid = torch.arange(self.num_basis, dtype=dtype, device=device)  # [0, 1, 2, ..., num_basis-1]

        # Expand dimensions for broadcasting
        # x_norm: (batch_size,) -> (batch_size, 1)
        # grid: (num_basis,) -> (1, num_basis)
        x_expanded = x_norm.unsqueeze(1)  # (batch_size, 1)
        grid_expanded = grid.unsqueeze(0)  # (1, num_basis)

        # Compute distances: |x - grid_point|
        distances = torch.abs(x_expanded - grid_expanded)  # (batch_size, num_basis)

        # Linear tent function: max(0, 1 - distance)
        basis = torch.clamp(1.0 - distances, min=0.0)  # (batch_size, num_basis)

        return basis
