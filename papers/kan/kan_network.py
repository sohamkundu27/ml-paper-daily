import torch
import torch.nn as nn
import numpy as np
from kan_layer import KANLayer


class KANNetwork(nn.Module):
    """Full Kolmogorov-Arnold Network composed of stacked KAN layers."""

    def __init__(self, layer_sizes, grid_size=5, degree=1,
                 use_activation=False, grid_range=(-1.0, 1.0)):
        """Initialize KAN network.

        Args:
            layer_sizes: list of ints specifying the architecture
                         e.g., [2, 8, 8, 1] for a 2->8->8->1 network
            grid_size: int, number of grid points per input dimension
            degree: int, degree of B-spline (not used in this pass, kept for future)
            use_activation: bool, whether to apply ReLU between layers
            grid_range: tuple, (min, max) for grid normalization
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.grid_size = grid_size
        self.degree = degree
        self.use_activation = use_activation
        self.grid_range = grid_range

        # Build KAN layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            layer = KANLayer(
                in_features=in_features,
                out_features=out_features,
                grid_size=grid_size,
                degree=degree,
                grid_range=grid_range
            )
            self.layers.append(layer)

        # Activation function between layers (if enabled)
        if use_activation:
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def forward(self, x):
        """Forward pass through the KAN network.

        Args:
            x: tensor of shape (batch_size, input_features)

        Returns:
            tensor of shape (batch_size, output_features)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation function between layers (but not after the last layer)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def refine_grid(self, layer_idx=None, new_grid_size=None):
        """Refine the B-spline grid by increasing grid size.

        This method increases the grid resolution for one or all layers,
        effectively increasing the number of basis functions used.

        Args:
            layer_idx: int or None. If int, refine only that layer.
                      If None, refine all layers.
            new_grid_size: int, new grid size. If None, use current grid_size * 2.
        """
        if new_grid_size is None:
            new_grid_size = self.grid_size * 2

        layers_to_refine = []
        if layer_idx is None:
            layers_to_refine = list(range(len(self.layers)))
        else:
            layers_to_refine = [layer_idx]

        for idx in layers_to_refine:
            layer = self.layers[idx]
            old_basis = layer.num_basis

            # Create a new layer with the larger grid size
            new_layer = KANLayer(
                in_features=layer.in_features,
                out_features=layer.out_features,
                grid_size=new_grid_size,
                degree=layer.degree,
                grid_range=(layer.grid_min, layer.grid_max)
            )

            # Transfer control points: interpolate old points into new finer grid
            with torch.no_grad():
                old_points = layer.control_points.data  # (out, in, old_basis)
                new_points = new_layer.control_points.data  # (out, in, new_basis)

                # Simple linear interpolation: map old grid points to new grid
                for out_idx in range(layer.out_features):
                    for in_idx in range(layer.in_features):
                        # Interpolate control points onto the finer grid
                        old_grid = np.linspace(0, 1, old_basis)
                        new_grid = np.linspace(0, 1, new_grid_size)

                        old_vals = old_points[out_idx, in_idx].cpu().numpy()
                        # Linear interpolation using numpy
                        new_vals = np.interp(new_grid, old_grid, old_vals)
                        new_points[out_idx, in_idx] = torch.from_numpy(new_vals).to(
                            device=new_points.device, dtype=new_points.dtype
                        )

                # Copy bias
                new_layer.bias.data = layer.bias.data.clone()

            self.layers[idx] = new_layer

        self.grid_size = new_grid_size

    def get_grid_info(self):
        """Return information about the grid configuration."""
        return {
            'grid_size': self.grid_size,
            'layer_sizes': self.layer_sizes,
            'num_layers': len(self.layers),
            'use_activation': self.use_activation,
        }
