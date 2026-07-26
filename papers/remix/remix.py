import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class LoRALayer(nn.Module):
    """
    A single LoRA layer that injects trainable low-rank matrices into a linear layer.
    Computes: output = W @ x + (alpha * A @ B @ x)
    where W is frozen, and A, B are trainable low-rank matrices.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Low-rank trainable matrices
        self.lora_a = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA: compute alpha * (x @ A @ B)
        x shape: (..., in_features)
        output shape: (..., out_features)
        """
        # Compute x @ A @ B
        # x: (..., in_features)
        # lora_a: (in_features, rank)
        # lora_b: (rank, out_features)
        xa = torch.matmul(x, self.lora_a)  # (..., rank)
        xab = torch.matmul(xa, self.lora_b)  # (..., out_features)
        return self.alpha * xab


class LoRALinear(nn.Module):
    """
    A linear layer with injected LoRA. Base weight W is frozen.
    Computes: output = W @ x + lora_layer(x)
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_linear = nn.Linear(in_features, out_features)
        # Freeze base weights
        self.base_linear.weight.requires_grad = False
        self.base_linear.bias.requires_grad = False

        self.lora = LoRALayer(in_features, out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = self.lora(x)
        return base_out + lora_out


class SimpleRouter(nn.Module):
    """
    A simple router that uniformly selects k LoRAs from n total LoRAs.
    This is the foundational router for Pass 1 - later passes will use learned routing.
    """

    def __init__(self, num_loras: int, num_active: int):
        super().__init__()
        self.num_loras = num_loras
        self.num_active = num_active

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Returns routing decisions: a binary matrix of shape (batch_size, num_loras)
        where each row has exactly num_active ones.
        """
        # For each sample, randomly select num_active LoRAs
        routing = torch.zeros(batch_size, self.num_loras, dtype=torch.float32)
        for i in range(batch_size):
            selected = torch.randperm(self.num_loras)[:self.num_active]
            routing[i, selected] = 1.0
        return routing


class MixtureOfLoRAs(nn.Module):
    """
    A mixture of multiple LoRA layers with routing.
    Each input is routed to a subset of LoRAs, which contribute equally to the output.
    """

    def __init__(self, in_features: int, out_features: int,
                 num_loras: int, num_active: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_loras = num_loras
        self.num_active = num_active

        # Create multiple LoRA layers
        self.loras = nn.ModuleList([
            LoRALayer(in_features, out_features, rank, alpha)
            for _ in range(num_loras)
        ])

        # Simple uniform router
        self.router = SimpleRouter(num_loras, num_active)

        # Base frozen linear layer
        self.base_linear = nn.Linear(in_features, out_features)
        self.base_linear.weight.requires_grad = False
        self.base_linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with mixture of LoRAs.
        Args:
            x: input tensor of shape (batch_size, in_features)
        Returns:
            output: (batch_size, out_features)
            routing: (batch_size, num_loras) routing decisions
        """
        batch_size = x.shape[0]

        # Get routing decisions
        routing = self.router(batch_size)
        if x.is_cuda:
            routing = routing.to(x.device)

        # Compute base output
        base_out = self.base_linear(x)

        # Compute LoRA contributions and route them
        lora_out = torch.zeros_like(base_out)
        for i, lora in enumerate(self.loras):
            lora_contribution = lora(x)  # (batch_size, out_features)
            # Weight by routing decision and number of active LoRAs for averaging
            weighted = routing[:, i:i+1] * lora_contribution / max(self.num_active, 1)
            lora_out = lora_out + weighted

        output = base_out + lora_out
        return output, routing
