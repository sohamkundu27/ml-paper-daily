import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


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


class LearnedRouter(nn.Module):
    """
    A learned router using policy gradient with RLOO-style baseline and load balancing.
    The router learns a policy network to select which LoRAs to activate based on input features.
    """

    def __init__(self, in_features: int, num_loras: int, num_active: int,
                 hidden_dim: int = 64, load_balance_weight: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.num_loras = num_loras
        self.num_active = num_active
        self.load_balance_weight = load_balance_weight

        # Policy network: takes input features and outputs logits for each LoRA
        self.policy_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_loras)
        )

        # Load tracking for statistics
        self.register_buffer('cumulative_load', torch.zeros(num_loras))
        self.register_buffer('load_count', torch.tensor(0, dtype=torch.long))

    def get_routing(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get routing decisions and probabilities.
        Args:
            x: input features (batch_size, in_features)
        Returns:
            routing: (batch_size, num_loras) hard routing decisions (binary)
            probs: (batch_size, num_loras) soft routing probabilities
        """
        logits = self.policy_net(x)
        probs = F.softmax(logits, dim=1)

        # Select top-k LoRAs deterministically
        batch_size = x.shape[0]
        routing = torch.zeros_like(probs)
        for i in range(batch_size):
            _, top_indices = torch.topk(probs[i], self.num_active)
            routing[i, top_indices] = 1.0

        return routing, probs

    def compute_policy_loss(self, probs: torch.Tensor, routing: torch.Tensor,
                          task_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute policy gradient loss using RLOO-style baseline.
        Lower task loss is better (reward), so we use negative loss as reward.

        Args:
            probs: (batch_size, num_loras) routing probabilities from policy
            routing: (batch_size, num_loras) hard routing decisions
            task_loss: (batch_size,) or scalar task loss for each sample

        Returns:
            policy_loss: scalar loss for policy gradient
        """
        # Ensure task_loss is per-sample
        if task_loss.dim() == 0:
            task_loss = task_loss.unsqueeze(0).expand(routing.shape[0])

        # Compute log probabilities
        log_probs = torch.log(probs + 1e-8)

        # Compute advantage using RLOO-style baseline
        # For each sample, baseline is the mean loss (reward)
        baseline = task_loss.mean().detach()
        # Reward is negative loss (higher loss = lower reward)
        reward = -task_loss.detach()
        advantage = reward - baseline

        # Policy loss: -E[log(π) * advantage]
        # Select log-probs for routed LoRAs
        selected_log_probs = (routing * log_probs).sum(dim=1)
        policy_loss = -(selected_log_probs * advantage).mean()

        # Load balancing loss: encourage uniform activation across LoRAs
        avg_selection = routing.mean(dim=0)
        # Entropy regularization: minimize entropy of selection (encourage all LoRAs equally)
        # Target is uniform: 1/num_loras probability for each
        uniform = torch.ones(self.num_loras, device=probs.device) / self.num_loras
        load_balance_loss = F.kl_div(
            torch.log(avg_selection + 1e-8),
            uniform,
            reduction='batchmean'
        )

        total_loss = policy_loss + self.load_balance_weight * load_balance_loss

        # Update load statistics
        with torch.no_grad():
            self.cumulative_load += routing.sum(dim=0)
            self.load_count += routing.shape[0]

        return total_loss

    def get_load_statistics(self) -> torch.Tensor:
        """Return normalized load per LoRA."""
        if self.load_count == 0:
            return torch.zeros(self.num_loras, device=self.cumulative_load.device)
        return self.cumulative_load / (self.load_count * self.num_active)

    def reset_load_statistics(self):
        """Reset load tracking counters."""
        self.cumulative_load.zero_()
        self.load_count.zero_()


class MixtureOfLoRAsRL(nn.Module):
    """
    A mixture of LoRAs with learned RL-based routing.
    Combines multiple LoRA layers with a learned router that optimizes routing via policy gradient.
    """

    def __init__(self, in_features: int, out_features: int,
                 num_loras: int, num_active: int, rank: int,
                 alpha: float = 1.0, router_hidden_dim: int = 64,
                 load_balance_weight: float = 0.1):
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

        # Learned router with policy gradient
        self.router = LearnedRouter(in_features, num_loras, num_active,
                                   hidden_dim=router_hidden_dim,
                                   load_balance_weight=load_balance_weight)

        # Base frozen linear layer
        self.base_linear = nn.Linear(in_features, out_features)
        self.base_linear.weight.requires_grad = False
        self.base_linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with learned routing.
        Args:
            x: input tensor of shape (batch_size, in_features)
        Returns:
            output: (batch_size, out_features)
            routing: (batch_size, num_loras) hard routing decisions
            probs: (batch_size, num_loras) routing probabilities for policy gradient
        """
        batch_size = x.shape[0]

        # Get routing from learned policy
        routing, probs = self.router.get_routing(x)
        if x.is_cuda:
            routing = routing.to(x.device)
            probs = probs.to(x.device)

        # Compute base output
        base_out = self.base_linear(x)

        # Compute LoRA contributions and route them
        lora_out = torch.zeros_like(base_out)
        for i, lora in enumerate(self.loras):
            lora_contribution = lora(x)  # (batch_size, out_features)
            # Weight by routing decision and average over active LoRAs
            weighted = routing[:, i:i+1] * lora_contribution / max(self.num_active, 1)
            lora_out = lora_out + weighted

        output = base_out + lora_out
        return output, routing, probs

    def compute_loss(self, x: torch.Tensor, y_target: torch.Tensor,
                    task_loss_fn=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total loss including task loss and policy gradient loss.

        Args:
            x: input features (batch_size, in_features)
            y_target: target output (batch_size, out_features)
            task_loss_fn: loss function (default: MSE)

        Returns:
            total_loss: task loss + policy loss
            task_loss: just the task loss for monitoring
        """
        if task_loss_fn is None:
            task_loss_fn = nn.MSELoss()

        # Forward pass
        output, routing, probs = self.forward(x)

        # Compute task loss
        task_loss = task_loss_fn(output, y_target)

        # Compute policy gradient loss
        policy_loss = self.router.compute_policy_loss(probs, routing, task_loss)

        return task_loss + policy_loss, task_loss
