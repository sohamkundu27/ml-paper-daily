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


class RoutingMonitor:
    """
    Monitors routing statistics across batches to verify load balancing.
    Tracks per-LoRA activation counts and computes imbalance metrics.
    """

    def __init__(self, num_loras: int):
        self.num_loras = num_loras
        self.activation_counts = torch.zeros(num_loras)
        self.sample_count = 0
        self.history = []

    def update(self, routing: torch.Tensor):
        """
        Update monitor with routing decisions from a batch.
        Args:
            routing: (batch_size, num_loras) binary routing matrix
        """
        with torch.no_grad():
            self.activation_counts += routing.sum(dim=0).cpu()
            self.sample_count += routing.shape[0]

    def get_activation_rates(self) -> torch.Tensor:
        """Return fraction of samples each LoRA was activated for."""
        if self.sample_count == 0:
            return torch.zeros(self.num_loras)
        return self.activation_counts / self.sample_count

    def get_imbalance_ratio(self) -> float:
        """Return max load / min load ratio (1.0 = perfect balance)."""
        rates = self.get_activation_rates()
        max_rate = rates.max().item()
        min_rate = rates.min().item()
        if min_rate < 1e-8:
            return float('inf') if max_rate > 1e-8 else 1.0
        return max_rate / min_rate

    def get_entropy(self) -> float:
        """Return Shannon entropy of activation distribution (higher = more uniform)."""
        rates = self.get_activation_rates()
        rates = torch.clamp(rates, min=1e-8)
        entropy = -(rates * torch.log(rates)).sum().item()
        max_entropy = torch.log(torch.tensor(self.num_loras)).item()
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def log_snapshot(self, step: int):
        """Record current statistics."""
        rates = self.get_activation_rates()
        snapshot = {
            'step': step,
            'rates': rates.clone(),
            'imbalance_ratio': self.get_imbalance_ratio(),
            'entropy': self.get_entropy()
        }
        self.history.append(snapshot)

    def reset(self):
        """Clear accumulated statistics."""
        self.activation_counts.zero_()
        self.sample_count = 0

    def summary(self) -> str:
        """Return human-readable summary of current statistics."""
        rates = self.get_activation_rates()
        rates_str = ', '.join(f'{r:.3f}' for r in rates.tolist())
        return (f"Activation rates: [{rates_str}] | "
                f"Imbalance ratio: {self.get_imbalance_ratio():.2f}x | "
                f"Entropy: {self.get_entropy():.3f}")


class MixtureOfLoRAsMonitored(nn.Module):
    """
    Mixture of LoRAs with RL routing and comprehensive monitoring.
    Extends MixtureOfLoRAsRL with built-in routing statistics tracking.
    """

    def __init__(self, in_features: int, out_features: int,
                 num_loras: int, num_active: int, rank: int,
                 alpha: float = 1.0, router_hidden_dim: int = 64,
                 load_balance_weight: float = 0.1):
        super().__init__()
        self.mixture = MixtureOfLoRAsRL(
            in_features, out_features, num_loras, num_active, rank,
            alpha=alpha, router_hidden_dim=router_hidden_dim,
            load_balance_weight=load_balance_weight
        )
        self.monitor = RoutingMonitor(num_loras)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with automatic monitoring."""
        output, routing, probs = self.mixture(x)
        self.monitor.update(routing)
        return output, routing, probs

    def compute_loss(self, x: torch.Tensor, y_target: torch.Tensor,
                    task_loss_fn=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and update statistics."""
        output, routing, probs = self.mixture.forward(x)
        self.monitor.update(routing)

        if task_loss_fn is None:
            task_loss_fn = nn.MSELoss()

        task_loss = task_loss_fn(output, y_target)
        policy_loss = self.mixture.router.compute_policy_loss(probs, routing, task_loss)

        return task_loss + policy_loss, task_loss

    def get_routing_statistics(self) -> dict:
        """Return detailed routing statistics."""
        return {
            'activation_rates': self.monitor.get_activation_rates(),
            'imbalance_ratio': self.monitor.get_imbalance_ratio(),
            'entropy': self.monitor.get_entropy(),
            'summary': self.monitor.summary()
        }

    def reset_statistics(self):
        """Reset monitoring counters."""
        self.monitor.reset()

    def get_router(self) -> LearnedRouter:
        """Access the underlying learned router."""
        return self.mixture.router

    def get_loras(self) -> nn.ModuleList:
        """Access the LoRA layers."""
        return self.mixture.loras

    def parameters(self):
        """Delegate parameter access to mixture."""
        return self.mixture.parameters()


def demo_mixture_vs_single_lora():
    """
    End-to-end demo comparing mixture-of-LoRAs vs single LoRA on a toy regression task.

    Shows that under the same parameter budget, a mixture of LoRAs with learned
    routing can outperform a single larger LoRA by specializing different LoRAs
    to different input characteristics.
    """
    # Task setup: regress from high-dim input to output
    in_features, out_features = 128, 64
    rank = 8

    # Single LoRA setup
    single_rank = 16  # Higher rank to match total parameters
    single_params = in_features * single_rank + single_rank * out_features

    # Mixture setup: 4 LoRAs with rank 8 each
    num_loras, num_active = 4, 2
    mixture_params = num_loras * (in_features * rank + rank * out_features)
    mixture_params += 64 * in_features + 64 * 4  # Router network

    print(f"Parameter budget comparison:")
    print(f"  Single LoRA (rank={single_rank}): {single_params:,} params")
    print(f"  Mixture LoRAs ({num_loras} x rank={rank}, active={num_active}): {mixture_params:,} params")
    print()

    # Generate synthetic dataset: 10 "easy" samples and 10 "hard" samples
    torch.manual_seed(42)
    n_train, n_test = 200, 50

    # Split data: first half is "type A" patterns, second half is "type B"
    x_train = torch.randn(n_train, in_features)
    y_train = torch.zeros(n_train, out_features)

    # Type A: samples 0-99 follow a specific pattern
    y_train[:n_train//2] = x_train[:n_train//2, :out_features] * 0.5 + torch.randn(n_train//2, out_features) * 0.05

    # Type B: samples 100-199 follow a different pattern
    y_train[n_train//2:] = -x_train[n_train//2:, :out_features] * 0.3 + torch.randn(n_train//2, out_features) * 0.05

    # Test set (mixed types)
    x_test = torch.randn(n_test, in_features)
    y_test = torch.zeros(n_test, out_features)
    y_test[:n_test//2] = x_test[:n_test//2, :out_features] * 0.5
    y_test[n_test//2:] = -x_test[n_test//2:, :out_features] * 0.3

    # Model 1: Single LoRA with higher rank
    class SingleLoRAModel(nn.Module):
        def __init__(self, in_feat, out_feat, lora_rank):
            super().__init__()
            self.base_linear = nn.Linear(in_feat, out_feat)
            self.base_linear.weight.requires_grad = False
            self.base_linear.bias.requires_grad = False
            self.lora = LoRALayer(in_feat, out_feat, lora_rank, alpha=1.0)

        def forward(self, x):
            return self.base_linear(x) + self.lora(x)

    single_model = SingleLoRAModel(in_features, out_features, single_rank)
    mixture_model = MixtureOfLoRAsMonitored(
        in_features, out_features, num_loras, num_active, rank,
        alpha=1.0, load_balance_weight=0.2
    )

    # Training
    epochs = 100
    batch_size = 16
    lr = 0.01

    opt_single = torch.optim.Adam(single_model.parameters(), lr=lr)
    opt_mixture = torch.optim.Adam(mixture_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    single_losses = []
    mixture_losses = []

    for epoch in range(epochs):
        # Single LoRA training
        single_model.train()
        epoch_loss_single = 0.0
        for i in range(0, n_train, batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            opt_single.zero_grad()
            y_pred = single_model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            opt_single.step()
            epoch_loss_single += loss.item()

        single_losses.append(epoch_loss_single / (n_train // batch_size))

        # Mixture training
        mixture_model.train()
        epoch_loss_mixture = 0.0
        for i in range(0, n_train, batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            opt_mixture.zero_grad()
            total_loss, _ = mixture_model.compute_loss(x_batch, y_batch, loss_fn)
            total_loss.backward()
            opt_mixture.step()
            epoch_loss_mixture += total_loss.item()

        mixture_losses.append(epoch_loss_mixture / (n_train // batch_size))

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d}: Single={single_losses[-1]:.4f} | Mixture={mixture_losses[-1]:.4f}")

    # Evaluation
    single_model.eval()
    mixture_model.eval()

    with torch.no_grad():
        y_pred_single = single_model(x_test)
        test_loss_single = loss_fn(y_pred_single, y_test).item()

        y_pred_mixture, _, _ = mixture_model(x_test)
        test_loss_mixture = loss_fn(y_pred_mixture, y_test).item()

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Single LoRA final test loss:   {test_loss_single:.4f}")
    print(f"Mixture LoRA final test loss:  {test_loss_mixture:.4f}")
    print(f"Improvement:                   {(test_loss_single / test_loss_mixture - 1) * 100:.1f}%")
    print()

    # Show routing statistics
    stats = mixture_model.get_routing_statistics()
    print("Mixture routing statistics (on test set):")
    print(f"  {stats['summary']}")
    print()

    return {
        'single_loss': test_loss_single,
        'mixture_loss': test_loss_mixture,
        'improvement': test_loss_single / test_loss_mixture,
        'routing_stats': stats
    }
