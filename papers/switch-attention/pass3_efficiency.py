"""
Pass 3: Efficiency analysis and optimization.

Demonstrates:
1. Sparsity regularization to encourage binary routing decisions
2. Benchmarking of compute efficiency across different sequence lengths
3. Trade-offs between routing entropy and accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from switch_attention import SwitchAttention, FullAttention, SlidingWindowAttention


def create_synthetic_task(batch_size, seq_len, d_model, num_important=5):
    """Create synthetic data with target routing labels."""
    x = torch.randn(batch_size, seq_len, d_model)
    token_importance = x.norm(dim=2)

    labels = torch.zeros(batch_size, seq_len, 1, device=x.device)
    for b in range(batch_size):
        top_k_indices = torch.topk(token_importance[b], num_important).indices
        labels[b, top_k_indices, 0] = 1.0

    return x, labels


def train_with_sparsity(num_epochs=50, learning_rate=1e-3, batch_size=4,
                        seq_len=32, d_model=64, window_size=8, sparsity_weight=0.1):
    """Train SwitchAttention with sparsity regularization."""
    print("=" * 70)
    print("Pass 3: Training with Sparsity Regularization")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = SwitchAttention(d_model, num_heads=4, window_size=window_size,
                           sparsity_weight=sparsity_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    print(f"Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Window size: {window_size}")
    print(f"  Sparsity weight (λ): {sparsity_weight}\n")

    routing_losses = []
    sparsity_losses = []
    total_losses = []
    sparsities = []

    model.train()
    for epoch in range(num_epochs):
        x, labels = create_synthetic_task(batch_size, seq_len, d_model, num_important=5)
        x = x.to(device)
        labels = labels.to(device)

        output = model(x)
        routing_probs = model.get_routing_decisions(x)

        routing_loss = criterion(routing_probs, labels)
        sparsity_loss = model.routing_loss
        total_loss = routing_loss + sparsity_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        routing_losses.append(routing_loss.item())
        sparsity_losses.append(sparsity_loss.item())
        total_losses.append(total_loss.item())

        with torch.no_grad():
            sparsity = model.get_routing_sparsity(x)
            sparsities.append(sparsity)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} - Routing: {routing_loss.item():.4f}, "
                  f"Sparsity: {sparsity_loss.item():.4f}, Total: {total_loss.item():.4f}, "
                  f"Full-attn tokens: {sparsity:.2%}")

    print(f"\nTraining Results:")
    print(f"  Routing loss: {routing_losses[0]:.4f} → {routing_losses[-1]:.4f}")
    print(f"  Sparsity loss: {sparsity_losses[0]:.4f} → {sparsity_losses[-1]:.4f}")
    print(f"  Full-attn routing: {sparsities[0]:.2%} → {sparsities[-1]:.2%}")
    print()

    return model, routing_losses, sparsity_losses, total_losses, sparsities


def benchmark_attention_methods(d_model=64, num_heads=4, window_size=8,
                               sequence_lengths=[32, 64, 128, 256, 512],
                               batch_size=4):
    """Benchmark different attention mechanisms across sequence lengths."""
    print("=" * 70)
    print("Pass 3: Efficiency Benchmarks")
    print("=" * 70)
    print(f"Configuration: d_model={d_model}, num_heads={num_heads}, "
          f"window_size={window_size}, batch_size={batch_size}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models
    full_attn = FullAttention(d_model, num_heads).to(device)
    sliding_attn = SlidingWindowAttention(d_model, num_heads, window_size).to(device)
    switch_attn = SwitchAttention(d_model, num_heads, window_size).to(device)

    full_attn.eval()
    sliding_attn.eval()
    switch_attn.eval()

    results = {
        'seq_len': [],
        'full_attn_time': [],
        'sliding_attn_time': [],
        'switch_attn_time': [],
        'switch_speedup': []
    }

    print(f"{'Seq Len':<10} {'Full (ms)':<12} {'Sliding (ms)':<14} {'Switch (ms)':<12} {'Speedup':<8}")
    print("-" * 70)

    with torch.no_grad():
        for seq_len in sequence_lengths:
            x = torch.randn(batch_size, seq_len, d_model).to(device)

            # Warm up
            _ = full_attn(x)
            _ = sliding_attn(x)
            _ = switch_attn(x)

            # Benchmark full attention
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            for _ in range(10):
                _ = full_attn(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            full_time = (time.time() - t0) / 10 * 1000

            # Benchmark sliding window attention
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            for _ in range(10):
                _ = sliding_attn(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            sliding_time = (time.time() - t0) / 10 * 1000

            # Benchmark switch attention (interpolation, both computed)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            for _ in range(10):
                _ = switch_attn(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            switch_time = (time.time() - t0) / 10 * 1000

            # Compute speedup vs full attention
            speedup = full_time / switch_time if switch_time > 0 else 0

            results['seq_len'].append(seq_len)
            results['full_attn_time'].append(full_time)
            results['sliding_attn_time'].append(sliding_time)
            results['switch_attn_time'].append(switch_time)
            results['switch_speedup'].append(speedup)

            print(f"{seq_len:<10} {full_time:<12.3f} {sliding_time:<14.3f} "
                  f"{switch_time:<12.3f} {speedup:<8.3f}x")

    print()
    return results


def analyze_routing_efficiency(model, num_samples=5, seq_len=128, d_model=64):
    """Analyze what fraction of tokens are routed to each path."""
    print("=" * 70)
    print("Pass 3: Routing Efficiency Analysis")
    print("=" * 70 + "\n")

    device = next(model.parameters()).device
    model.eval()

    sparsities = []
    with torch.no_grad():
        for sample_idx in range(num_samples):
            x = torch.randn(1, seq_len, d_model).to(device)
            routing_probs = model.get_routing_decisions(x).squeeze(-1).squeeze(0)

            full_count = (routing_probs > 0.5).float().mean().item()
            sparsities.append(full_count)

            print(f"Sample {sample_idx + 1}:")
            print(f"  Routing to full attention: {full_count:.1%}")
            print(f"  Routing to sliding window: {1 - full_count:.1%}")

            entropy = -(routing_probs * torch.log(routing_probs + 1e-7) +
                       (1 - routing_probs) * torch.log(1 - routing_probs + 1e-7)).mean()
            print(f"  Routing entropy: {entropy:.3f}")
            print()

    print(f"Average full-attention routing: {np.mean(sparsities):.1%}")
    print()


def main():
    # Train with sparsity regularization
    model, routing_losses, sparsity_losses, total_losses, sparsities = train_with_sparsity(
        num_epochs=50, learning_rate=1e-3, batch_size=4, seq_len=32, d_model=64,
        window_size=8, sparsity_weight=0.1
    )

    # Benchmark efficiency
    results = benchmark_attention_methods(d_model=64, num_heads=4, window_size=8)

    # Analyze routing patterns
    analyze_routing_efficiency(model, num_samples=3, seq_len=128, d_model=64)

    print("=" * 70)
    print("✓ Pass 3 Efficiency Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
