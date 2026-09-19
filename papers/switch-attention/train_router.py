"""
Pass 2: Training the learnable per-token router.

Demonstrates that the router learns to route tokens to appropriate attention mechanisms.
We use a synthetic task where certain token positions should prefer global context (full attention)
while others only need local context (sliding window attention).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from switch_attention import SwitchAttention


def create_synthetic_routing_task(batch_size, seq_len, d_model, num_important=5):
    """
    Create synthetic data where:
    - Input: random token embeddings
    - Labels: position-based ground truth routing (important tokens need full attention)

    Important tokens are spread throughout the sequence and should route to full attention.
    """
    x = torch.randn(batch_size, seq_len, d_model)

    # Create ground truth routing labels: 1 for full attention, 0 for sliding window
    # Important tokens are at positions with larger embeddings (deterministic)
    token_importance = x.norm(dim=2)  # (batch, seq_len)

    # Top-k tokens per sequence are marked as important
    labels = torch.zeros(batch_size, seq_len, 1, device=x.device)
    for b in range(batch_size):
        top_k_indices = torch.topk(token_importance[b], num_important).indices
        labels[b, top_k_indices, 0] = 1.0

    return x, labels


def train_router(num_epochs=50, learning_rate=1e-3, batch_size=4, seq_len=32, d_model=64, window_size=8):
    """
    Train the SwitchAttention router to learn meaningful routing patterns.
    """
    print("=" * 60)
    print("Training SwitchAttention Router (Pass 2)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Initialize model
    model = SwitchAttention(d_model, num_heads=4, window_size=window_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    print(f"Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Window size: {window_size}")
    print(f"  Learning rate: {learning_rate}\n")

    # Training loop
    losses = []
    for epoch in range(num_epochs):
        x, labels = create_synthetic_routing_task(batch_size, seq_len, d_model, num_important=5)
        x = x.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(x)
        routing_probs = model.get_routing_decisions(x)

        # Loss: how well does routing match the ground truth labels
        loss = criterion(routing_probs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.6f}")

    print(f"\nTraining completed!")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%\n")

    return model, losses


def evaluate_routing_patterns(model, num_samples=5):
    """
    Evaluate and visualize what the trained router learns.
    """
    print("=" * 60)
    print("Evaluating Learned Routing Patterns")
    print("=" * 60 + "\n")

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for sample_idx in range(num_samples):
            batch_size = 1
            seq_len = 32
            d_model = 64

            x = torch.randn(batch_size, seq_len, d_model).to(device)
            routing_probs = model.get_routing_decisions(x).squeeze(-1).squeeze(0)  # (seq_len,)

            # Compute importance scores (as in training)
            token_importance = x.norm(dim=2).squeeze(0)

            # Analyze correlation between token importance and routing
            correlation = torch.corrcoef(torch.stack([
                token_importance,
                routing_probs
            ]))[0, 1].item()

            print(f"Sample {sample_idx + 1}:")
            print(f"  Token importance range: [{token_importance.min():.3f}, {token_importance.max():.3f}]")
            print(f"  Routing probability range: [{routing_probs.min():.3f}, {routing_probs.max():.3f}]")
            print(f"  Correlation (importance vs full-attention routing): {correlation:.3f}")

            # Show which tokens prefer full attention (routing prob > 0.5)
            full_attn_tokens = (routing_probs > 0.5).sum().item()
            sliding_attn_tokens = (routing_probs <= 0.5).sum().item()
            print(f"  Full attention tokens: {full_attn_tokens}/{seq_len}")
            print(f"  Sliding window tokens: {sliding_attn_tokens}/{seq_len}")

            # Average routing probability of important tokens
            top_5_indices = torch.topk(token_importance, 5).indices
            top_5_routing = routing_probs[top_5_indices].mean().item()
            print(f"  Avg routing prob for top-5 important tokens: {top_5_routing:.3f}")
            print()


def main():
    # Train the router
    model, losses = train_router(num_epochs=50, learning_rate=1e-3)

    # Evaluate learned patterns
    evaluate_routing_patterns(model, num_samples=3)

    print("✓ Pass 2 training completed successfully!")


if __name__ == "__main__":
    main()
