"""
Simple demo of Switch Attention module.
Shows how the hybrid attention computes outputs by routing between full and sliding window attention.
"""

import torch
from switch_attention import SwitchAttention


def main():
    print("=" * 60)
    print("Switch Attention Demo")
    print("=" * 60)

    # Create a simple example
    batch_size = 1
    seq_len = 32
    d_model = 64
    num_heads = 4
    window_size = 8

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Sliding window size: {window_size}")

    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")

    # Create SwitchAttention module
    attention = SwitchAttention(d_model, num_heads, window_size)
    attention.eval()

    with torch.no_grad():
        # Get routing probability
        routing_input = x.mean(dim=1)
        routing_prob = attention.router(routing_input)

        # Compute outputs
        full_output = attention.full_attention(x)
        sliding_output = attention.sliding_attention(x)
        hybrid_output = attention(x)

        print(f"\nRouting probability (full attention): {routing_prob.item():.4f}")
        print(f"Routing probability (sliding window): {1 - routing_prob.item():.4f}")

        print(f"\nOutput shapes:")
        print(f"  Full attention output: {full_output.shape}")
        print(f"  Sliding window output: {sliding_output.shape}")
        print(f"  Hybrid output: {hybrid_output.shape}")

        # Show some statistics
        print(f"\nOutput statistics:")
        print(f"  Full attention - mean: {full_output.mean().item():.4f}, std: {full_output.std().item():.4f}")
        print(f"  Sliding window - mean: {sliding_output.mean().item():.4f}, std: {sliding_output.std().item():.4f}")
        print(f"  Hybrid - mean: {hybrid_output.mean().item():.4f}, std: {hybrid_output.std().item():.4f}")

        # Verify interpolation
        reconstructed = routing_prob * full_output + (1 - routing_prob) * sliding_output
        diff = torch.abs(reconstructed - hybrid_output).max().item()
        print(f"\nInterpolation verification:")
        print(f"  Max difference between computed and expected: {diff:.2e}")

    print("\n✓ Demo completed successfully")


if __name__ == "__main__":
    main()
