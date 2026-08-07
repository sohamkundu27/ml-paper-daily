#!/usr/bin/env python3
"""
Pass 4 Demo: End-to-end PISA denoising with performance comparison.
This script ties together all passes: block-wise sparse attention, Taylor approximation,
diffusion timestep integration, and efficiency metrics.
"""

import torch
import sys
from pisa import (
    run_denoising_demo,
    benchmark_attention,
    count_attention_flops,
    DiffusionTransformer
)


def main():
    print("=" * 70)
    print("PISA Pass 4: End-to-End Denoising Demo")
    print("=" * 70)
    print()

    # Show the pipeline overview
    print("PISA: Piecewise Sparse Attention for Efficient Diffusion Transformers")
    print("-" * 70)
    print("Pass 1: Block-wise sparse attention with critical block detection")
    print("Pass 2: Taylor expansion approximation for non-critical blocks")
    print("Pass 3: Diffusion transformer with timestep embedding and efficiency metrics")
    print("Pass 4: End-to-end denoising demo on synthetic data")
    print()

    # Part 1: Run sparse denoising demo
    print("Part 1: Training with Sparse Attention")
    print("-" * 70)
    sparse_results = run_denoising_demo(use_sparse=True, num_epochs=3, batch_size=4)
    print(f"  Epochs: {sparse_results['num_epochs']}")
    print(f"  Losses per epoch: {[f'{l:.4f}' for l in sparse_results['losses']]}")
    print(f"  Final loss: {sparse_results['final_loss']:.4f}")
    print()

    # Part 2: Run dense denoising demo
    print("Part 2: Training with Dense Attention (baseline)")
    print("-" * 70)
    dense_results = run_denoising_demo(use_sparse=False, num_epochs=3, batch_size=4)
    print(f"  Epochs: {dense_results['num_epochs']}")
    print(f"  Losses per epoch: {[f'{l:.4f}' for l in dense_results['losses']]}")
    print(f"  Final loss: {dense_results['final_loss']:.4f}")
    print()

    # Part 3: Efficiency comparison
    print("Part 3: Efficiency Metrics (FLOPs & Latency)")
    print("-" * 70)
    batch_size = 4
    seq_len = 64
    dim = 64
    num_heads = 4

    print(f"  Configuration: batch={batch_size}, seq_len={seq_len}, dim={dim}, heads={num_heads}")
    print()

    sparse_flops = count_attention_flops(
        batch_size, seq_len, dim, num_heads,
        use_sparse=True, sparsity_ratio=0.5, block_size=16
    )
    dense_flops = count_attention_flops(
        batch_size, seq_len, dim, num_heads,
        use_sparse=False
    )

    flops_reduction = 1.0 - (sparse_flops / dense_flops)
    print(f"  Dense attention FLOPs: {dense_flops:.0f}")
    print(f"  Sparse attention FLOPs: {sparse_flops:.0f}")
    print(f"  FLOPs reduction: {flops_reduction*100:.1f}%")
    print()

    print(f"  Benchmarking latency (this may take a moment)...")
    benchmark_results = benchmark_attention(
        batch_size=batch_size,
        seq_len=seq_len,
        dim=dim,
        num_heads=num_heads,
        block_size=16,
        sparsity_ratio=0.5,
        num_runs=3
    )
    print(f"  Sparse latency: {benchmark_results['sparse_latency_ms']:.2f} ms")
    print(f"  Dense latency: {benchmark_results['dense_latency_ms']:.2f} ms")
    print(f"  Speedup: {benchmark_results['speedup']:.2f}x")
    print()

    # Part 4: Summary
    print("Part 4: Summary")
    print("-" * 70)
    print("PISA achieves efficiency through:")
    print("  1. Block-wise partitioning: divide sequence into manageable blocks")
    print("  2. Importance detection: identify critical blocks via attention score variance")
    print("  3. Piecewise computation: exact softmax for critical, Taylor approximation for rest")
    print("  4. Timestep integration: condition generation on diffusion timestep")
    print()
    print("Results demonstrate:")
    print(f"  • Sparse approach converges similarly to dense (loss {sparse_results['final_loss']:.4f} vs {dense_results['final_loss']:.4f})")
    print(f"  • {flops_reduction*100:.0f}% reduction in attention computation FLOPs")
    print(f"  • {benchmark_results['speedup']:.2f}x latency speedup on this hardware")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
