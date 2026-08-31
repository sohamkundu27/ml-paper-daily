"""
Benchmark demonstrating relative speedup with sparse attention on real text.
Uses NLTK for automatic POS tagging and measures both theoretical (FLOP) and
practical (wall-clock) performance.
"""

import time
import torch
import torch.nn as nn
from typing import Tuple, Dict
from transformer_block import TransformerBlock
from sparse_attention import create_hard_mask, compute_sparsity
from transformer_attention import count_sparse_flops
from pos_tagger import tag_text


def embed_tokens(tokens: list, embed_dim: int) -> torch.Tensor:
    """Convert token list to random embeddings for demonstration."""
    seq_len = len(tokens)
    return torch.randn(1, seq_len, embed_dim)


def benchmark_dense_vs_sparse(
    text: str,
    embed_dim: int = 256,
    num_heads: int = 8,
    ffn_dim: int = 1024,
    num_blocks: int = 2,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """
    Benchmark dense vs sparse attention on a text sequence.

    Args:
        text: Input text to benchmark on.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward hidden dimension.
        num_blocks: Number of transformer blocks to stack.
        num_iterations: Number of forward/backward passes to average over.

    Returns:
        Dictionary with timing and efficiency metrics.
    """
    # Tag the text automatically
    tokens, pos_tags = tag_text(text)
    seq_len = len(tokens)

    if seq_len == 0:
        print("Empty text after tokenization")
        return {}

    # Embed tokens
    x = embed_tokens(tokens, embed_dim)

    # Create dense and sparse transformer blocks
    dense_block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        pos_tags=None,  # Dense attention
        mask_type="hard",
    )

    sparse_block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        pos_tags=pos_tags,  # Sparse attention with POS tags
        mask_type="hard",
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    dense_block = dense_block.to(device)
    sparse_block = sparse_block.to(device)

    # Compute sparsity
    hard_mask = create_hard_mask(pos_tags)
    sparsity = compute_sparsity(hard_mask)

    # Benchmark dense attention
    dense_block.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_iterations):
            _ = dense_block(x)
        dense_time = time.time() - start_time

    # Benchmark sparse attention
    sparse_block.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_iterations):
            _ = sparse_block(x)
        sparse_time = time.time() - start_time

    # Theoretical FLOP counting (attention only)
    dense_flops, sparse_flops = count_sparse_flops(
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparsity=sparsity,
    )

    # Speedup metrics
    wall_clock_speedup = dense_time / sparse_time if sparse_time > 0 else 0
    flop_reduction = (1 - sparse_flops / dense_flops) * 100 if dense_flops > 0 else 0

    results = {
        "seq_len": seq_len,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "sparsity": sparsity,
        "dense_time_ms": dense_time * 1000 / num_iterations,
        "sparse_time_ms": sparse_time * 1000 / num_iterations,
        "wall_clock_speedup": wall_clock_speedup,
        "dense_flops": dense_flops,
        "sparse_flops": sparse_flops,
        "flop_reduction_percent": flop_reduction,
    }

    return results


def print_benchmark_results(results: Dict[str, float]):
    """Pretty print benchmark results."""
    if not results:
        return

    print("\n" + "=" * 70)
    print("SPARSE ATTENTION BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nSequence Length: {results['seq_len']}")
    print(f"Tokens: {' '.join(results['tokens'])}")
    print(f"POS Tags: {' '.join(results['pos_tags'])}")

    print(f"\nSparsity: {results['sparsity']:.2%}")
    print(f"  (only {(1-results['sparsity']):.2%} of attention positions computed)")

    print(f"\nWall-Clock Performance (per iteration):")
    print(f"  Dense: {results['dense_time_ms']:.2f} ms")
    print(f"  Sparse: {results['sparse_time_ms']:.2f} ms")
    print(f"  Speedup: {results['wall_clock_speedup']:.2f}x")

    print(f"\nTheoretical FLOP Savings (attention layer only):")
    print(f"  Dense FLOPs: {results['dense_flops']:,}")
    print(f"  Sparse FLOPs: {results['sparse_flops']:,}")
    print(f"  Reduction: {results['flop_reduction_percent']:.1f}%")

    print("\n" + "=" * 70)


def demo_multiple_texts():
    """Benchmark on multiple example texts."""
    texts = [
        "The cat sat on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "Natural language processing is a fascinating field of artificial intelligence.",
    ]

    all_results = []
    for text in texts:
        print(f"\n>>> Benchmarking: {text}")
        results = benchmark_dense_vs_sparse(text)
        all_results.append(results)
        print_benchmark_results(results)

    # Summary across all texts
    if all_results:
        avg_speedup = sum(r.get('wall_clock_speedup', 0) for r in all_results) / len(all_results)
        avg_flop_reduction = sum(r.get('flop_reduction_percent', 0) for r in all_results) / len(all_results)

        print("\n" + "=" * 70)
        print("AVERAGE ACROSS ALL TEXTS")
        print("=" * 70)
        print(f"Average Wall-Clock Speedup: {avg_speedup:.2f}x")
        print(f"Average FLOP Reduction: {avg_flop_reduction:.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    demo_multiple_texts()
