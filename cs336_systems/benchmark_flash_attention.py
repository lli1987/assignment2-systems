"""
Benchmark FlashAttention-2 Triton implementation vs PyTorch baseline.

Compares forward, backward, and end-to-end latencies across various configurations:
- Sequence lengths: powers of 2 from 128 to 65536
- Embedding dimensions: powers of 2 from 16 to 128
- Precisions: bfloat16, float32
- Batch size: 1 (fixed)
- Causal masking: True (fixed)
"""

import argparse
import itertools
import torch
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple

try:
    import triton
    import triton.testing
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available. This benchmark requires Triton and a CUDA GPU.")

if TRITON_AVAILABLE:
    from cs336_systems.flash_attention import FlashAttentionTriton


def pytorch_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    """
    Standard PyTorch attention implementation (non-Flash).

    Args:
        Q: Query tensor (B, N, d)
        K: Key tensor (B, N, d)
        V: Value tensor (B, N, d)
        is_causal: Whether to apply causal masking

    Returns:
        Output tensor (B, N, d)
    """
    B, N, d = Q.shape
    scale = d ** -0.5

    # Compute attention scores
    S = torch.matmul(Q, K.transpose(-1, -2)) * scale  # (B, N, N)

    # Apply causal mask
    if is_causal:
        causal_mask = torch.tril(torch.ones(N, N, device=Q.device, dtype=torch.bool))
        S = S.masked_fill(~causal_mask, float('-inf'))

    # Softmax
    P = torch.softmax(S, dim=-1)

    # Output
    O = torch.matmul(P, V)
    return O


def benchmark_forward(
    fn: Callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = True,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Benchmark forward pass using triton.testing.do_bench."""

    def forward_fn():
        return fn(Q, K, V, is_causal)

    ms = triton.testing.do_bench(forward_fn, warmup=warmup, rep=rep)
    return ms


def benchmark_backward(
    fn: Callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = True,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Benchmark backward pass using triton.testing.do_bench."""

    # Create grad output
    dO = torch.randn_like(Q)

    def backward_fn():
        Q.grad = None
        K.grad = None
        V.grad = None

        O = fn(Q, K, V, is_causal)
        O.backward(dO)

    ms = triton.testing.do_bench(backward_fn, warmup=warmup, rep=rep)
    return ms


def benchmark_end_to_end(
    fn: Callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = True,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Benchmark end-to-end forward + backward pass."""

    dO = torch.randn_like(Q)

    def e2e_fn():
        Q.grad = None
        K.grad = None
        V.grad = None

        O = fn(Q, K, V, is_causal)
        O.backward(dO)

    ms = triton.testing.do_bench(e2e_fn, warmup=warmup, rep=rep)
    return ms


def run_benchmark(
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: str = "cuda",
    batch_size: int = 1,
    is_causal: bool = True,
) -> dict:
    """
    Run benchmark for a specific configuration.

    Args:
        seq_len: Sequence length
        d_model: Embedding dimension
        dtype: Data type (torch.float32 or torch.bfloat16)
        device: Device to run on
        batch_size: Batch size (fixed at 1)
        is_causal: Whether to use causal masking (fixed at True)

    Returns:
        Dictionary with benchmark results
    """
    # Create random inputs
    Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    # Benchmark PyTorch implementation
    pytorch_fwd = benchmark_forward(pytorch_attention, Q, K, V, is_causal)
    pytorch_bwd = benchmark_backward(pytorch_attention, Q, K, V, is_causal)
    pytorch_e2e = benchmark_end_to_end(pytorch_attention, Q, K, V, is_causal)

    # Benchmark FlashAttention Triton implementation
    flash_fwd = benchmark_forward(FlashAttentionTriton.apply, Q, K, V, is_causal)
    flash_bwd = benchmark_backward(FlashAttentionTriton.apply, Q, K, V, is_causal)
    flash_e2e = benchmark_end_to_end(FlashAttentionTriton.apply, Q, K, V, is_causal)

    # Compute speedups
    speedup_fwd = pytorch_fwd / flash_fwd if flash_fwd > 0 else 0
    speedup_bwd = pytorch_bwd / flash_bwd if flash_bwd > 0 else 0
    speedup_e2e = pytorch_e2e / flash_e2e if flash_e2e > 0 else 0

    return {
        'seq_len': seq_len,
        'd_model': d_model,
        'dtype': str(dtype).split('.')[-1],  # 'float32' or 'bfloat16'
        'batch_size': batch_size,
        'pytorch_fwd_ms': pytorch_fwd,
        'pytorch_bwd_ms': pytorch_bwd,
        'pytorch_e2e_ms': pytorch_e2e,
        'flash_fwd_ms': flash_fwd,
        'flash_bwd_ms': flash_bwd,
        'flash_e2e_ms': flash_e2e,
        'speedup_fwd': speedup_fwd,
        'speedup_bwd': speedup_bwd,
        'speedup_e2e': speedup_e2e,
    }


def main():
    # Check Triton availability
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not available. This benchmark requires Triton.\n"
            "Triton is only available on Linux with CUDA GPUs.\n"
            "Please run this benchmark on a GPU machine with Triton installed."
        )

    parser = argparse.ArgumentParser(description="Benchmark FlashAttention-2 vs PyTorch")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=128,
        help="Minimum sequence length (power of 2)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=65536,
        help="Maximum sequence length (power of 2)",
    )
    parser.add_argument(
        "--min_d_model",
        type=int,
        default=16,
        help="Minimum embedding dimension (power of 2)",
    )
    parser.add_argument(
        "--max_d_model",
        type=int,
        default=128,
        help="Maximum embedding dimension (power of 2)",
    )

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate parameter ranges (powers of 2)
    seq_lens = [2 ** i for i in range(7, 17) if args.min_seq_len <= 2**i <= args.max_seq_len]  # 128 to 65536
    d_models = [2 ** i for i in range(4, 8) if args.min_d_model <= 2**i <= args.max_d_model]   # 16 to 128
    dtypes = [torch.float32, torch.bfloat16]

    # Fixed parameters
    batch_size = 1
    is_causal = True

    print(f"Benchmarking FlashAttention-2 (Triton) vs PyTorch")
    print(f"=" * 80)
    print(f"Sequence lengths: {seq_lens}")
    print(f"Embedding dimensions: {d_models}")
    print(f"Data types: {[str(dt).split('.')[-1] for dt in dtypes]}")
    print(f"Batch size: {batch_size} (fixed)")
    print(f"Causal masking: {is_causal} (fixed)")
    print(f"Device: {args.device}")
    print(f"=" * 80)
    print()

    # Run benchmarks
    results = []
    total_configs = len(seq_lens) * len(d_models) * len(dtypes)
    current = 0

    for seq_len, d_model, dtype in itertools.product(seq_lens, d_models, dtypes):
        current += 1
        dtype_str = str(dtype).split('.')[-1]

        print(f"[{current}/{total_configs}] Benchmarking: "
              f"seq_len={seq_len}, d_model={d_model}, dtype={dtype_str}")

        try:
            result = run_benchmark(
                seq_len=seq_len,
                d_model=d_model,
                dtype=dtype,
                device=args.device,
                batch_size=batch_size,
                is_causal=is_causal,
            )
            results.append(result)

            print(f"  PyTorch - Fwd: {result['pytorch_fwd_ms']:.3f}ms, "
                  f"Bwd: {result['pytorch_bwd_ms']:.3f}ms, "
                  f"E2E: {result['pytorch_e2e_ms']:.3f}ms")
            print(f"  Flash   - Fwd: {result['flash_fwd_ms']:.3f}ms, "
                  f"Bwd: {result['flash_bwd_ms']:.3f}ms, "
                  f"E2E: {result['flash_e2e_ms']:.3f}ms")
            print(f"  Speedup - Fwd: {result['speedup_fwd']:.2f}x, "
                  f"Bwd: {result['speedup_bwd']:.2f}x, "
                  f"E2E: {result['speedup_e2e']:.2f}x")
            print()

        except Exception as e:
            print(f"  FAILED: {e}")
            print()
            continue

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        output_path = output_dir / "flash_attention_benchmark.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

        # Print summary table
        print("\n" + "=" * 120)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 120)

        # Format for display
        display_df = df.copy()
        for col in ['pytorch_fwd_ms', 'pytorch_bwd_ms', 'pytorch_e2e_ms',
                    'flash_fwd_ms', 'flash_bwd_ms', 'flash_e2e_ms']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        for col in ['speedup_fwd', 'speedup_bwd', 'speedup_e2e']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}x")

        print(display_df.to_string(index=False))
        print("=" * 120)

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Average Forward Speedup:  {df['speedup_fwd'].mean():.2f}x")
        print(f"Average Backward Speedup: {df['speedup_bwd'].mean():.2f}x")
        print(f"Average E2E Speedup:      {df['speedup_e2e'].mean():.2f}x")
        print(f"Max Forward Speedup:      {df['speedup_fwd'].max():.2f}x")
        print(f"Max Backward Speedup:     {df['speedup_bwd'].max():.2f}x")
        print(f"Max E2E Speedup:          {df['speedup_e2e'].max():.2f}x")
    else:
        print("\nNo successful benchmarks completed.")


if __name__ == "__main__":
    main()
