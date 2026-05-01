import argparse
import torch
import logging
import itertools
import numpy as np
from pathlib import Path
from cs336_basics import model
from timeit import default_timer
import pandas as pd

logger = logging.getLogger(__name__)


def benchmark_attention(
    batch_size: int,
    d_model: int,
    seq_length: int,
    num_iterations: int = 100,
    num_warmup: int = 10,
    device: str = "cuda",
    use_compile: bool = False,
):
    """Benchmark attention at a specific scale.

    Args:
        batch_size: Batch size (fixed at 8)
        d_model: Head embedding dimension
        seq_length: Sequence length
        num_iterations: Number of iterations to time
        num_warmup: Number of warmup iterations
        device: Device to run on
        use_compile: Whether to use torch.compile

    Returns:
        Dictionary with benchmark results
    """
    # Create random inputs Q, K, V of shape (batch_size, seq_length, d_model)
    # Note: No head dimension as per requirement (a)
    Q = torch.randn(batch_size, seq_length, d_model, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_length, d_model, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_length, d_model, device=device, requires_grad=True)

    # Create causal mask
    mask = torch.tril(torch.ones(seq_length, seq_length, device=device, dtype=torch.bool))

    # Optionally compile the attention function
    attention_fn = model.scaled_dot_product_attention
    if use_compile:
        attention_fn = torch.compile(attention_fn)

    # Warmup
    for _ in range(num_warmup):
        output = attention_fn(Q, K, V, mask)
        loss = output.sum()
        loss.backward()
        Q.grad = None
        K.grad = None
        V.grad = None
        torch.cuda.synchronize()

    # Benchmark forward pass
    forward_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        t0 = default_timer()
        output = attention_fn(Q, K, V, mask)
        torch.cuda.synchronize()
        t1 = default_timer()
        forward_times.append(t1 - t0)

    # Measure memory before backward pass
    torch.cuda.synchronize()
    memory_before_backward = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB

    # Benchmark backward pass
    backward_times = []
    for _ in range(num_iterations):
        # Need to do forward pass for each backward
        output = attention_fn(Q, K, V, mask)
        loss = output.sum()

        torch.cuda.synchronize()
        t0 = default_timer()
        loss.backward()
        torch.cuda.synchronize()
        t1 = default_timer()
        backward_times.append(t1 - t0)

        # Clear gradients for next iteration
        Q.grad = None
        K.grad = None
        V.grad = None

    return {
        "batch_size": batch_size,
        "d_model": d_model,
        "seq_length": seq_length,
        "compiled": use_compile,
        "forward_mean_ms": np.mean(forward_times) * 1000,
        "forward_std_ms": np.std(forward_times) * 1000,
        "backward_mean_ms": np.mean(backward_times) * 1000,
        "backward_std_ms": np.std(backward_times) * 1000,
        "memory_before_backward_mb": memory_before_backward,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention at different scales")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of iterations to time (default: 100)",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results (default: ./benchmark_results)",
    )
    parser.add_argument(
        "--compare_compile",
        action="store_true",
        help="Compare compiled vs uncompiled versions",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define parameter ranges
    d_model_values = [16, 32, 64, 128]
    seq_length_values = [256, 1024, 4096, 8192, 16384]

    # Determine which versions to test
    compile_versions = [False, True] if args.compare_compile else [False]

    # Iterate through cartesian product
    results = []
    total_combinations = len(d_model_values) * len(seq_length_values) * len(compile_versions)
    current = 0

    for d_model, seq_length, use_compile in itertools.product(
        d_model_values, seq_length_values, compile_versions
    ):
        current += 1
        compile_str = "compiled" if use_compile else "uncompiled"
        logger.info(
            f"[{current}/{total_combinations}] Benchmarking ({compile_str}): "
            f"batch_size={args.batch_size}, d_model={d_model}, seq_length={seq_length}"
        )

        try:
            result = benchmark_attention(
                batch_size=args.batch_size,
                d_model=d_model,
                seq_length=seq_length,
                num_iterations=args.num_iterations,
                num_warmup=args.num_warmup,
                device=args.device,
                use_compile=use_compile,
            )
            results.append(result)

            logger.info(
                f"  Forward: {result['forward_mean_ms']:.4f} ± {result['forward_std_ms']:.4f} ms"
            )
            logger.info(
                f"  Backward: {result['backward_mean_ms']:.4f} ± {result['backward_std_ms']:.4f} ms"
            )
            logger.info(
                f"  Memory before backward: {result['memory_before_backward_mb']:.2f} MB"
            )
        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

    # Save results to CSV
    df = pd.DataFrame(results)
    output_path = output_dir / "attention_benchmark_results.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
