"""
Benchmark all-reduce operations in single-node multi-process setup.

Compares performance across:
- Backends: Gloo (CPU) vs NCCL (GPU)
- Data sizes: 1MB, 10MB, 100MB, 1GB (float32)
- Number of processes: 2, 4, 6
"""

import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def setup_process_group(rank: int, world_size: int, backend: str, port: int = 29500):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # For NCCL, set device before init
    if backend == "nccl":
        torch.cuda.set_device(rank)

    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def benchmark_allreduce(
    rank: int,
    world_size: int,
    backend: str,
    data_size_mb: float,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    port: int = 29500,
) -> Tuple[float, float]:
    """
    Benchmark all-reduce operation.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Backend to use ('gloo' or 'nccl')
        data_size_mb: Size of data in MB
        warmup_iters: Number of warmup iterations
        benchmark_iters: Number of benchmark iterations
        port: Master port for communication

    Returns:
        Tuple of (mean_time_ms, std_time_ms) from rank 0, None for other ranks
    """
    setup_process_group(rank, world_size, backend, port)

    # Calculate tensor size (float32 = 4 bytes)
    num_elements = int(data_size_mb * 1024 * 1024 / 4)

    # Create tensor on appropriate device
    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
    else:  # gloo
        tensor = torch.randn(num_elements, dtype=torch.float32)

    # Warmup
    for _ in range(warmup_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if backend == "nccl":
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(benchmark_iters):
        if backend == "nccl":
            torch.cuda.synchronize()

        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if backend == "nccl":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Only rank 0 computes and returns statistics
    if rank == 0:
        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
        result = (mean_time, std_time)
    else:
        result = None

    cleanup()
    return result


def run_benchmark_wrapper(
    rank: int,
    world_size: int,
    backend: str,
    data_size_mb: float,
    warmup_iters: int,
    benchmark_iters: int,
    port: int,
    result_queue: mp.Queue,
):
    """
    Wrapper to run benchmark and put results in queue.
    This is needed because mp.spawn doesn't return values directly.
    """
    try:
        result = benchmark_allreduce(
            rank, world_size, backend, data_size_mb,
            warmup_iters, benchmark_iters, port
        )
        if rank == 0:
            result_queue.put(result)
    except Exception as e:
        if rank == 0:
            result_queue.put(e)


def run_single_config(
    world_size: int,
    backend: str,
    data_size_mb: float,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    port: int = 29500,
) -> dict:
    """
    Run benchmark for a single configuration.

    Args:
        world_size: Number of processes
        backend: Backend to use ('gloo' or 'nccl')
        data_size_mb: Size of data in MB
        warmup_iters: Number of warmup iterations
        benchmark_iters: Number of benchmark iterations
        port: Master port for communication

    Returns:
        Dictionary with benchmark results
    """
    # Use a queue to get results from rank 0
    result_queue = mp.Queue()

    # Spawn processes
    mp.spawn(
        fn=run_benchmark_wrapper,
        args=(world_size, backend, data_size_mb, warmup_iters, benchmark_iters, port, result_queue),
        nprocs=world_size,
        join=True,
    )

    # Get result from queue
    result = result_queue.get()

    if isinstance(result, Exception):
        raise result

    mean_time, std_time = result

    # Calculate bandwidth (MB/s)
    # all-reduce transfers (world_size - 1) * data_size worth of data
    # Ring all-reduce: 2 * (world_size - 1) / world_size * data_size
    data_transferred = 2 * (world_size - 1) / world_size * data_size_mb
    bandwidth = data_transferred / (mean_time / 1000)  # MB/s

    return {
        'world_size': world_size,
        'backend': backend,
        'data_size_mb': data_size_mb,
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'bandwidth_mbps': bandwidth,
    }


def plot_results(df: pd.DataFrame, output_dir: Path):
    """
    Create visualization plots for benchmark results.

    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    sns.set_style("whitegrid")

    # Plot 1: Latency vs Data Size (separate subplots for each backend)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, backend in enumerate(['gloo', 'nccl']):
        ax = axes[idx]
        backend_df = df[df['backend'] == backend]

        for world_size in sorted(backend_df['world_size'].unique()):
            ws_df = backend_df[backend_df['world_size'] == world_size]
            ax.plot(ws_df['data_size_mb'], ws_df['mean_time_ms'],
                   marker='o', label=f'{world_size} processes')

        ax.set_xlabel('Data Size (MB)')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'All-Reduce Latency - {backend.upper()}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'allreduce_latency_by_backend.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_dir / 'allreduce_latency_by_backend.png'}")
    plt.close()

    # Plot 2: Bandwidth vs Number of Processes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, backend in enumerate(['gloo', 'nccl']):
        ax = axes[idx]
        backend_df = df[df['backend'] == backend]

        for data_size in sorted(backend_df['data_size_mb'].unique()):
            size_df = backend_df[backend_df['data_size_mb'] == data_size]
            ax.plot(size_df['world_size'], size_df['bandwidth_mbps'],
                   marker='s', label=f'{int(data_size)} MB')

        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Bandwidth (MB/s)')
        ax.set_title(f'All-Reduce Bandwidth - {backend.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'allreduce_bandwidth_by_processes.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_dir / 'allreduce_bandwidth_by_processes.png'}")
    plt.close()

    # Plot 3: Gloo vs NCCL Comparison (for each data size)
    unique_sizes = sorted(df['data_size_mb'].unique())
    n_sizes = len(unique_sizes)

    fig, axes = plt.subplots(1, n_sizes, figsize=(5 * n_sizes, 5))
    if n_sizes == 1:
        axes = [axes]

    for idx, data_size in enumerate(unique_sizes):
        ax = axes[idx]
        size_df = df[df['data_size_mb'] == data_size]

        # Group by world_size and backend
        for world_size in sorted(size_df['world_size'].unique()):
            ws_df = size_df[size_df['world_size'] == world_size]
            gloo_time = ws_df[ws_df['backend'] == 'gloo']['mean_time_ms'].values
            nccl_time = ws_df[ws_df['backend'] == 'nccl']['mean_time_ms'].values

            x = [f'{world_size}P Gloo', f'{world_size}P NCCL']
            y = [gloo_time[0] if len(gloo_time) > 0 else 0,
                 nccl_time[0] if len(nccl_time) > 0 else 0]

            ax.bar(x, y, alpha=0.7)

        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'{int(data_size)} MB Data Size')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'allreduce_gloo_vs_nccl.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_dir / 'allreduce_gloo_vs_nccl.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark all-reduce operations")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./allreduce_benchmark_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--benchmark_iters",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--skip_gloo",
        action="store_true",
        help="Skip Gloo benchmarks (CPU)",
    )
    parser.add_argument(
        "--skip_nccl",
        action="store_true",
        help="Skip NCCL benchmarks (GPU)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=29500,
        help="Master port for communication",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU availability for NCCL
    if not args.skip_nccl and not torch.cuda.is_available():
        print("Warning: CUDA not available. Skipping NCCL benchmarks.")
        args.skip_nccl = True

    if not args.skip_nccl:
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs")
        if n_gpus < 6:
            print(f"Warning: Only {n_gpus} GPUs available. Some configurations may be skipped.")

    # Define benchmark configurations
    data_sizes_mb = [1, 10, 100, 1000]  # 1MB, 10MB, 100MB, 1GB
    world_sizes = [2, 4, 6]
    backends = []

    if not args.skip_gloo:
        backends.append('gloo')
    if not args.skip_nccl:
        backends.append('nccl')

    if not backends:
        print("Error: Both Gloo and NCCL are skipped. Nothing to benchmark.")
        return

    print("=" * 80)
    print("All-Reduce Benchmark Configuration")
    print("=" * 80)
    print(f"Backends: {backends}")
    print(f"Data sizes (MB): {data_sizes_mb}")
    print(f"World sizes: {world_sizes}")
    print(f"Warmup iterations: {args.warmup_iters}")
    print(f"Benchmark iterations: {args.benchmark_iters}")
    print("=" * 80)
    print()

    # Run benchmarks
    results = []
    total_configs = len(backends) * len(data_sizes_mb) * len(world_sizes)
    current = 0

    for backend in backends:
        for world_size in world_sizes:
            # Skip if not enough GPUs for NCCL
            if backend == 'nccl' and world_size > torch.cuda.device_count():
                print(f"Skipping: {backend}, {world_size} processes (not enough GPUs)")
                current += len(data_sizes_mb)
                continue

            for data_size_mb in data_sizes_mb:
                current += 1
                print(f"[{current}/{total_configs}] Benchmarking: "
                      f"backend={backend}, world_size={world_size}, data_size={data_size_mb}MB")

                try:
                    result = run_single_config(
                        world_size=world_size,
                        backend=backend,
                        data_size_mb=data_size_mb,
                        warmup_iters=args.warmup_iters,
                        benchmark_iters=args.benchmark_iters,
                        port=args.port + current,  # Use different port for each run
                    )
                    results.append(result)

                    print(f"  Mean time: {result['mean_time_ms']:.3f} ± {result['std_time_ms']:.3f} ms")
                    print(f"  Bandwidth: {result['bandwidth_mbps']:.2f} MB/s")
                    print()

                except Exception as e:
                    print(f"  FAILED: {e}")
                    print()
                    continue

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        output_path = output_dir / "allreduce_benchmark_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

        # Create plots
        print("\nGenerating plots...")
        plot_results(df, output_dir)

        # Print summary table
        print("\n" + "=" * 120)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 120)

        # Format for display
        display_df = df.copy()
        display_df['mean_time_ms'] = display_df['mean_time_ms'].apply(lambda x: f"{x:.3f}")
        display_df['std_time_ms'] = display_df['std_time_ms'].apply(lambda x: f"{x:.3f}")
        display_df['bandwidth_mbps'] = display_df['bandwidth_mbps'].apply(lambda x: f"{x:.2f}")

        print(display_df.to_string(index=False))
        print("=" * 120)

        # Analysis
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        # Compare Gloo vs NCCL
        if 'gloo' in backends and 'nccl' in backends:
            print("\nGloo vs NCCL Speedup (for common configurations):")
            for world_size in world_sizes:
                for data_size in data_sizes_mb:
                    gloo_df = df[(df['backend'] == 'gloo') &
                                (df['world_size'] == world_size) &
                                (df['data_size_mb'] == data_size)]
                    nccl_df = df[(df['backend'] == 'nccl') &
                                (df['world_size'] == world_size) &
                                (df['data_size_mb'] == data_size)]

                    if not gloo_df.empty and not nccl_df.empty:
                        gloo_time = gloo_df['mean_time_ms'].values[0]
                        nccl_time = nccl_df['mean_time_ms'].values[0]
                        speedup = gloo_time / nccl_time
                        print(f"  {world_size} processes, {int(data_size)}MB: "
                              f"NCCL is {speedup:.2f}x faster than Gloo")

        # Scaling analysis
        print("\nScaling Analysis (bandwidth per process):")
        for backend in backends:
            backend_df = df[df['backend'] == backend]
            for data_size in data_sizes_mb:
                size_df = backend_df[backend_df['data_size_mb'] == data_size]
                if len(size_df) > 1:
                    print(f"\n  {backend.upper()} - {int(data_size)}MB:")
                    for _, row in size_df.iterrows():
                        bandwidth_per_process = row['bandwidth_mbps'] / row['world_size']
                        print(f"    {int(row['world_size'])} processes: "
                              f"{bandwidth_per_process:.2f} MB/s per process")
    else:
        print("\nNo successful benchmarks completed.")


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()
