# All-Reduce Benchmark Instructions

## Overview

This benchmark measures the performance of all-reduce operations across different backends, data sizes, and numbers of processes in a single-node multi-GPU setup.

## Requirements

- **Hardware**: Up to 6 GPUs (NVIDIA recommended for NCCL)
- **Software**: PyTorch with distributed support
- **Python packages**: torch, pandas, matplotlib, seaborn

## Benchmark Configuration

### Fixed Parameters
- Operation: `all_reduce` with `SUM`
- Warmup iterations: 10 (default)
- Benchmark iterations: 100 (default)
- Data type: float32

### Variable Parameters
| Parameter | Values | Count |
|-----------|--------|-------|
| Backend | Gloo (CPU), NCCL (GPU) | 2 |
| Data size | 1MB, 10MB, 100MB, 1GB | 4 |
| Number of processes | 2, 4, 6 | 3 |
| **Total configurations** | | **24** |

## Running the Benchmark

### Full Benchmark (All Configurations)

```bash
# Run both Gloo and NCCL benchmarks
uv run python cs336_systems/benchmark_allreduce.py --output_dir ./allreduce_results

# Install dependencies first if needed
uv sync
```

### GPU-Only (NCCL)

```bash
# Skip Gloo (CPU) benchmarks
uv run python cs336_systems/benchmark_allreduce.py \
    --skip_gloo \
    --output_dir ./allreduce_nccl_only
```

### CPU-Only (Gloo)

```bash
# Skip NCCL (GPU) benchmarks
uv run python cs336_systems/benchmark_allreduce.py \
    --skip_nccl \
    --output_dir ./allreduce_gloo_only
```

### Custom Iterations

```bash
# Use more iterations for more accurate measurements
uv run python cs336_systems/benchmark_allreduce.py \
    --warmup_iters 20 \
    --benchmark_iters 200 \
    --output_dir ./allreduce_high_precision
```

## Command Line Arguments

```
--output_dir        Directory to save results (default: ./allreduce_benchmark_results)
--warmup_iters      Number of warmup iterations (default: 10)
--benchmark_iters   Number of benchmark iterations (default: 100)
--skip_gloo         Skip Gloo (CPU) benchmarks
--skip_nccl         Skip NCCL (GPU) benchmarks
--port              Master port for communication (default: 29500)
```

## Output

The benchmark produces:

### 1. CSV File (`allreduce_benchmark_results.csv`)

Columns:
- `world_size`: Number of processes
- `backend`: Backend used (gloo or nccl)
- `data_size_mb`: Data size in megabytes
- `mean_time_ms`: Mean latency in milliseconds
- `std_time_ms`: Standard deviation of latency
- `bandwidth_mbps`: Effective bandwidth in MB/s

### 2. Plots

Three visualization files are generated:

**a) `allreduce_latency_by_backend.png`**
- Shows latency vs data size
- Separate subplot for Gloo and NCCL
- Different lines for different numbers of processes
- Log-log scale

**b) `allreduce_bandwidth_by_processes.png`**
- Shows bandwidth vs number of processes
- Separate subplot for Gloo and NCCL
- Different lines for different data sizes

**c) `allreduce_gloo_vs_nccl.png`**
- Direct comparison between Gloo and NCCL
- Separate subplot for each data size
- Grouped bars for different process counts

### 3. Console Output

- Progress updates for each configuration
- Mean ± std latency for each run
- Bandwidth metrics
- Summary table
- Analysis section with:
  - Gloo vs NCCL speedup comparisons
  - Scaling efficiency analysis

## Expected Results

### Latency Patterns

1. **Data Size Impact**:
   - Latency increases with data size
   - Roughly linear for small sizes, may saturate for large sizes
   - NCCL handles large data much better than Gloo

2. **Process Count Impact**:
   - More processes → slightly higher latency
   - Ring all-reduce: O(log n) communication steps
   - Overhead increases with coordination complexity

3. **Backend Comparison**:
   - **NCCL (GPU)**:
     - Lower latency for all sizes
     - Much better bandwidth utilization
     - Optimized for GPUs and NVLink
   - **Gloo (CPU)**:
     - Higher latency, especially for large data
     - Limited by CPU memory bandwidth
     - Suitable for CPU-only workloads

### Bandwidth Patterns

1. **Effective Bandwidth Formula**:
   ```
   bandwidth = data_transferred / time
   where data_transferred = 2 * (world_size - 1) / world_size * data_size
   ```

2. **Expected Bandwidth**:
   - **NCCL**: Can achieve 100-300 GB/s with NVLink on modern GPUs
   - **Gloo**: Limited to ~10-50 GB/s depending on CPU/memory

3. **Scaling**:
   - Bandwidth per process may decrease with more processes
   - NCCL scales better than Gloo
   - Large data sizes show better bandwidth utilization

## Example Output

```
All-Reduce Benchmark Configuration
================================================================================
Backends: ['gloo', 'nccl']
Data sizes (MB): [1, 10, 100, 1000]
World sizes: [2, 4, 6]
Warmup iterations: 10
Benchmark iterations: 100
================================================================================

[1/24] Benchmarking: backend=gloo, world_size=2, data_size=1MB
  Mean time: 0.524 ± 0.032 ms
  Bandwidth: 1.90 MB/s

[2/24] Benchmarking: backend=gloo, world_size=2, data_size=10MB
  Mean time: 4.156 ± 0.089 ms
  Bandwidth: 2.41 MB/s

...

ANALYSIS
================================================================================

Gloo vs NCCL Speedup (for common configurations):
  2 processes, 1MB: NCCL is 3.45x faster than Gloo
  2 processes, 10MB: NCCL is 8.23x faster than Gloo
  2 processes, 100MB: NCCL is 15.67x faster than Gloo
  2 processes, 1000MB: NCCL is 28.34x faster than Gloo
  ...

Scaling Analysis (bandwidth per process):
  NCCL - 1000MB:
    2 processes: 45678.23 MB/s per process
    4 processes: 38234.56 MB/s per process
    6 processes: 32456.78 MB/s per process
```

## Understanding the Results

### Key Metrics

1. **Latency (ms)**: Time to complete one all-reduce operation
   - Lower is better
   - Includes communication and synchronization overhead

2. **Bandwidth (MB/s)**: Effective data transfer rate
   - Higher is better
   - Calculated based on theoretical data movement
   - May be lower than peak hardware bandwidth due to overhead

3. **Speedup**: Ratio of Gloo time to NCCL time
   - Shows how much faster NCCL is
   - Typically 3-30x depending on configuration

### Factors Affecting Performance

1. **Hardware**:
   - GPU interconnect (NVLink vs PCIe)
   - CPU memory bandwidth
   - Network topology

2. **Software**:
   - Backend optimization (NCCL is highly optimized for GPUs)
   - PyTorch version
   - CUDA version

3. **Configuration**:
   - Data size (larger → better bandwidth utilization)
   - Number of processes (more → more coordination overhead)
   - Backend choice (NCCL for GPU, Gloo for CPU)

## Troubleshooting

### "Not enough GPUs" Warning

If you see:
```
Skipping: nccl, 6 processes (not enough GPUs)
```

Solution: The benchmark automatically skips configurations that require more GPUs than available.

### "Address already in use" Error

If processes fail to initialize:
```
RuntimeError: Address already in use
```

Solution: Use a different port:
```bash
uv run python cs336_systems/benchmark_allreduce.py --port 29600
```

### Slow Performance

If benchmarks take too long:
- Reduce iterations: `--warmup_iters 5 --benchmark_iters 50`
- Skip large data sizes (modify script)
- Test fewer configurations

### GPU Memory Issues

For very large data (1GB):
- Ensure GPUs have enough memory
- Each process allocates `data_size_mb` of GPU memory
- 6 processes × 1GB = 6GB total GPU memory needed

## Analysis Tips

### 1. Identify Optimal Configuration
- Look for configuration with best bandwidth
- Consider latency vs throughput tradeoff

### 2. Check Scaling Efficiency
- Compare bandwidth per process
- Good scaling: bandwidth per process stays constant
- Poor scaling: bandwidth per process decreases significantly

### 3. Backend Selection
- Use NCCL for GPU-based training
- Use Gloo for CPU-only or mixed CPU-GPU scenarios
- NCCL typically 5-30x faster for large data

### 4. Data Size Considerations
- Small data (< 10MB): Latency-bound, less difference between backends
- Large data (> 100MB): Bandwidth-bound, NCCL shines
- Very large data (1GB): May need to consider chunking

## Estimated Runtime

Full benchmark (24 configurations):
- Warmup: 10 iterations × 24 configs = 240 operations
- Benchmark: 100 iterations × 24 configs = 2400 operations
- **Total time**: ~3-5 minutes on modern hardware

With custom iterations:
- `--benchmark_iters 200`: ~6-10 minutes
- `--benchmark_iters 50`: ~2-3 minutes

## Citation

If using results from this benchmark in reports or papers, note:
- PyTorch version
- CUDA version
- GPU model (e.g., NVIDIA A100, H100)
- Number of GPUs
- Interconnect type (NVLink, PCIe)
