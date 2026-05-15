# FlashAttention-2 Benchmark Summary

## Implementation Overview

This benchmark compares **FlashAttention-2 (Triton implementation)** with a **standard PyTorch attention** implementation to measure performance improvements from memory-efficient attention.

## Benchmark Script

**Location**: `cs336_systems/benchmark_flash_attention.py`

### What It Does

1. **Implements baseline PyTorch attention**:
   - Standard `Q @ K^T @ V` computation
   - Explicit softmax materialization (memory-intensive)
   - Causal masking support

2. **Benchmarks both implementations** using `triton.testing.do_bench`:
   - Forward pass latency
   - Backward pass latency
   - End-to-end (forward + backward) latency

3. **Sweeps comprehensive parameter space**:
   - Sequence lengths: 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
   - Embedding dimensions: 16, 32, 64, 128
   - Data types: `float32`, `bfloat16`
   - **Total: 80 configurations**

4. **Outputs results**:
   - CSV file with detailed timings
   - Console summary with speedup metrics
   - Summary statistics (average/max speedups)

## Running the Benchmark

### Prerequisites

- **Hardware**: NVIDIA H100 GPU (or other CUDA-capable GPU)
- **Software**: CUDA toolkit, PyTorch, Triton, pandas
- **Platform**: Linux (Triton is Linux-only)

### Commands

**Full benchmark (80 configurations)**:
```bash
uv run python cs336_systems/benchmark_flash_attention.py --output_dir ./flash_benchmark_results
```

**Quick test (2 configurations)**:
```bash
uv run python cs336_systems/benchmark_flash_attention.py \
    --min_seq_len 128 --max_seq_len 128 \
    --min_d_model 64 --max_d_model 64 \
    --output_dir ./flash_benchmark_test
```

**Custom range**:
```bash
uv run python cs336_systems/benchmark_flash_attention.py \
    --min_seq_len 1024 --max_seq_len 8192 \
    --min_d_model 32 --max_d_model 128 \
    --output_dir ./flash_benchmark_custom
```

## Benchmark Configuration

### Fixed Parameters
- **Batch size**: 1
- **Causal masking**: True (always enabled)
- **Warmup iterations**: 25
- **Benchmark repetitions**: 100

### Variable Parameters
| Parameter | Values | Count |
|-----------|--------|-------|
| Sequence length | 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 | 10 |
| Embedding dimension | 16, 32, 64, 128 | 4 |
| Data type | float32, bfloat16 | 2 |
| **Total configurations** | | **80** |

## Expected Results

### Performance Metrics

The benchmark reports:

1. **Absolute latencies** (milliseconds):
   - PyTorch forward/backward/end-to-end
   - FlashAttention forward/backward/end-to-end

2. **Speedup ratios**:
   - Forward speedup = PyTorch_fwd / Flash_fwd
   - Backward speedup = PyTorch_bwd / Flash_bwd
   - End-to-end speedup = PyTorch_e2e / Flash_e2e

### Why FlashAttention is Faster

1. **Memory efficiency**:
   - PyTorch: Materializes full `(N×N)` attention matrix → O(N²) memory
   - FlashAttention: Tiled computation → O(1) memory for intermediate values

2. **Memory bandwidth**:
   - Reduces HBM (slow) reads/writes
   - Keeps intermediate values in SRAM (fast)

3. **Kernel fusion**:
   - Fuses softmax, masking, and matrix multiplication
   - Reduces kernel launch overhead

### Expected Speedup Trends

- **Larger sequence lengths** → Higher speedups (O(N²) vs O(1) memory becomes critical)
- **Smaller embedding dimensions** → Higher speedups (memory bottleneck dominates over compute)
- **bfloat16 vs float32** → Similar speedups (both benefit from memory efficiency)

## Output Format

### CSV Columns

```
seq_len, d_model, dtype, batch_size,
pytorch_fwd_ms, pytorch_bwd_ms, pytorch_e2e_ms,
flash_fwd_ms, flash_bwd_ms, flash_e2e_ms,
speedup_fwd, speedup_bwd, speedup_e2e
```

### Example Output

```
seq_len  d_model  dtype      pytorch_fwd_ms  flash_fwd_ms  speedup_fwd
128      16       float32    0.234           0.123         1.90x
128      16       bfloat16   0.198           0.105         1.89x
256      16       float32    0.456           0.198         2.30x
...
```

### Summary Statistics

```
Average Forward Speedup:  2.34x
Average Backward Speedup: 2.56x
Average E2E Speedup:      2.45x
Max Forward Speedup:      3.89x
Max Backward Speedup:     4.12x
Max E2E Speedup:          3.98x
```

## Memory Considerations

### Sequence Length Limits

Approximate GPU memory requirements (batch_size=1):

| Seq Length | d_model=16 | d_model=64 | d_model=128 | Notes |
|------------|------------|------------|-------------|-------|
| 1024       | ~10 MB     | ~40 MB     | ~80 MB      | All configs OK |
| 4096       | ~150 MB    | ~600 MB    | ~1.2 GB     | All configs OK |
| 16384      | ~2.5 GB    | ~10 GB     | ~20 GB      | May OOM on smaller GPUs |
| 65536      | ~40 GB     | ~160 GB    | ~320 GB     | **Requires H100 80GB** |

**Note**: PyTorch implementation uses more memory than FlashAttention due to materializing full attention matrix.

### OOM Handling

If you encounter OOM errors:
1. Reduce `--max_seq_len` (e.g., to 16384 or 8192)
2. Skip float32 for large configs (use only bfloat16)
3. Run subsets of configurations separately

## Validation

Before benchmarking, ensure your FlashAttention implementation passes correctness tests:

```bash
# Test PyTorch implementation
uv run pytest -s -k test_flash_forward_pass_pytorch

# Test Triton implementation
uv run pytest -s -k test_flash_forward_pass_triton

# Test backward passes
uv run pytest -s -k test_flash_backward
```

## Files Created

1. **`cs336_systems/benchmark_flash_attention.py`** - Main benchmark script
2. **`BENCHMARK_INSTRUCTIONS.md`** - Detailed usage instructions
3. **`FLASH_ATTENTION_BENCHMARK_SUMMARY.md`** - This file
4. **`test_benchmark_script.py`** - Validation tests (CPU-compatible subset)

## Common Issues

### "Triton not available"
- Triton requires Linux + CUDA GPU
- Cannot run on macOS or Windows
- Solution: Run on GPU cluster/server

### "CUDA out of memory"
- Large sequence lengths require significant memory
- Solution: Reduce `--max_seq_len` or run smaller batches

### "Incorrect results"
- FlashAttention implementation may have bugs
- Solution: Verify correctness tests pass first

### Slow benchmarking
- Full sweep (80 configs) can take 1-2 hours on H100
- Solution: Use `--max_seq_len` to limit configurations

## Next Steps

1. **Run the benchmark** on an H100 GPU
2. **Analyze results**: Look for configurations where FlashAttention provides highest speedups
3. **Generate visualizations**: Plot speedup vs sequence length, embedding dimension
4. **Compare with published results**: FlashAttention paper reports 2-4x speedups for similar configs
5. **Optimize further**: Tune tile sizes, try different compilation options

## References

- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
- Triton documentation: https://triton-lang.org/
