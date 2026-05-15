# FlashAttention-2 Benchmark Instructions

## Overview

This benchmark compares the performance of FlashAttention-2 (Triton implementation) with a standard PyTorch attention implementation.

## Requirements

- NVIDIA H100 GPU (or other CUDA-capable GPU)
- CUDA toolkit
- Python packages: torch, triton, pandas

## Running the Benchmark

### Basic Usage

Run the full benchmark sweep:

```bash
uv run python cs336_systems/benchmark_flash_attention.py
```

### Custom Parameters

```bash
uv run python cs336_systems/benchmark_flash_attention.py \
    --output_dir ./my_results \
    --min_seq_len 128 \
    --max_seq_len 65536 \
    --min_d_model 16 \
    --max_d_model 128
```

### Arguments

- `--output_dir`: Directory to save results (default: `./benchmark_results`)
- `--device`: Device to run on (default: `cuda`)
- `--min_seq_len`: Minimum sequence length, power of 2 (default: 128)
- `--max_seq_len`: Maximum sequence length, power of 2 (default: 65536)
- `--min_d_model`: Minimum embedding dimension, power of 2 (default: 16)
- `--max_d_model`: Maximum embedding dimension, power of 2 (default: 128)

## Benchmark Configuration

**Fixed Parameters:**
- Batch size: 1
- Causal masking: True
- Warmup iterations: 25
- Benchmark repetitions: 100

**Variable Parameters:**
- Sequence lengths: Powers of 2 from 128 to 65536 (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
- Embedding dimensions: Powers of 2 from 16 to 128 (16, 32, 64, 128)
- Data types: torch.float32, torch.bfloat16

**Total configurations:** 10 seq_lens × 4 d_models × 2 dtypes = **80 configurations**

## Output

The benchmark produces:

1. **CSV file** (`flash_attention_benchmark.csv`) with columns:
   - `seq_len`: Sequence length
   - `d_model`: Embedding dimension
   - `dtype`: Data type (float32 or bfloat16)
   - `batch_size`: Batch size (always 1)
   - `pytorch_fwd_ms`: PyTorch forward pass time (ms)
   - `pytorch_bwd_ms`: PyTorch backward pass time (ms)
   - `pytorch_e2e_ms`: PyTorch end-to-end time (ms)
   - `flash_fwd_ms`: FlashAttention forward pass time (ms)
   - `flash_bwd_ms`: FlashAttention backward pass time (ms)
   - `flash_e2e_ms`: FlashAttention end-to-end time (ms)
   - `speedup_fwd`: Forward speedup (PyTorch/Flash)
   - `speedup_bwd`: Backward speedup
   - `speedup_e2e`: End-to-end speedup

2. **Console output** showing:
   - Progress for each configuration
   - Per-configuration timings and speedups
   - Final summary table
   - Summary statistics (average and max speedups)

## Example Output

```
Benchmarking FlashAttention-2 (Triton) vs PyTorch
================================================================================
Sequence lengths: [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
Embedding dimensions: [16, 32, 64, 128]
Data types: ['float32', 'bfloat16']
Batch size: 1 (fixed)
Causal masking: True (fixed)
Device: cuda
================================================================================

[1/80] Benchmarking: seq_len=128, d_model=16, dtype=float32
  PyTorch - Fwd: 0.234ms, Bwd: 0.456ms, E2E: 0.690ms
  Flash   - Fwd: 0.123ms, Bwd: 0.234ms, E2E: 0.357ms
  Speedup - Fwd: 1.90x, Bwd: 1.95x, E2E: 1.93x

...

Summary Statistics:
Average Forward Speedup:  2.34x
Average Backward Speedup: 2.56x
Average E2E Speedup:      2.45x
Max Forward Speedup:      3.89x
Max Backward Speedup:     4.12x
Max E2E Speedup:          3.98x
```

## Notes

- The benchmark uses `triton.testing.do_bench` which handles CUDA synchronization and timing automatically
- Each configuration is benchmarked with 25 warmup iterations and 100 measurement iterations
- For very large sequence lengths (32768, 65536), you may need significant GPU memory (H100 recommended)
- The benchmark clears gradients between iterations to avoid memory accumulation
- If a configuration fails (e.g., OOM), it's skipped and logged

## Troubleshooting

**Out of Memory (OOM):**
- Reduce `--max_seq_len` to a smaller value (e.g., 16384 or 8192)
- Ensure no other processes are using GPU memory

**Triton Compilation Errors:**
- Check that Triton is properly installed: `pip show triton`
- Verify CUDA compatibility with your GPU

**Incorrect Results:**
- Verify FlashAttention implementation passes correctness tests first
- Check that causal masking is implemented in both forward and backward passes

## Advanced Usage

### Run Subset of Configurations

Test only small configurations:
```bash
uv run python cs336_systems/benchmark_flash_attention.py \
    --min_seq_len 128 \
    --max_seq_len 1024 \
    --min_d_model 16 \
    --max_d_model 64
```

### Quick Test Run

For a quick sanity check (2 configs):
```bash
uv run python cs336_systems/benchmark_flash_attention.py \
    --min_seq_len 128 \
    --max_seq_len 128 \
    --min_d_model 64 \
    --max_d_model 64
```

This will test only `seq_len=128, d_model=64` for both dtypes.
