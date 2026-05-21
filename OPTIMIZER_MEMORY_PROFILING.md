# Optimizer State Sharding Memory Profiling

This document explains how to profile memory usage with and without optimizer state sharding, and provides analysis of the results.

## Overview

The profiling script measures peak GPU memory usage at three critical checkpoints:
1. **After model initialization** - Just parameters loaded
2. **Before optimizer step** - After backward pass (params + gradients + activations)
3. **After optimizer step** - After optimizer state is created (params + gradients + optimizer state)

## Running the Profiler

### Basic Usage (XL Model, 2 GPUs)

```bash
python cs336_systems/profile_optimizer_memory.py \
    --model_size xl \
    --world_size 2 \
    --use_amp
```

### With Different Configurations

```bash
# Small model for quick testing
python cs336_systems/profile_optimizer_memory.py --model_size small --world_size 2

# Larger batch size
python cs336_systems/profile_optimizer_memory.py --model_size xl --world_size 2 --batch_size 8 --use_amp

# More GPUs
python cs336_systems/profile_optimizer_memory.py --model_size xl --world_size 4 --use_amp
```

### Arguments

- `--model_size`: Model size (`small`, `medium`, `large`, `xl`) - default: `xl`
- `--world_size`: Number of GPUs - default: `2`
- `--vocab_size`: Vocabulary size - default: `50257`
- `--context_length`: Sequence length - default: `128`
- `--batch_size`: Batch size per GPU - default: `4`
- `--use_amp`: Use bfloat16 mixed precision (recommended for large models)

## Expected Results for XL Model (2 GPUs)

### Configuration
- **Model**: XL (d_model=2560, num_layers=32, d_ff=10240)
- **Parameters**: ~2.5B
- **GPUs**: 2x V100 (32GB each)
- **Precision**: fp32 (without --use_amp) or bf16 (with --use_amp)

### Non-Sharded Optimizer

#### Memory Breakdown (fp32):
```
CHECKPOINT 1: After Model Initialization
  Allocated:     ~10.0 GB
  Parameters:    10.0 GB (2.5B params × 4 bytes)

CHECKPOINT 2: Before Optimizer Step
  Allocated:     ~20.0 GB
  Parameters:    10.0 GB
  Gradients:     10.0 GB (same size as params)
  Activations:   Variable (depends on batch size)

CHECKPOINT 3: After Optimizer Step
  Allocated:     ~40.0 GB
  Parameters:    10.0 GB
  Gradients:     10.0 GB
  Optimizer State: 20.0 GB (2 moments × 10.0 GB each)
    - exp_avg (first moment):  10.0 GB
    - exp_avg_sq (second moment): 10.0 GB
```

**Total per GPU**: ~40 GB (params + grads + optimizer state)

### Sharded Optimizer

#### Memory Breakdown (fp32):
```
CHECKPOINT 1: After Model Initialization
  Allocated:     ~10.0 GB
  Parameters:    10.0 GB

CHECKPOINT 2: Before Optimizer Step
  Allocated:     ~20.0 GB
  Parameters:    10.0 GB
  Gradients:     10.0 GB

CHECKPOINT 3: After Optimizer Step
  Allocated:     ~30.0 GB
  Parameters:    10.0 GB
  Gradients:     10.0 GB
  Optimizer State (local): 10.0 GB (1/2 of total = 20.0 GB / 2 ranks)
    - Rank 0 stores optimizer state for params 0, 2, 4, 6...
    - Rank 1 stores optimizer state for params 1, 3, 5, 7...
```

**Total per GPU**: ~30 GB (params + grads + 1/2 optimizer state)

### Memory Savings

**Per GPU savings**: 10 GB (25% reduction)
- Non-sharded: 40 GB
- Sharded: 30 GB
- **Savings: 10 GB = 40 GB × (1 - 1/2)**

**With N GPUs**: Saves `optimizer_state × (1 - 1/N)` per GPU

| GPUs | Optimizer State per GPU | Memory Saved per GPU | Percentage Saved |
|------|-------------------------|---------------------|------------------|
| 1    | 20.0 GB (100%)          | 0 GB                | 0%               |
| 2    | 10.0 GB (50%)           | 10.0 GB             | 25%              |
| 4    | 5.0 GB (25%)            | 15.0 GB             | 37.5%            |
| 8    | 2.5 GB (12.5%)          | 17.5 GB             | 43.75%           |

## Memory Breakdown Formulas

### Total Memory Per GPU

For AdamW optimizer with fp32:

**Non-Sharded:**
```
Memory = Params + Grads + Optimizer_State
       = P + P + 2P
       = 4P
where P = number of parameters × 4 bytes
```

**Sharded (N GPUs):**
```
Memory = Params + Grads + (Optimizer_State / N)
       = P + P + 2P/N
       = P(2 + 2/N)
```

**Savings per GPU:**
```
Savings = 2P × (1 - 1/N)
        = 2P × (N-1)/N
```

### For XL Model (2.5B params)

With fp32 (4 bytes per param):
- P = 2.5B × 4 = 10 GB

**Non-Sharded (N=1):**
- Memory = 4P = 40 GB

**Sharded (N=2):**
- Memory = P(2 + 2/2) = 3P = 30 GB
- Savings = 10 GB (25%)

**Sharded (N=4):**
- Memory = P(2 + 2/4) = 2.5P = 25 GB
- Savings = 15 GB (37.5%)

## Understanding the Results

### Why Optimizer State is 2× Parameters

AdamW (and Adam) maintain **two moment estimates** per parameter:

1. **First moment (exp_avg)**: Moving average of gradients
   - Size: Same as parameters
   - Memory: P

2. **Second moment (exp_avg_sq)**: Moving average of squared gradients
   - Size: Same as parameters
   - Memory: P

**Total optimizer state**: 2P

### Why Sharding Saves Memory

Each GPU only stores optimizer state for **its assigned parameters**:

**With 2 GPUs:**
- GPU 0: Stores optimizer state for params [0, 2, 4, 6, ...]
- GPU 1: Stores optimizer state for params [1, 3, 5, 7, ...]

Each GPU stores **1/2 the optimizer state** but **all parameters** are broadcast after each step.

### Why Not Shard Parameters Too?

In basic optimizer state sharding (ZeRO-1):
- ✅ Shard: Optimizer state
- ❌ Don't shard: Parameters and gradients

**Reason**: Parameters and gradients are needed for every forward/backward pass. Sharding them would require communication during forward/backward, which is expensive.

More advanced approaches (ZeRO-2, ZeRO-3) do shard gradients and parameters, but require more sophisticated communication patterns.

## Alignment with Expectations

### Expected Behavior ✓

1. **Model initialization**: Same for both (only parameters)
2. **Before optimizer step**: Same for both (params + grads)
3. **After optimizer step**: Sharded uses less memory
   - Non-sharded: +2P for optimizer state
   - Sharded: +2P/N for optimizer state

### Memory Scaling

With N GPUs:
- **Non-sharded**: Each GPU uses 4P (constant)
- **Sharded**: Each GPU uses P(2 + 2/N) (decreases as N increases)

**Diminishing returns:**
- 1→2 GPUs: Save 50% of optimizer state (25% total)
- 2→4 GPUs: Save additional 25% of optimizer state (12.5% total)
- 4→8 GPUs: Save additional 12.5% of optimizer state (6.25% total)

## Practical Implications

### When to Use Optimizer State Sharding

✅ **Use when:**
- Training very large models (billions of parameters)
- GPU memory is the bottleneck
- Have multiple GPUs available
- Can tolerate slight communication overhead

❌ **Skip when:**
- Model fits comfortably in GPU memory
- Single GPU training
- Communication bandwidth is limited

### Real-World Example

**Training a 175B parameter model (GPT-3 scale):**

Without sharding (fp32):
- Parameters: 700 GB
- Gradients: 700 GB
- Optimizer: 1400 GB
- **Total: 2800 GB per GPU** ❌ Won't fit!

With sharding across 8 GPUs:
- Parameters: 700 GB
- Gradients: 700 GB
- Optimizer: 175 GB (1400/8)
- **Total: 1575 GB per GPU** ✅ Fits on 8× A100 (80GB each with model parallelism)

## Troubleshooting

### Out of Memory Even with Sharding

Try:
1. **Use mixed precision**: Add `--use_amp` (reduces memory by ~50%)
2. **Reduce batch size**: Use smaller `--batch_size`
3. **Use smaller model**: Try `--model_size large` instead of `xl`
4. **More GPUs**: Increase `--world_size`

### Script Hangs

- Make sure ports aren't in use: `kill -9 $(lsof -ti:29500)`
- Check all GPUs are available: `nvidia-smi`

## Summary

Optimizer state sharding provides **memory savings proportional to the number of GPUs**:
- Saves `optimizer_state × (1 - 1/N)` per GPU
- For AdamW: Saves `2P × (N-1)/N` where P = parameter memory
- Essential for training very large models
- Minimal computational overhead (only broadcast after optimizer step)

The profiling results should clearly show these memory savings and align with theoretical expectations!
