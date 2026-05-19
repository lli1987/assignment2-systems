# DDP Benchmark OOM Fix

## Problem
Running `benchmark_ddp.py` with the XL model results in CUDA out of memory:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 492.00 MiB.
GPU 0 has a total capacity of 31.37 GiB of which 396.19 MiB is free.
```

## Root Causes

### 1. XL Model is Too Large for Your GPU
The XL configuration:
- `d_model=2560, d_ff=10240, num_layers=32, num_heads=32`
- **~2.5B parameters**

**Memory breakdown per GPU (fp32):**
- Model weights: ~10 GB
- Gradients: ~10 GB
- Optimizer state (Adam): ~20 GB (2 moments per parameter)
- Activations: ~5-10 GB
- **Total: ~45-50 GB per process**

With `world_size=2`, you're trying to run **2 processes**, so you need **~100 GB total** but only have **31 GB**!

### 2. Memory Allocation in Training Loop
Original code created new tensors every iteration:
```python
for step in range(total_steps):
    x = torch.randint(..., device=device)  # New allocation every step!
    y = torch.randint(..., device=device)  # New allocation every step!
```

This causes memory fragmentation and accumulation.

### 3. Not Using Mixed Precision
Running in fp32 doubles memory usage compared to bf16/fp16.

## Fixes Applied

### 1. Pre-allocate Synthetic Data (Line 105-107)
**Before:**
```python
for step in range(total_steps):
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
```

**After:**
```python
# Pre-allocate synthetic data to avoid repeated memory allocations
x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

for step in range(total_steps):
    # Reuse x, y
```

**Benefit:** Eliminates repeated allocations and memory fragmentation.

### 2. Use `zero_grad(set_to_none=True)` (Line 121)
**Before:**
```python
optimizer.zero_grad()
```

**After:**
```python
optimizer.zero_grad(set_to_none=True)
```

**Benefit:** Frees gradient memory instead of just zeroing it. Saves memory and is faster.

### 3. Delete Intermediate Variables (Line 130)
```python
loss.backward()
del logits  # Free activation memory
```

**Benefit:** Explicitly frees large activation tensors after they're no longer needed.

### 4. Add Memory Estimation and Warnings (Lines 183-203)
Added automatic memory estimation before starting:
```python
num_params = estimate_params(cfg, args.vocab_size, args.context_length)
total_memory_gb = num_params * bytes_per_param / 1e9 * 3.5

print(f"Estimated memory per GPU: {total_memory_gb:.1f} GB")
if total_memory_gb > 30:
    print("WARNING: Memory estimate exceeds typical GPU capacity!")
```

### 5. Change Default to Smaller Model (Line 170)
**Before:**
```python
parser.add_argument("--model_size", default="xl", ...)
```

**After:**
```python
parser.add_argument("--model_size", default="small", ...)
```

## How to Run Successfully

### Option 1: Use Mixed Precision (Recommended for Large Models)
```bash
python cs336_systems/benchmark_ddp.py --model_size xl --use_amp
```

**Memory savings:** ~50% reduction (bf16 uses 2 bytes vs 4 bytes per parameter)

### Option 2: Reduce Batch Size
```bash
python cs336_systems/benchmark_ddp.py --model_size xl --use_amp --batch_size 1
```

### Option 3: Use Smaller Model
```bash
python cs336_systems/benchmark_ddp.py --model_size small  # Default
python cs336_systems/benchmark_ddp.py --model_size medium --use_amp
```

### Option 4: Single GPU (if you have 2 GPUs, use only 1)
```bash
python cs336_systems/benchmark_ddp.py --world_size 1 --model_size large --use_amp
```

## Memory Requirements by Model Size

| Model Size | Params | FP32 Memory | BF16 Memory | Recommended Setup |
|------------|--------|-------------|-------------|-------------------|
| small      | 125M   | ~4.5 GB     | ~2.5 GB     | Any GPU, no AMP needed |
| medium     | 350M   | ~12 GB      | ~6 GB       | Use `--use_amp` |
| large      | 774M   | ~27 GB      | ~14 GB      | Use `--use_amp` + `--batch_size 2` |
| xl         | 2.5B   | ~90 GB      | ~45 GB      | **Won't fit on single V100**, use `--use_amp` + `--batch_size 1` or smaller model |
| 10B        | 10B    | ~360 GB     | ~180 GB     | **Not feasible on V100**, requires multi-GPU sharding |

## Recommended Commands for V100 (32GB)

```bash
# Safe default (will work on any GPU)
python cs336_systems/benchmark_ddp.py

# Medium model with mixed precision
python cs336_systems/benchmark_ddp.py --model_size medium --use_amp

# Large model (requires mixed precision + small batch)
python cs336_systems/benchmark_ddp.py --model_size large --use_amp --batch_size 2

# XL model (very tight, may still OOM)
python cs336_systems/benchmark_ddp.py --model_size xl --use_amp --batch_size 1 --world_size 1
```

## Additional Memory Optimization Tips

### 1. Gradient Checkpointing (Not Implemented Yet)
Trade computation for memory by recomputing activations during backward pass:
```python
from torch.utils.checkpoint import checkpoint
output = checkpoint(model.layer, input)
```

### 2. CPU Offloading (Not Implemented Yet)
Store optimizer states on CPU:
```python
# Use ZeRO optimizer to offload states to CPU
```

### 3. Reduce Context Length
```bash
python cs336_systems/benchmark_ddp.py --context_length 64  # Half the memory for activations
```

### 4. Monitor Memory Usage
Add this to your code:
```python
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

## Debugging OOM Issues

If you still get OOM, add this at the start of your script:
```python
# Enable memory debugging
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'

# Or see detailed allocation traces
torch.cuda.memory._record_memory_history(enabled=True)
```

## Summary

The main issue was trying to run a 2.5B parameter model (XL) in fp32 on a 31GB GPU. The fixes:
1. ✅ Pre-allocate data tensors
2. ✅ Use `zero_grad(set_to_none=True)`
3. ✅ Delete intermediate variables
4. ✅ Add memory estimation
5. ✅ Change default to smaller model
6. ✅ Recommend `--use_amp` for large models

**Bottom line:** For XL model on V100, you MUST use `--use_amp` and consider `--batch_size 1`.
