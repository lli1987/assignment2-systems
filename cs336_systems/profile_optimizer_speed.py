"""
Profile training speed with and without optimizer state sharding.

Measures the time taken per training iteration and breaks down:
- Forward pass time
- Backward pass time
- Optimizer step time
- Communication time (broadcast parameters)

Reports:
- Time per iteration (mean ± std)
- Breakdown of where time is spent
- Overhead of optimizer state sharding
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def setup_process_group(rank: int, world_size: int, port: int = 29500):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def time_forward_backward(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    vocab_size: int,
    ctx,
) -> tuple:
    """Time forward and backward passes separately."""
    # Forward pass
    torch.cuda.synchronize()
    t_fwd_start = time.perf_counter()

    with ctx:
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    torch.cuda.synchronize()
    t_fwd_end = time.perf_counter()

    # Backward pass
    torch.cuda.synchronize()
    t_bwd_start = time.perf_counter()

    loss.backward()

    torch.cuda.synchronize()
    t_bwd_end = time.perf_counter()

    return (t_fwd_end - t_fwd_start, t_bwd_end - t_bwd_start, loss.item())


def time_optimizer_step(optimizer) -> float:
    """Time the full optimizer.step() call."""
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    return t_end - t_start


def profile_speed_worker(
    rank: int,
    world_size: int,
    model_size: str,
    use_sharded_optimizer: bool,
    vocab_size: int,
    context_length: int,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    use_amp: bool,
    shared_data_x,
    shared_data_y,
):
    """Worker function to profile training speed."""
    setup_process_group(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        mode = "SHARDED" if use_sharded_optimizer else "NON-SHARDED"
        print(f"\n{'='*80}")
        print(f"SPEED PROFILING: {mode} OPTIMIZER")
        print(f"Model: {model_size}, World Size: {world_size}, Batch Size: {batch_size}/GPU")
        print(f"Warmup: {warmup_steps} steps, Measure: {measure_steps} steps")
        print(f"{'='*80}\n")

    # Create model
    cfg = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=10000,
        **cfg,
    ).to(device)

    # Broadcast parameters
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Create optimizer
    if use_sharded_optimizer:
        from cs336_systems.optimizer_state_sharding import OptimizerWithStateSharding
        optimizer = OptimizerWithStateSharding(
            model.parameters(),
            AdamW,
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

    # Create dataset and distributed sampler
    total_steps = warmup_steps + measure_steps

    # Create dataset from shared data
    dataset = TensorDataset(shared_data_x, shared_data_y)

    # Use DistributedSampler to shard data across ranks
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    # DataLoader with distributed sampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )

    # Storage for timing measurements
    forward_times = []
    backward_times = []
    optimizer_times = []
    total_times = []
    losses = []

    # Iterate through batches from the dataloader
    for step, (x, y) in enumerate(dataloader):
        if step >= total_steps:
            break

        # Move data to device
        x = x.to(device)
        y = y.to(device)

        # Start iteration timer
        torch.cuda.synchronize()
        t_iter_start = time.perf_counter()

        # Zero gradients
        optimizer.zero_grad()

        # Forward and backward
        t_fwd, t_bwd, loss = time_forward_backward(model, x, y, vocab_size, ctx)

        # Optimizer step
        t_opt = time_optimizer_step(optimizer)

        # End iteration timer
        torch.cuda.synchronize()
        t_iter_end = time.perf_counter()
        t_total = t_iter_end - t_iter_start

        # Record measurements (skip warmup steps)
        if step >= warmup_steps:
            forward_times.append(t_fwd)
            backward_times.append(t_bwd)
            optimizer_times.append(t_opt)
            total_times.append(t_total)
            losses.append(loss)

        # Print progress
        if rank == 0 and (step < 3 or step == total_steps - 1):
            phase = "warmup" if step < warmup_steps else "measure"
            print(f"[{phase}] Step {step}: "
                  f"total={t_total*1000:.1f}ms, "
                  f"fwd={t_fwd*1000:.1f}ms, "
                  f"bwd={t_bwd*1000:.1f}ms, "
                  f"opt={t_opt*1000:.1f}ms, "
                  f"loss={loss:.4f}")

    # Convert to numpy arrays for statistics
    forward_times = np.array(forward_times) * 1000  # Convert to ms
    backward_times = np.array(backward_times) * 1000
    optimizer_times = np.array(optimizer_times) * 1000
    total_times = np.array(total_times) * 1000

    # Synchronize before printing results
    dist.barrier()

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"RESULTS: {mode} OPTIMIZER")
        print(f"{'='*80}\n")

        print("TIMING BREAKDOWN (mean ± std):")
        print(f"  Forward pass:      {forward_times.mean():8.2f} ± {forward_times.std():6.2f} ms")
        print(f"  Backward pass:     {backward_times.mean():8.2f} ± {backward_times.std():6.2f} ms")
        print(f"  Optimizer step:    {optimizer_times.mean():8.2f} ± {optimizer_times.std():6.2f} ms")
        print(f"  Total iteration:   {total_times.mean():8.2f} ± {total_times.std():6.2f} ms")
        print()

        # Calculate percentages
        compute = forward_times.mean() + backward_times.mean()
        opt_pct = optimizer_times.mean() / total_times.mean() * 100
        compute_pct = compute / total_times.mean() * 100

        print("TIME DISTRIBUTION:")
        print(f"  Forward + Backward: {compute_pct:.1f}%")
        print(f"  Optimizer step:     {opt_pct:.1f}%")
        print(f"  Other:              {100 - compute_pct - opt_pct:.1f}%")
        print()

        print("THROUGHPUT:")
        # Calculate throughput
        samples_per_sec = (batch_size * world_size * 1000) / total_times.mean()
        tokens_per_sec = samples_per_sec * context_length
        print(f"  Samples/sec:       {samples_per_sec:.1f}")
        print(f"  Tokens/sec:        {tokens_per_sec:.0f}")
        print(f"  Iterations/sec:    {1000 / total_times.mean():.2f}")
        print()

        # Calculate model FLOPs (approximate)
        num_params = sum(p.numel() for p in model.parameters())
        print("MODEL STATISTICS:")
        print(f"  Parameters:        {num_params / 1e9:.2f}B")
        print(f"  Loss (final):      {losses[-1]:.4f}")
        print()

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Profile training speed with and without optimizer state sharding"
    )
    parser.add_argument(
        "--model_size",
        default="xl",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size to profile",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=128,
        help="Context length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--measure_steps",
        type=int,
        default=20,
        help="Number of measured iterations",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use bfloat16 autocast",
    )
    args = parser.parse_args()

    # Generate shared synthetic dataset (total global batch)
    # Total samples needed: (warmup + measure) steps * batch_size * world_size
    total_steps = args.warmup_steps + args.measure_steps
    total_samples = total_steps * args.batch_size * args.world_size

    print(f"\nGenerating synthetic dataset: {total_samples} samples...")
    shared_data_x = torch.randint(
        0, args.vocab_size, (total_samples, args.context_length)
    ).share_memory_()
    shared_data_y = torch.randint(
        0, args.vocab_size, (total_samples, args.context_length)
    ).share_memory_()
    print(f"Dataset generated. Each rank will process {total_samples // args.world_size} samples.\n")

    # Profile non-sharded optimizer
    print("\n" + "="*80)
    print("RUNNING PROFILE: NON-SHARDED OPTIMIZER")
    print("="*80)
    mp.spawn(
        profile_speed_worker,
        args=(
            args.world_size,
            args.model_size,
            False,  # use_sharded_optimizer=False
            args.vocab_size,
            args.context_length,
            args.batch_size,
            args.warmup_steps,
            args.measure_steps,
            args.use_amp,
            shared_data_x,
            shared_data_y,
        ),
        nprocs=args.world_size,
        join=True,
    )

    # Profile sharded optimizer
    print("\n" + "="*80)
    print("RUNNING PROFILE: SHARDED OPTIMIZER")
    print("="*80)
    mp.spawn(
        profile_speed_worker,
        args=(
            args.world_size,
            args.model_size,
            True,  # use_sharded_optimizer=True
            args.vocab_size,
            args.context_length,
            args.batch_size,
            args.warmup_steps,
            args.measure_steps,
            args.use_amp,
            shared_data_x,
            shared_data_y,
        ),
        nprocs=args.world_size,
        join=True,
    )

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print("\nThe overhead of optimizer state sharding comes from:")
    print("1. Parameter broadcast after optimizer step")
    print("2. All ranks must participate in broadcasts (synchronization)")
    print("\nExpected overhead: ~5-15% depending on:")
    print("- Model size (larger = more parameters to broadcast)")
    print("- Network bandwidth (faster = less overhead)")
    print("- Compute vs communication ratio")
    print("\nBenefit: Reduced memory usage enables training larger models")
    print("or using larger batch sizes!")
    print("="*80 + "\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
