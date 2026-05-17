"""Benchmark naive DDP training for the language model.

Measures total time per training step and the proportion spent on
gradient all-reduce communication.

Setup: 1 node x 2 GPUs (NCCL backend)

Usage:
    uv run python cs336_systems/benchmark_ddp.py
    uv run python cs336_systems/benchmark_ddp.py --model_size xl --world_size 2
"""

import os
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


def setup(rank, world_size, port=29500):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def benchmark_worker(
    rank,
    world_size,
    model_size,
    vocab_size,
    context_length,
    batch_size,
    warmup_steps,
    measure_steps,
    use_amp,
):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    cfg = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=10000,
        **cfg,
    ).to(device)

    # Ensure all ranks start with identical parameters
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    step_times = []
    allreduce_times = []

    total_steps = warmup_steps + measure_steps
    for step in range(total_steps):
        # Synthetic token data — no dataset dependency needed for benchmarking
        x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        optimizer.zero_grad()

        with ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        loss.backward()

        # ---- Gradient communication ----
        # Synchronize first so backward is fully complete before timing all-reduce
        torch.cuda.synchronize()
        t_ar_start = time.perf_counter()

        param_tensors = []
        for param in model.parameters():
            if param.grad is not None:
                param_tensors.append(param.grad)
        flat_param_tensor = torch._utils_._flatten_dense_tensors(param_tensors)
        dist.all_reduce(flat_param_tensor, dist.ReduceOp.AVG)
        unflattened = torch._utils_.unflatten_dense_tensors(flat_param_tensor, param_tensors)
        for orig, unflat in zip(param_tensors, unflattened):
            orig.copy_(unflat)

        # Synchronize so all NCCL ops are done before recording end time
        torch.cuda.synchronize()
        t_ar_end = time.perf_counter()
        # --------------------------------

        optimizer.step()

        torch.cuda.synchronize()
        t_step_end = time.perf_counter()

        if step >= warmup_steps:
            step_times.append(t_step_end - t_step_start)
            allreduce_times.append(t_ar_end - t_ar_start)

        if rank == 0:
            phase = "warmup" if step < warmup_steps else "measure"
            print(f"[{phase}] step {step}: loss={loss.item():.4f}")

    dist.destroy_process_group()

    if rank == 0:
        step_ms = np.array(step_times) * 1000
        ar_ms = np.array(allreduce_times) * 1000
        compute_ms = step_ms - ar_ms

        num_params = sum(p.numel() for p in model.parameters())
        grad_bytes = num_params * (2 if use_amp else 4)  # bf16 or fp32

        print()
        print("=" * 70)
        print(f"NAIVE DDP BENCHMARK  —  {model_size.upper()} model, {world_size} GPU(s)")
        print("=" * 70)
        print(
            f"Model:        d_model={cfg['d_model']}, num_layers={cfg['num_layers']}, "
            f"num_heads={cfg['num_heads']}, d_ff={cfg['d_ff']}"
        )
        print(f"Parameters:   {num_params / 1e9:.2f}B")
        print(f"Gradient data per all-reduce: {grad_bytes / 1e9:.2f} GB ({'bf16' if use_amp else 'fp32'})")
        print(f"Batch/GPU:    {batch_size}, context_length: {context_length}")
        print(f"Steps:        {measure_steps} measured (after {warmup_steps} warmup)")
        print("-" * 70)
        print(f"Total step time  : {step_ms.mean():8.1f} ± {step_ms.std():5.1f} ms")
        print(
            f"  Compute        : {compute_ms.mean():8.1f} ± {compute_ms.std():5.1f} ms"
            f"  ({compute_ms.mean() / step_ms.mean() * 100:.1f}%)"
        )
        print(
            f"  AllReduce comm : {ar_ms.mean():8.1f} ± {ar_ms.std():5.1f} ms"
            f"  ({ar_ms.mean() / step_ms.mean() * 100:.1f}%)"
        )
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark naive DDP training")
    parser.add_argument("--model_size", default="xl", choices=list(MODEL_CONFIGS))
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=20)
    parser.add_argument("--use_amp", action="store_true", help="Use bfloat16 autocast (recommended for GPU)")
    args = parser.parse_args()

    mp.spawn(
        benchmark_worker,
        args=(
            args.world_size,
            args.model_size,
            args.vocab_size,
            args.context_length,
            args.batch_size,
            args.warmup_steps,
            args.measure_steps,
            args.use_amp,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
