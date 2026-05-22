"""
Profile peak memory usage for optimizer state sharding.

Compares memory usage between:
1. Standard optimizer (each rank has full optimizer state)
2. Sharded optimizer (optimizer state distributed across ranks)

Measures memory at three key points:
- After model initialization
- Before optimizer step
- After optimizer step
"""

import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Optional
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


def get_memory_stats(device: torch.device) -> dict:
    """Get current GPU memory statistics in bytes."""
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
    }


def format_memory(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def calculate_model_memory(model: torch.nn.Module) -> dict:
    """Calculate memory breakdown for model components."""
    param_memory = 0
    grad_memory = 0

    for param in model.parameters():
        # Parameter memory (typically fp32 = 4 bytes per element)
        param_memory += param.numel() * param.element_size()

        # Gradient memory (if exists)
        if param.grad is not None:
            grad_memory += param.grad.numel() * param.grad.element_size()

    return {
        "parameters": param_memory,
        "gradients": grad_memory,
        "total_model": param_memory + grad_memory,
    }


def calculate_optimizer_memory(optimizer, world_size: int = 1, is_sharded: bool = False) -> dict:
    """
    Calculate memory used by optimizer state.

    For AdamW, the state includes:
    - exp_avg (first moment): same size as parameters
    - exp_avg_sq (second moment): same size as parameters
    """
    state_memory = 0
    num_states = 0

    # For the underlying optimizer
    if hasattr(optimizer, 'base_optimizer'):
        opt = optimizer.base_optimizer
    elif hasattr(optimizer, 'optimizer'):
        opt = optimizer.optimizer
    else:
        opt = optimizer

    for param_group in opt.param_groups:
        for param in param_group['params']:
            if param in opt.state:
                param_state = opt.state[param]
                # Count memory for each state tensor (exp_avg, exp_avg_sq for Adam)
                for key, value in param_state.items():
                    if torch.is_tensor(value):
                        state_memory += value.numel() * value.element_size()
                        num_states += 1

    total_memory = state_memory
    if is_sharded:
        # With sharding, each rank only stores 1/world_size of the optimizer state
        # But we report the total memory across all ranks for fair comparison
        total_memory_all_ranks = state_memory * world_size
    else:
        total_memory_all_ranks = state_memory

    return {
        "optimizer_state_local": state_memory,
        "optimizer_state_total": total_memory_all_ranks,
        "num_state_tensors": num_states,
    }


def profile_worker(
    rank: int,
    world_size: int,
    model_size: str,
    use_sharded_optimizer: bool,
    vocab_size: int,
    context_length: int,
    batch_size: int,
    use_amp: bool,
    shared_data_x,
    shared_data_y,
):
    """Worker function to profile memory usage."""
    setup_process_group(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    if rank == 0:
        mode = "SHARDED" if use_sharded_optimizer else "NON-SHARDED"
        print(f"\n{'='*80}")
        print(f"MEMORY PROFILING: {mode} OPTIMIZER")
        print(f"Model: {model_size}, World Size: {world_size}, Batch Size: {batch_size}/GPU")
        print(f"{'='*80}\n")

    # ===== CHECKPOINT 1: After Model Initialization =====
    cfg = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=10000,
        **cfg,
    ).to(device)

    # Broadcast parameters to ensure all ranks start identically
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    torch.cuda.synchronize()
    checkpoint1 = get_memory_stats(device)
    model_memory = calculate_model_memory(model)

    if rank == 0:
        print("CHECKPOINT 1: After Model Initialization")
        print(f"  Allocated:     {format_memory(checkpoint1['allocated'])}")
        print(f"  Reserved:      {format_memory(checkpoint1['reserved'])}")
        print(f"  Parameters:    {format_memory(model_memory['parameters'])}")
        print(f"  Model params:  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        print()

    # ===== Create Optimizer =====
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
    dataset = TensorDataset(shared_data_x, shared_data_y)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    # Get first batch
    data_iter = iter(dataloader)
    x, y = next(data_iter)
    x = x.to(device)
    y = y.to(device)

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )

    # ===== CHECKPOINT 2: Before Optimizer Step =====
    # Do forward and backward to populate gradients
    optimizer.zero_grad()

    with ctx:
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    loss.backward()

    torch.cuda.synchronize()
    checkpoint2 = get_memory_stats(device)
    model_memory_with_grads = calculate_model_memory(model)

    if rank == 0:
        print("CHECKPOINT 2: Before Optimizer Step (after backward)")
        print(f"  Allocated:     {format_memory(checkpoint2['allocated'])}")
        print(f"  Reserved:      {format_memory(checkpoint2['reserved'])}")
        print(f"  Parameters:    {format_memory(model_memory_with_grads['parameters'])}")
        print(f"  Gradients:     {format_memory(model_memory_with_grads['gradients'])}")
        print(f"  Increase:      {format_memory(checkpoint2['allocated'] - checkpoint1['allocated'])}")
        print()

    # ===== CHECKPOINT 3: After Optimizer Step =====
    optimizer.step()

    torch.cuda.synchronize()
    checkpoint3 = get_memory_stats(device)
    optimizer_memory = calculate_optimizer_memory(optimizer, world_size, use_sharded_optimizer)

    if rank == 0:
        print("CHECKPOINT 3: After Optimizer Step")
        print(f"  Allocated:     {format_memory(checkpoint3['allocated'])}")
        print(f"  Reserved:      {format_memory(checkpoint3['reserved'])}")
        print(f"  Max Allocated: {format_memory(checkpoint3['max_allocated'])}")
        print(f"  Increase:      {format_memory(checkpoint3['allocated'] - checkpoint2['allocated'])}")
        print()

        print("MEMORY BREAKDOWN:")
        print(f"  Parameters:           {format_memory(model_memory_with_grads['parameters'])}")
        print(f"  Gradients:            {format_memory(model_memory_with_grads['gradients'])}")
        print(f"  Optimizer State (local): {format_memory(optimizer_memory['optimizer_state_local'])}")
        if use_sharded_optimizer:
            print(f"  Optimizer State (total across ranks): {format_memory(optimizer_memory['optimizer_state_total'])}")
        print(f"  Other (activations, temp): {format_memory(checkpoint3['allocated'] - model_memory_with_grads['total_model'] - optimizer_memory['optimizer_state_local'])}")
        print()

        # Calculate theoretical optimizer state size
        num_params = sum(p.numel() for p in model.parameters())
        bytes_per_param = 4  # fp32
        theoretical_optimizer_state = num_params * bytes_per_param * 2  # 2 moments for Adam

        print("THEORETICAL MEMORY:")
        print(f"  Params (fp32):        {format_memory(num_params * bytes_per_param)}")
        print(f"  Grads (fp32):         {format_memory(num_params * bytes_per_param)}")
        print(f"  Optimizer (2 moments): {format_memory(theoretical_optimizer_state)}")
        print(f"  Total (params+grads+opt): {format_memory(num_params * bytes_per_param * 4)}")

        if use_sharded_optimizer:
            print(f"  Per-rank with sharding:   {format_memory(num_params * bytes_per_param * (2 + 1/world_size * 2))}")
            print(f"    (params + grads + 1/{world_size} optimizer state)")
        print()

        print("MEMORY SAVINGS WITH SHARDING:")
        savings_per_rank = theoretical_optimizer_state * (1 - 1/world_size)
        print(f"  Optimizer state reduction per rank: {format_memory(savings_per_rank)}")
        print(f"  Percentage saved: {(1 - 1/world_size) * 100:.1f}%")
        print(f"  (With {world_size} ranks, each stores 1/{world_size} = {100/world_size:.1f}% of optimizer state)")
        print()

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Profile memory usage with and without optimizer state sharding"
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
        "--use_amp",
        action="store_true",
        help="Use bfloat16 autocast",
    )
    args = parser.parse_args()

    # Generate shared synthetic dataset (just one batch for memory profiling)
    print(f"\nGenerating synthetic dataset: {args.batch_size * args.world_size} samples...")
    shared_data_x = torch.randint(
        0, args.vocab_size, (args.batch_size * args.world_size, args.context_length)
    ).share_memory_()
    shared_data_y = torch.randint(
        0, args.vocab_size, (args.batch_size * args.world_size, args.context_length)
    ).share_memory_()
    print(f"Dataset generated. Each rank will process {args.batch_size} samples.\n")

    # Profile non-sharded optimizer
    print("\n" + "="*80)
    print("RUNNING PROFILE: NON-SHARDED OPTIMIZER")
    print("="*80)
    mp.spawn(
        profile_worker,
        args=(
            args.world_size,
            args.model_size,
            False,  # use_sharded_optimizer=False
            args.vocab_size,
            args.context_length,
            args.batch_size,
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
        profile_worker,
        args=(
            args.world_size,
            args.model_size,
            True,  # use_sharded_optimizer=True
            args.vocab_size,
            args.context_length,
            args.batch_size,
            args.use_amp,
            shared_data_x,
            shared_data_y,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
