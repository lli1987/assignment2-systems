"""Debug script to compare FSDP vs non-parallel outputs with same data as test.

Run with: uv run python debug_fsdp_detailed.py
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from copy import deepcopy


def _setup_process_group(rank, world_size, backend="gloo"):
    """Initialize the distributed process group."""
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return torch.device("cpu")


def _apply_mixed_precision_hooks(model, compute_dtype):
    """
    Apply hooks to a non-parallel model that replicate FSDP's mixed-precision
    behavior: cast Linear/Embedding weights to compute_dtype for
    forward/backward, keep master weights and optimizer updates in fp32.
    """
    from cs336_basics.model import Embedding, Linear

    for mod in model.modules():
        if not isinstance(mod, (Linear, Embedding)):
            continue

        # Forward: cast weight to compute_dtype, restore fp32 after
        def make_fwd_pre(dt):
            def hook(m, inp):
                m._saved_fp32 = m.weight.data
                m.weight.data = m.weight.data.to(dt)

            return hook

        def make_fwd_post():
            def hook(m, inp, out):
                m.weight.data = m._saved_fp32
                del m._saved_fp32
                m.weight.grad = None

            return hook

        mod.register_forward_pre_hook(make_fwd_pre(compute_dtype))
        mod.register_forward_hook(make_fwd_post())

        # Linear backward needs the weight in compute_dtype for grad_input
        if isinstance(mod, Linear):

            def make_bwd_pre(dt):
                def hook(m, grad_output):
                    m._saved_fp32_bwd = m.weight.data
                    m.weight.data = m.weight.data.to(dt)
                    m.weight.grad = None

                return hook

            mod.register_full_backward_pre_hook(make_bwd_pre(compute_dtype))

        # After gradient is computed, restore fp32 weight and cast grad to fp32
        def make_grad_hook(m, is_linear):
            def hook(param):
                if is_linear and hasattr(m, "_saved_fp32_bwd"):
                    m.weight.data = m._saved_fp32_bwd
                    del m._saved_fp32_bwd
                if param.grad is not None:
                    param.grad = param.grad.to(torch.float32)

            return hook

        mod.weight.register_post_accumulate_grad_hook(make_grad_hook(mod, isinstance(mod, Linear)))


def debug_worker(rank: int, world_size: int, compute_dtype=None):
    """Debug FSDP behavior on a single training step."""
    torch.use_deterministic_algorithms(True)
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    # Import after process group is set up
    from cs336_basics.model import Embedding, Linear
    from tests.adapters import get_fsdp, fsdp_on_after_backward, fsdp_gather_full_params

    # Simple model - just embedding + linear
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = Embedding(100, 64)
            self.linear = Linear(64, 100)

        def forward(self, x):
            x = self.embedding(x)
            x = self.linear(x)
            return x

    # Create base model with same seed as test
    torch.manual_seed(42)
    base_model = SimpleModel().to(device)

    # Non-parallel baseline (with matching mixed-precision if needed)
    non_parallel_model = deepcopy(base_model)
    if compute_dtype is not None:
        _apply_mixed_precision_hooks(non_parallel_model, compute_dtype)

    # FSDP model
    fsdp_model = get_fsdp(deepcopy(base_model), compute_dtype=compute_dtype)

    # Optimizers
    loss_fn = nn.CrossEntropyLoss()
    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)
    non_parallel_optimizer = torch.optim.SGD(non_parallel_model.parameters(), lr=0.01)

    # Generate data - same as test
    torch.manual_seed(123)
    batch_size = 20
    seq_len = 8
    all_input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    all_labels = torch.randint(0, 100, (batch_size,), device=device)

    local_bs = batch_size // world_size

    if rank == 0:
        print(f"\n=== Initial Parameters ===")
        print(f"Non-parallel embedding.weight[0, :5]: {non_parallel_model.embedding.weight.data[0, :5]}")
        print(f"Non-parallel linear.weight[0, :5]: {non_parallel_model.linear.weight.data[0, :5]}")

    # Gather initial FSDP params
    initial_fsdp_params = fsdp_gather_full_params(fsdp_model)
    if rank == 0:
        print(f"FSDP embedding.weight[0, :5]: {initial_fsdp_params['embedding.weight'][0, :5]}")
        print(f"FSDP linear.weight[0, :5]: {initial_fsdp_params['linear.weight'][0, :5]}")

    # Check initial params match
    if rank == 0:
        emb_diff = (non_parallel_model.embedding.weight.data - initial_fsdp_params["embedding.weight"]).abs().max()
        lin_diff = (non_parallel_model.linear.weight.data - initial_fsdp_params["linear.weight"]).abs().max()
        print(f"\nInitial param diff - embedding: {emb_diff:.10f}, linear: {lin_diff:.10f}")

    # === STEP 0 ===
    if rank == 0:
        print(f"\n=== Step 0 ===")

    fsdp_optimizer.zero_grad(set_to_none=True)
    non_parallel_optimizer.zero_grad(set_to_none=True)

    # Non-parallel: forward on all data
    non_parallel_out = non_parallel_model(all_input_ids)
    non_parallel_loss = loss_fn(non_parallel_out[:, -1, :].float(), all_labels)

    if rank == 0:
        print(f"Non-parallel loss: {non_parallel_loss.item():.6f}")
        print(f"Non-parallel output[0, -1, :5]: {non_parallel_out[0, -1, :5]}")

    non_parallel_loss.backward()

    if rank == 0:
        print(f"Non-parallel embedding.grad[0, :5]: {non_parallel_model.embedding.weight.grad[0, :5]}")
        print(f"Non-parallel linear.grad[0, :5]: {non_parallel_model.linear.weight.grad[0, :5]}")

    non_parallel_optimizer.step()

    # FSDP: each rank sees a different subset
    offset = rank * local_bs
    local_input = all_input_ids[offset : offset + local_bs]
    local_labels = all_labels[offset : offset + local_bs]

    fsdp_out = fsdp_model(local_input)
    fsdp_loss = loss_fn(fsdp_out[:, -1, :].float(), local_labels)

    if rank == 0:
        print(f"Rank {rank} FSDP loss: {fsdp_loss.item():.6f}")
        print(f"Rank {rank} FSDP output[0, -1, :5]: {fsdp_out[0, -1, :5]}")

    fsdp_loss.backward()

    # Check gradients before synchronization
    if rank == 0:
        print(f"Rank {rank} FSDP embedding.grad (before sync) shape: {fsdp_model.module.embedding.weight.grad.shape}")
        print(
            f"Rank {rank} FSDP embedding.grad (before sync) [0:5]: {fsdp_model.module.embedding.weight.grad.flatten()[:5]}"
        )

    fsdp_on_after_backward(fsdp_model, fsdp_optimizer)

    # Check gradients after synchronization
    if rank == 0:
        print(f"Rank {rank} FSDP embedding.grad (after sync) shape: {fsdp_model.module.embedding.weight.grad.shape}")
        print(
            f"Rank {rank} FSDP embedding.grad (after sync) [0:5]: {fsdp_model.module.embedding.weight.grad.flatten()[:5]}"
        )

    fsdp_optimizer.step()

    # Check param and grad shapes after optimizer step
    if rank == 0:
        print(f"Rank {rank} after optimizer.step():")
        print(f"  embedding.weight.data shape: {fsdp_model.module.embedding.weight.data.shape}")
        print(f"  embedding.weight.grad shape: {fsdp_model.module.embedding.weight.grad.shape}")
        print(f"  linear.weight.data shape: {fsdp_model.module.linear.weight.data.shape}")
        print(f"  linear.weight.grad shape: {fsdp_model.module.linear.weight.grad.shape}")

    # Compare all parameters after step
    full_params = fsdp_gather_full_params(fsdp_model)

    if rank == 0:
        print(f"\n=== After Step 0 ===")
        print(f"Non-parallel embedding.weight[0, :5]: {non_parallel_model.embedding.weight.data[0, :5]}")
        print(f"FSDP embedding.weight[0, :5]: {full_params['embedding.weight'][0, :5]}")

        print(f"\nNon-parallel linear.weight[0, :5]: {non_parallel_model.linear.weight.data[0, :5]}")
        print(f"FSDP linear.weight[0, :5]: {full_params['linear.weight'][0, :5]}")

        # Check differences
        emb_diff = (non_parallel_model.embedding.weight.data - full_params["embedding.weight"]).abs()
        lin_diff = (non_parallel_model.linear.weight.data - full_params["linear.weight"]).abs()

        print(f"\n=== Differences ===")
        print(f"Embedding max diff: {emb_diff.max():.10f}")
        print(f"Embedding mean diff: {emb_diff.mean():.10f}")
        print(f"Linear max diff: {lin_diff.max():.10f}")
        print(f"Linear mean diff: {lin_diff.mean():.10f}")

        # Show where max diff occurs
        max_idx = emb_diff.argmax()
        max_row = max_idx // 64
        max_col = max_idx % 64
        print(f"\nMax embedding diff at [{max_row}, {max_col}]:")
        print(f"  Non-parallel: {non_parallel_model.embedding.weight.data[max_row, max_col]:.10f}")
        print(f"  FSDP: {full_params['embedding.weight'][max_row, max_col]:.10f}")
        print(f"  Diff: {emb_diff[max_row, max_col]:.10f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    import sys

    world_size = 2

    # Allow optional command-line argument to test FP16
    # Usage: python debug_fsdp_detailed.py [fp16|fp32]
    compute_dtype = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "fp16":
            compute_dtype = torch.float16
            print("Testing with FP16 mixed precision")
        elif sys.argv[1] == "fp32":
            compute_dtype = None
            print("Testing with FP32")
        else:
            print(f"Unknown dtype: {sys.argv[1]}. Use 'fp16' or 'fp32'. Defaulting to FP32.")
    else:
        print("Testing with FP32 (default). Use 'python debug_fsdp_detailed.py fp16' for FP16.")
        sys.exit()

    mp.spawn(debug_worker, args=(world_size, compute_dtype), nprocs=world_size, join=True)
