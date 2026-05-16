import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_process_group(rank: int, world_size: int, backend: str, port: int = 29500):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # For NCCL, set device before init
    if backend == "nccl":
        torch.cuda.set_device(rank)

    dist.init_process_group(backend, rank=rank, world_size=world_size)


def train_func(rank, world_size, batch_data, model, optimizer, backend="gloo"):
    setup_process_group(rank, world_size, backend)

    # Set device based on backend
    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Move model to the correct device
    model = model.to(device)

    # CRITICAL FIX #1: Broadcast parameters from rank 0 to all other ranks
    # This ensures all processes start with the same initial model
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # IMPROVEMENT: Use clearer data sharding logic
    # Each rank gets batch_size / world_size examples
    local_bs = batch_data.size(0) // world_size
    offset = rank * local_bs
    rows = batch_data[offset : offset + local_bs].to(device)

    optimizer.zero_grad()
    loss = model.forward(rows).mean()
    loss.backward()

    # All-reduce gradients across all ranks (averaging them)
    for name, param in model.named_parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, dist.ReduceOp.AVG)

    optimizer.step()

    # Clean up process group
    dist.destroy_process_group()

    # Return the updated model (will be sent back to main process)
    return model


def train_func_multiple_steps(rank, world_size, batch_data, model, optimizer, backend, num_steps):
    """
    Training function that supports multiple training steps.
    This is useful for verification tests.
    """
    setup_process_group(rank, world_size, backend)

    # Set device based on backend
    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Move model to the correct device
    model = model.to(device)

    # CRITICAL FIX #1: Broadcast parameters from rank 0 to all other ranks
    # This ensures all processes start with the same initial model
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Perform multiple training steps
    for step in range(num_steps):
        # IMPROVEMENT: Use clearer data sharding logic
        # Each rank gets batch_size / world_size examples
        local_bs = batch_data.size(0) // world_size
        offset = rank * local_bs
        rows = batch_data[offset : offset + local_bs].to(device)

        optimizer.zero_grad()
        loss = model.forward(rows).mean()
        loss.backward()

        # All-reduce gradients across all ranks (averaging them)
        for name, param in model.named_parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, dist.ReduceOp.AVG)

        optimizer.step()

    # Clean up process group
    dist.destroy_process_group()

    # Return the updated model (will be sent back to main process)
    return model


def native_ddp(model, optimizer, batch_data, d, backend="gloo", num_steps=1):
    """
    Perform naive distributed data parallel training.

    Args:
        model: The model to train
        optimizer: The optimizer to use
        batch_data: The full batch of data (will be sharded across ranks)
        d: Number of processes (world_size)
        backend: Backend to use ('gloo' for CPU, 'nccl' for GPU)
        num_steps: Number of training steps to perform

    Note: The original model is not modified. To get the trained model,
    you need to save it from within the worker process or use the
    verification function below.
    """
    mp.spawn(
        fn=train_func_multiple_steps,
        args=(d, batch_data, model, optimizer, backend, num_steps),
        nprocs=d,
        join=True
    )


def _ddp_verification_worker(rank, world_size, all_data, all_labels, in_features, out_features, num_steps, backend):
    """Worker function that performs DDP training and saves final model state."""
    from toy_model import ToyModel
    import torch.optim as optim
    import torch.nn as nn

    setup_process_group(rank, world_size, backend)

    device = torch.device("cpu") if backend == "gloo" else torch.device(f"cuda:{rank}")

    # Each worker creates its own model with the same seed
    torch.manual_seed(0)  # Same seed as single-process
    ddp_model = ToyModel(in_features, out_features).to(device)
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Broadcast parameters from rank 0 to all ranks
    for param in ddp_model.parameters():
        dist.broadcast(param.data, src=0)

    loss_fn = nn.MSELoss()

    # Training loop
    local_bs = all_data.size(0) // world_size
    offset = rank * local_bs

    for step in range(num_steps):
        # Shard the data
        local_data = all_data[offset : offset + local_bs].to(device)
        local_labels = all_labels[offset : offset + local_bs].to(device)

        ddp_optimizer.zero_grad()
        outputs = ddp_model(local_data)
        loss = loss_fn(outputs, local_labels)
        loss.backward()

        # All-reduce gradients
        for param in ddp_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, dist.ReduceOp.AVG)

        ddp_optimizer.step()

        if rank == 0 and (step == 0 or step == num_steps - 1):
            print(f"   Step {step}: Loss (rank 0) = {loss.item():.6f}")

    # Save the final model state from rank 0
    if rank == 0:
        torch.save(ddp_model.state_dict(), "/tmp/ddp_model_state.pt")

    dist.destroy_process_group()


def verify_ddp_correctness(world_size=2, batch_size=20, num_steps=5, backend="gloo"):
    """
    Verify that DDP training produces the same results as single-process training.

    This function:
    1. Trains a model using single-process on all data
    2. Trains the same model using DDP with sharded data
    3. Compares the final weights to verify they match

    Args:
        world_size: Number of processes for DDP
        batch_size: Total batch size (must be divisible by world_size)
        num_steps: Number of training steps
        backend: 'gloo' for CPU, 'nccl' for GPU
    """
    from toy_model import ToyModel
    import torch.optim as optim
    import torch.nn as nn

    assert batch_size % world_size == 0, "batch_size must be divisible by world_size"

    # Set a seed for reproducibility
    torch.manual_seed(42)

    # Create the full dataset
    in_features = 10
    out_features = 5
    all_data = torch.randn(batch_size, in_features)
    all_labels = torch.randn(batch_size, out_features)

    print("="*80)
    print("VERIFICATION TEST: Single-process vs DDP Training")
    print("="*80)

    # ===== SINGLE-PROCESS BASELINE =====
    print("\n1. Running single-process baseline training...")
    torch.manual_seed(0)  # Same seed for model initialization
    single_process_model = ToyModel(in_features, out_features)
    single_process_optimizer = optim.SGD(single_process_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for step in range(num_steps):
        single_process_optimizer.zero_grad()
        outputs = single_process_model(all_data)
        loss = loss_fn(outputs, all_labels)
        loss.backward()
        single_process_optimizer.step()
        if step == 0 or step == num_steps - 1:
            print(f"   Step {step}: Loss = {loss.item():.6f}")

    # ===== DDP TRAINING =====
    print(f"\n2. Running DDP training with {world_size} processes...")

    # Run DDP training
    mp.spawn(
        _ddp_verification_worker,
        args=(world_size, all_data, all_labels, in_features, out_features, num_steps, backend),
        nprocs=world_size,
        join=True
    )

    # Load the DDP model state
    torch.manual_seed(0)
    ddp_model = ToyModel(in_features, out_features)
    ddp_model.load_state_dict(torch.load("/tmp/ddp_model_state.pt"))

    # ===== COMPARISON =====
    print("\n3. Comparing final model parameters...")
    all_close = True
    max_diff = 0.0

    for (name, single_param), (_, ddp_param) in zip(
        single_process_model.named_parameters(),
        ddp_model.named_parameters()
    ):
        if torch.allclose(single_param, ddp_param, rtol=1e-5, atol=1e-6):
            print(f"   ✓ {name}: MATCH")
        else:
            diff = (single_param - ddp_param).abs().max().item()
            max_diff = max(max_diff, diff)
            print(f"   ✗ {name}: MISMATCH (max diff: {diff:.2e})")
            all_close = False

    print("\n" + "="*80)
    if all_close:
        print("SUCCESS: DDP training matches single-process training!")
    else:
        print(f"FAILURE: Maximum difference = {max_diff:.2e}")
    print("="*80 + "\n")

    return all_close


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    # Run verification test
    success = verify_ddp_correctness(
        world_size=2,
        batch_size=20,
        num_steps=5,
        backend="gloo"
    )
