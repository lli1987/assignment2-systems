"""
Distributed Data Parallel (DDP) with individual parameter gradient synchronization.

This implementation overlaps gradient communication with backward computation by
using autograd hooks to trigger asynchronous all-reduce operations as soon as
each parameter's gradient is ready.

Key features:
- Broadcasts parameters from rank 0 during initialization
- Registers backward hooks on each parameter
- Launches async all-reduce as soon as gradient is computed
- Provides a method to wait for all async operations to complete
"""

import torch
import torch.distributed as dist
from torch.nn.modules import Linear, Embedding


class FSDP(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.handles = []
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.param_metadata = dict()
        self._broadcast_params()  # Ensure all ranks start with same params
        self._shard_params()
        self._register_forward_pre_hooks()
        self._register_forward_hooks()
        self._register_backward_pre_hooks()
        self._register_backward_hooks()

    def _broadcast_params(self):
        """Broadcast all parameters from rank 0 to ensure consistency."""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _shard_params(self):
        for module in self.module.modules():
            if isinstance(module, (Linear, Embedding)):
                for param in module.parameters(recurse=False):
                    flatten = param.data.view(-1)
                    shard_size = flatten.size(dim=0) // self.world_size
                    local_shard_param_data = flatten[self.rank * shard_size : (self.rank + 1) * shard_size]
                    self.param_metadata[id(param)] = param.data.shape
                    param.data = local_shard_param_data

    def _register_forward_pre_hooks(self):
        for module in self.module.modules():
            if isinstance(module, (Linear, Embedding)):
                module.register_forward_pre_hook(self._make_forward_pre_hook())

    def _register_backward_pre_hooks(self):
        for module in self.module.modules():
            if isinstance(module, (Linear, Embedding)):
                module.register_full_backward_pre_hook(self._make_backward_pre_hook())

    def _register_forward_hooks(self):
        for module in self.module.modules():
            if isinstance(module, (Linear, Embedding)):
                module.register_forward_hook(self._make_forward_hook())

    def _register_backward_hooks(self):
        for module in self.module.modules():
            if isinstance(module, (Linear, Embedding)):
                # Sharded params: reduce-scatter
                module.register_full_backward_hook(self._make_backward_hook())
            elif list(module.parameters(recurse=False)):
                # Replicated params: all-reduce
                module.register_full_backward_hook(self._make_replicated_backward_hook())

    def _make_forward_hook(self):
        def hook(module: torch.nn.Module, input, output):
            for param in module.parameters(recurse=False):
                # Cast back to FP32 if we used compute_dtype
                if self.compute_dtype is not None:
                    param.data = param.data.to(torch.float32)

                flatten = param.data.view(-1)
                shard_size = flatten.size(dim=0) // self.world_size
                param.data = flatten[self.rank * shard_size : (self.rank + 1) * shard_size]

        return hook

    def _make_backward_pre_hook(self):
        def hook(module, grad_output):
            for param in module.parameters(recurse=False):
                flatten = param.data
                gathered_param = [torch.empty_like(flatten) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_param, flatten)
                full_flat = torch.concat(gathered_param, dim=0)
                param.data = full_flat.view(self.param_metadata[id(param)])

                # Cast to compute_dtype for backward pass if specified
                if self.compute_dtype is not None:
                    param.data = param.data.to(self.compute_dtype)

        return hook

    def _make_forward_pre_hook(self):
        def hook(module: torch.nn.Module, input):
            for param in module.parameters(recurse=False):
                flatten = param.data
                gathered_param = [torch.empty_like(flatten) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_param, flatten)
                full_flat = torch.concat(gathered_param, dim=0)
                param.data = full_flat.view(self.param_metadata[id(param)])

                # Cast to compute_dtype for forward pass if specified
                if self.compute_dtype is not None:
                    param.data = param.data.to(self.compute_dtype)

        return hook

    def _make_backward_hook(self):
        def hook(module: torch.nn.Module, grad_in, grad_out):
            for param in module.parameters(recurse=False):
                if param.grad is None:
                    continue

                # Cast gradient to FP32 if we used compute_dtype
                if self.compute_dtype is not None:
                    param.grad = param.grad.to(torch.float32)

                # Restore param.data to FP32 before re-sharding
                if self.compute_dtype is not None:
                    param.data = param.data.to(torch.float32)

                flatten_weights = param.data.view(-1)
                shard_size = flatten_weights.size(dim=0) // self.world_size
                param.data = flatten_weights[self.rank * shard_size : (self.rank + 1) * shard_size]

                flatten = param.grad.view(-1)
                full_param_list = list(torch.chunk(flatten, self.world_size))
                grad_shard = torch.empty(shard_size, dtype=param.grad.dtype, device=param.grad.device)
                dist.reduce_scatter(grad_shard, full_param_list, dist.ReduceOp.AVG)
                param.grad = grad_shard

        return hook

    def _make_replicated_backward_hook(self):
        """All-reduce gradients for replicated (non-sharded) parameters."""
        def hook(module: torch.nn.Module, grad_in, grad_out):
            for param in module.parameters(recurse=False):
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        return hook

    def finish_gradient_synchronization(self):
        """
        Wait for all async gradient communication to complete.

        This should be called after backward() and before optimizer.step()
        to ensure all gradients are synchronized across ranks before updating
        parameters.

        Example:
            loss.backward()  # Launches async all-reduce for each param
            ddp_model.finish_gradient_synchronization()  # Wait for all
            optimizer.step()  # Now safe to update parameters
        """
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def forward(self, *inputs, **kwargs):
        """
        Forward pass through the wrapped module.

        Args:
            *args: Positional arguments to pass to the module
            **kwargs: Keyword arguments to pass to the module

        Returns:
            Output from the wrapped module
        """
        return self.module(*inputs, **kwargs)

    def __repr__(self):
        """String representation showing the wrapped module."""
        return f"DDPIndividualParameters(\n  {self.module}\n)"
