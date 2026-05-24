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

# Import from cs336_basics where the actual modules are defined
try:
    from cs336_basics.model import Linear, Embedding
except ImportError:
    # Fallback to torch.nn if cs336_basics not available
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
        self.saved_fp32_shards = dict()  # Save FP32 shards to avoid re-casting errors
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
        # Register parameter-level hooks instead of module-level hooks
        for module in self.module.modules():
            for param in module.parameters(recurse=False):
                if isinstance(module, (Linear, Embedding)):
                    # Sharded params: reduce-scatter
                    param.register_post_accumulate_grad_hook(self._make_param_backward_hook())
                else:
                    # Replicated params: all-reduce
                    param.register_post_accumulate_grad_hook(self._make_replicated_param_backward_hook())

    def _make_forward_hook(self):
        def hook(module: torch.nn.Module, input, output):
            for param in module.parameters(recurse=False):
                # Restore the saved FP32 shard instead of casting (avoids precision loss)
                if self.compute_dtype is not None and id(param) in self.saved_fp32_shards:
                    param.data = self.saved_fp32_shards[id(param)]
                    del self.saved_fp32_shards[id(param)]
                else:
                    # No mixed precision, just re-shard
                    flatten = param.data.view(-1)
                    shard_size = flatten.size(dim=0) // self.world_size
                    param.data = flatten[self.rank * shard_size : (self.rank + 1) * shard_size]

        return hook

    def _make_backward_pre_hook(self):
        def hook(module, grad_output):
            for param in module.parameters(recurse=False):

                flatten = param.data  # FP32 shard

                # Cast to compute_dtype BEFORE all-gather (saves bandwidth!)
                if self.compute_dtype is not None:
                    self.saved_fp32_shards[id(param)] = param.data.clone()
                    flatten = flatten.to(self.compute_dtype)

                gathered_param = [torch.empty_like(flatten) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_param, flatten)
                full_flat = torch.concat(gathered_param, dim=0)
                param.data = full_flat.view(self.param_metadata[id(param)])
                # param.data is now in compute_dtype for backward computation

        return hook

    def _make_forward_pre_hook(self):
        def hook(module: torch.nn.Module, input):
            for param in module.parameters(recurse=False):
                # Save the FP32 shard to restore later (avoids re-casting errors)
                if self.compute_dtype is not None:
                    self.saved_fp32_shards[id(param)] = param.data.clone()

                flatten = param.data  # FP32 shard

                # Cast to compute_dtype BEFORE all-gather (saves bandwidth!)
                if self.compute_dtype is not None:
                    flatten = flatten.to(self.compute_dtype)

                # All-gather in compute_dtype (FP16 if specified)
                gathered_param = [torch.empty_like(flatten) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_param, flatten)
                full_flat = torch.concat(gathered_param, dim=0)
                param.data = full_flat.view(self.param_metadata[id(param)])

        return hook

    def _make_param_backward_hook(self):
        """Create backward hook for sharded parameters (Linear/Embedding)."""

        def hook(p: torch.Tensor):
            if p.grad is None:
                return None

            # Cast gradient to FP32 if we used compute_dtype (before reduce-scatter)
            grad = p.grad
            if self.compute_dtype is not None:
                grad = grad.to(torch.float32)

            # Reduce-scatter gradient (in FP32)
            flatten = grad.view(-1)
            shard_size = flatten.size(dim=0) // self.world_size
            full_param_list = list(torch.chunk(flatten, self.world_size))
            grad_shard = torch.empty(shard_size, dtype=grad.dtype, device=grad.device)
            dist.reduce_scatter(grad_shard, full_param_list, dist.ReduceOp.AVG)

            # Restore the saved FP32 shard instead of casting (avoids precision loss)
            if self.compute_dtype is not None and id(p) in self.saved_fp32_shards:
                p.data = self.saved_fp32_shards[id(p)]
                del self.saved_fp32_shards[id(p)]
            else:
                # No mixed precision, just re-shard
                flatten_weights = p.data.view(-1)
                param_shard = flatten_weights[self.rank * shard_size : (self.rank + 1) * shard_size].clone()
                p.data = param_shard

            # Modify param.grad in place to be the sharded gradient (already FP32)
            p.grad = grad_shard

            # Must return None for post_accumulate_grad_hook
            return None

        return hook

    def _make_replicated_param_backward_hook(self):
        """Create backward hook for replicated parameters (RMSNorm, etc)."""

        def hook(p: torch.Tensor):
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            # Must return None for post_accumulate_grad_hook
            return None

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
