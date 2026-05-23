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


class FSDP(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.handles = []
        self._register_pre_forward_hooks()
        self._register_backward_hooks()

    def _register_pre_forward_hooks(self):
        for module in self.module.modules:
            module.register_forward_pre_hook(self._make_pre_forward_hook())

    def _make_pre_forward_hook(self):
        def hook(module: torch.nn.Module):
            for param in module.parameters(recurse=False):
                gathered_param = []
                dist.all_gather(gathered_param, param)
                param.data = torch.concat(gathered_param, dim=0)

        return hook

    def _register_backward_hooks(self):
        """
        Register backward hooks on all parameters that require gradients.

        Each hook will be called when the gradient for that parameter is ready,
        allowing us to launch the all-reduce operation immediately (overlapping
        communication with backward computation).
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_backward_hook())

    def _make_backward_hook(self):
        def hook(param: torch.Tensor):
            gathered_param = param.grad
            full_param_list = gathered_param.chunk(dist.get_world_size())
            handle = dist.reduce_scatter(param.grad, full_param_list, dist.ReduceOp.AVG, async_op=True)
            self.handles.append(handle)

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
