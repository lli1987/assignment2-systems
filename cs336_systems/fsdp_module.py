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
    """
    DDP wrapper that communicates gradients individually per parameter.

    This is the "naive" overlapping approach that launches one all-reduce
    operation per parameter as soon as its gradient is ready during the
    backward pass.

    Usage:
        model = MyModel()
        ddp_model = DDPIndividualParameters(model)

        # Training loop
        optimizer.zero_grad()
        loss = ddp_model(inputs)
        loss.backward()  # Hooks fire async all-reduce for each param
        ddp_model.finish_gradient_synchronization()  # Wait for all to complete
        optimizer.step()

    Attributes:
        module: The wrapped model
        process_group: The process group for distributed communication
        _async_handles: List of async operation handles to wait on
    """

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        """
        Initialize the DDP wrapper.

        Args:
            module: The model to wrap
        """
        super().__init__()
        self.module = module
        self.handles = []
        self._broadcast_parameters()
        self._register_backward_hooks()

    def _broadcast_parameters(self):
        """
        Broadcast all parameters from rank 0 to all other ranks.

        This is called during initialization to synchronize the initial model state.
        """
        for param in self.module.parameters():
            dist.broadcast(tensor=param.data, src=0)

    def _register_backward_hooks(self):
        """
        Register backward hooks on all parameters that require gradients.

        Each hook will be called when the gradient for that parameter is ready,
        allowing us to launch the all-reduce operation immediately (overlapping
        communication with backward computation).
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param: torch.nn.Parameter):
        """
        Create a backward hook for a specific parameter.

        The hook launches an async all-reduce operation when the parameter's
        gradient becomes available.

        Args:
            param: The parameter to create a hook for (used for closure context)

        Returns:
            A hook function that will be called with the gradient tensor

        Note:
            The param argument is kept for potential future use (e.g., debugging,
            selective sync) and to maintain a clear factory pattern.
        """

        def hook(param: torch.Tensor):
            handle = dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=True)
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
