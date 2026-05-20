from typing import Any, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class OptimizerWithStateSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        # Initialize all instance variables BEFORE calling super().__init__
        # because super().__init__ may call add_param_group which needs these
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer = optimizer_cls([{"params": []}])
        self.index = 0

        super().__init__(params, kwargs)

    def step(self, closure=None):
        """Step the optimizer and broadcast updated parameters."""
        # Step only updates this rank's parameters
        self.optimizer.step(closure=closure)

        # Broadcast ALL parameters (not just this rank's!)
        # All ranks must participate in each broadcast
        param_idx = 0
        for group in self.param_groups:  # Use parent's param_groups (all params)
            for param in group["params"]:
                # Determine which rank owns this parameter
                owner_rank = param_idx % self.world_size
                # All ranks call broadcast, but only owner is the source
                dist.broadcast(param.data, src=owner_rank)
                param_idx += 1

    def add_param_group(self, param_group: dict[str, Any]):
        """
        Add a param_group.

        - Parent class gets ALL parameters (for broadcasting)
        - Underlying optimizer gets only this rank's shard
        """
        # First, add ALL params to parent class
        super().add_param_group(param_group)

        # Then, shard the params for the underlying optimizer
        param_list = param_group["params"]
        my_params = []
        for param in param_list:
            if self.index % self.world_size == self.rank:
                my_params.append(param)
            self.index += 1

        # Add sharded group to underlying optimizer (only if we have params)
        if my_params:
            # Copy all hyperparameters except 'params'
            sharded_group = {k: v for k, v in param_group.items() if k != "params"}
            sharded_group["params"] = my_params
            self.optimizer.add_param_group(sharded_group)
