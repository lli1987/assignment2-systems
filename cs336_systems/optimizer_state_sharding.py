from typing import Any

import torch
from torch.optim import Optimizer


class OptimizerWithStateSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        pass

    def step(self, closure, **kwargs):
        pass

    def add_param_group(self, param_group: dict[str, Any]):
        pass
