# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Union

import torch
from torch.distributions import Normal

from mcbo.acq_funcs.acq_base import AcqBase
from mcbo.models import ModelBase


class RandomScalarization(AcqBase):
    """
    RandomScalarization of multi-objective.
    """

    @property
    def name(self) -> str:
        return f"RandScal-{self.base_acq.name}"
    
    @property
    def num_obj(self) -> int:
        return self.num_out

    @property
    def num_constr(self) -> int:
        return self.base_acq.num_constr
    
    def __init__(self, base_acq: AcqBase, aug_lambda: float = 0.05, num_out: int = 1):
        super(RandomScalarization, self).__init__()
        self.base_acq = base_acq
        self.aug_lambda = aug_lambda
        self.num_out = num_out
        
    def reset(self):
        weight = torch.rand(self.num_out)
        self.scalarization = (weight + self.aug_lambda) / (weight + self.aug_lambda).sum()
        
    def __call__(self, x: torch.Tensor, model: ModelBase, **kwargs) -> torch.Tensor:
        # acq values come back negative
        acq_values = self.base_acq(x, model, **kwargs)

        return (acq_values * self.scalarization).sum(dim=-1)
