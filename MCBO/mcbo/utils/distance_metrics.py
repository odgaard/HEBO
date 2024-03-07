# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import torch
from torch import Tensor

def hamming_distance(x1: Tensor, x2: Tensor, normalize: bool = False) -> Tensor:
    # TODO check that the 
    if len(x1) == 0:
        assert len(x2) == 0
        return 0
    delta = torch.abs(x1 - x2) > 1e-6

    if delta.ndim == 1:
        delta_sum = torch.sum(delta)
        if normalize:
            delta_sum = delta_sum / len(delta)
        return delta_sum
    else:
        delta_sum = torch.sum(delta, axis=-1)
        if normalize:
            delta_sum = delta_sum / delta.shape[1]
        return delta_sum


def euclidean_distance(x1: Tensor, x2: Tensor) -> Tensor:
    delta = ((x1 - x2) ** 2).sum(dim=-1)
    delta = np.sqrt(delta)
    return delta



def maxnorm_distance(x1: Tensor, x2: Tensor) -> Tensor:
    delta = (x1 - x2).abs().max(dim=-1).values
    return delta

def kendall_tau_distance(x1: Tensor, x2: Tensor, normalize: bool = True) -> Tensor:
    """Computes the kendall tau distance between two tensors of
     permutations.

    Args:
        x1 (Tensor): [batch_shape] x num_perms tensor of permutations.
        x2 (Tensor): [batch_shape] x num_perms tensor of permutations.

    Returns:
        batch_shape x batch_shape tensor of the number of
        concordant pairs.
    """
    if x1.ndim == 1:
        x1 = x1.reshape(1, -1)
    if x2.ndim == 1:
        x2 = x2.reshape(1, -1)
    
    max_pairs = (x1.shape[-1] * (x1.shape[-1] - 1)) / 2
    concordant_pairs = compute_concordant_pairs(x1, x2)
    discordant_pairs = max_pairs - concordant_pairs
    if normalize:
        return (discordant_pairs / max_pairs).squeeze(-1)
    else:
        return discordant_pairs.squeeze(-1)

def compute_concordant_pairs(x1: Tensor, x2: Tensor) -> Tensor:
    """Computes the number of concordant and discordant pairs
    in the permutation tensors.

    Args:
        x1 (Tensor): batch_shape x num_perms tensor of permutations.
        x2 (Tensor): batch_shape x num_perms tensor of permutations.

    Returns:
        batch_shape x batch_shape tensor of the number of
        concordant pairs.
    """
    x1 = x1.argsort(-1).unsqueeze(-2)
    x2 = x2.argsort(-1).unsqueeze(-3)
    
    order_diffs = ((x1.unsqueeze(-2) - x1.unsqueeze(-1))
            * (x2.unsqueeze(-2) - x2.unsqueeze(-1))
    )
    concordant_pairs = (order_diffs.tril() > 0).sum(dim=[-1, -2])

    return concordant_pairs


def position_distance(x1, x2):
    # position distance for permutation based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9002675
    # Note, to use numba, remove the axis variable from argsort and use a for loop to process all of the data
    x1 = torch.argsort(x1, axis=1)
    x2 = torch.argsort(x2, axis=1)

    return torch.abs(x1 - x2).sum(axis=1)
