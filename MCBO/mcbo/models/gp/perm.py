from typing import Union, List, Optional, Tuple, Dict, Any
from linear_operator.operators import LinearOperator

import numpy as np
import torch
from torch import Tensor
from gpytorch import settings, lazify, delazify
from gpytorch.priors import Prior
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.lazy import LazyEvaluatedKernelTensor
from torch.nn import ModuleList

from mcbo.search_space import SearchSpace
from mcbo.utils.hed_utils import diverse_random_dict_sample
from mcbo.utils.distance_metrics import compute_concordant_pairs

class KendallTauKernel(Kernel):
    """Implementation of the Kendall-Tau kernel.
    This kernel measures the number of concordant and discordant pairs, and 
    computes a correlation from it.
    """

    has_lengthscale = False

    @property
    def name(self) -> str:
        return "kendalltau"

    def __init__(self, **kwargs):
        super(KendallTauKernel, self).__init__(has_lengthscale=False, ard_num_dims=None, **kwargs)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> Tensor:
        max_pairs = (x1.shape[-1] * (x1.shape[-1] - 1)) / 2
        concordant_pairs = compute_concordant_pairs(x1, x2)
        discordant_pairs = max_pairs - concordant_pairs
        if diag:
            return torch.diagonal((concordant_pairs - discordant_pairs) / (max_pairs), dim1=-1, dim2=-2)
        return (concordant_pairs - discordant_pairs) / (max_pairs)


class MallowsKernel(Kernel):
    """Implementation of the Mallows kernel.
    This kernel measures the number of concordant and discordant pairs, and 
    computes a correlation from it.
    """

    has_lengthscale = True

    @property
    def name(self) -> str:
        return "mallows"

    def __init__(self, **kwargs):
        super(MallowsKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> Tensor:
        max_pairs = (x1.shape[-1] * (x1.shape[-1] - 1)) / 2
        concordant_pairs = compute_concordant_pairs(x1, x2)
        discordant_pairs = max_pairs - concordant_pairs
        if diag:
            return torch.diagonal(torch.exp(-discordant_pairs / self.lengthscale), dim1=-1, dim2=-2)
        return torch.exp(-discordant_pairs / self.lengthscale)
    

class AugmentedSpearmanKernel(Kernel):
    """The permutation kernel inspired by spearman distance from BaCO. 
    (Hellsten et. al, https://arxiv.org/pdf/2212.11142.pdf), where 
    large moves relatively more distant.
    """
    has_lengthscale = True

    @property
    def name(self) -> str:
        return "aug_spearman"

    def __init__(self, **kwargs):
        super(AugmentedSpearmanKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> Tensor:
        n_perm = x1.shape[-1]
        order_x1, order_x2 = x1.argsort(dim=-1) / n_perm, x2.argsort(dim=-1) / n_perm
        unnorm_dist = torch.pow((order_x1.unsqueeze(-2) - order_x2) / (self.lengthscale), 2).sum(dim=-1)
        if diag:
            return torch.diagonal(torch.exp(-unnorm_dist / 2), dim1=-1, dim2=-2)
        return torch.exp(-unnorm_dist / 2) 
