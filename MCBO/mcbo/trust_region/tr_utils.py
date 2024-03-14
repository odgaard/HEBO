# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Optional, List, Union, Dict, Callable, Tuple

import numpy as np
import torch
from torch import Tensor
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from torch.quasirandom import SobolEngine

from mcbo.models import ExactGPModel, ModelBase
from mcbo.models.gp.kernels import MixtureKernel, ConditionalTransformedOverlapKernel, DecompositionKernel
from mcbo.search_space import SearchSpace
from mcbo.trust_region import TrManagerBase
from mcbo.utils.discrete_vars_utils import round_discrete_vars


def get_num_tr_bounds(
        x_num: torch.Tensor,
        tr_manager: TrManagerBase,
        is_numeric: bool,
        is_mixed: bool,
        kernel: Optional[MixtureKernel] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 'numeric' in tr_manager.radii, "numeric not in radii"
    # This function requires the mixture kernel or the RBF or Matérn kernel to use lengthscales as weights
    if is_numeric:
        weights = get_numdim_weights(
            num_dim=x_num.shape[-1], is_numeric=is_numeric, is_mixed=is_mixed, kernel=kernel
        )

        # continuous variables
        lb = torch.clip(x_num - weights.to(x_num) * tr_manager.radii['numeric'] / 2.0, 0.0, 1.0)
        ub = torch.clip(x_num + weights.to(x_num) * tr_manager.radii['numeric'] / 2.0, 0.0, 1.0)

    else:
        lb = torch.zeros_like(x_num)
        ub = torch.ones_like(x_num)

    return lb, ub


def get_numdim_weights(num_dim: int, is_numeric: bool, is_mixed: bool, kernel: Optional[MixtureKernel]) -> torch.Tensor:
    if kernel is not None:
        if isinstance(kernel, ScaleKernel):
            kernel = kernel.base_kernel
        valid_kernel = False
        if is_mixed:
            if isinstance(kernel, MixtureKernel):
                valid_kernel = True
            elif isinstance(kernel, (ConditionalTransformedOverlapKernel, DecompositionKernel)):
                valid_kernel = True
            else:
                raise ValueError(kernel)
        elif is_numeric:
            if isinstance(kernel, (RBFKernel, MaternKernel, DecompositionKernel)):
                valid_kernel = True
            else:
                raise ValueError(kernel)
        if not valid_kernel:
            kernel = None

    if is_numeric:

        if kernel is not None:

            if is_mixed:
                if isinstance(kernel, (ConditionalTransformedOverlapKernel, DecompositionKernel)):
                    weights = kernel.get_lengthcales_numerical_dims().detach()
                elif isinstance(kernel, MixtureKernel):
                    weights = kernel.numeric_kernel.lengthscale.detach().cpu()
                else:
                    raise ValueError(kernel)

            else:
                if hasattr(kernel, "lengthscale"):
                    weights = kernel.lengthscale.detach().cpu()
                else:
                    raise ValueError(kernel)

            # Normalise the weights so that we have weights.prod() = 1
            weights = weights[0] / weights.mean()
            weights = weights / torch.pow(torch.prod(weights), 1 / len(weights))

        else:
            weights = torch.ones(num_dim)

    return weights


def sample_within_tr(
        x_centre: torch.Tensor,
        search_space: SearchSpace,
        tr_manager: TrManagerBase,
        n_points: int,
        numeric_dims: List[int],
        discrete_choices: Union[torch.FloatTensor, List[torch.Tensor]],
        seq_dims: Optional[Union[np.ndarray, List[int]]] = None,
        max_n_perturb_num: int = 10,
        model: Optional[ModelBase] = None,
        return_numeric_bounds: bool = False,
):
    is_numeric = search_space.num_numeric > 0
    is_mixed = is_numeric and search_space.num_nominal > 0

    if seq_dims is not None:
        nominal_dims = [nd for nd in search_space.nominal_dims if nd not in seq_dims]
    else:
        nominal_dims = search_space.nominal_dims

    x_centre = x_centre * torch.ones((n_points, search_space.num_dims)).to(x_centre)

    if search_space.num_numeric > 0:
        numeric_centre = x_centre[0, numeric_dims]
        x_numeric_new = sample_numeric_within_trust_region(
            tr_centre=numeric_centre, 
            search_space=search_space,
            tr_manager=tr_manager,
            n_points=n_points,
            discrete_choices=discrete_choices,
            max_n_perturb_num=max_n_perturb_num,
            model=model,
        )
        x_centre[:, numeric_dims] = x_numeric_new
    
    if len(nominal_dims) > 0:
        nominal_centre = x_centre[0, nominal_dims]
        x_nominal_new = sample_nominal_within_trust_region(
            tr_centre=nominal_centre, 
            search_space=search_space,
            tr_manager=tr_manager,
            n_points=n_points,
            nominal_dims=nominal_dims,
        )
        x_centre[:, nominal_dims] = x_nominal_new
        
    if search_space.num_permutation_dims > 0:
        perm_centre = x_centre[0, search_space.all_perm_dims]
        x_perm_new = sample_perm_within_trust_region(
            tr_centre=perm_centre, 
            search_space=search_space,
            tr_manager=tr_manager,
            n_points=n_points,
            custom_radius=max_n_perturb_num,
        )
        x_centre[:, search_space.all_perm_dims] = x_perm_new
    
    if seq_dims is not None:  # true for EDA task
        n_perturb_nominal = np.random.randint(low=0, high=tr_manager.radii['sequence'] + 1, size=n_points)
        from mcbo.search_space.search_space_eda import SearchSpaceEDA
        assert isinstance(search_space, SearchSpaceEDA)
        for i in range(n_points):
            for j in range(n_perturb_nominal[i]):
                dim = np.random.choice(seq_dims, replace=True)
                choices = [val.item() for val in
                           search_space.get_transformed_mutation_cand_values(transformed_x=x_centre[i], dim=dim) if
                           val != x_centre[i, dim].item()]
                if len(choices) > 0:
                    x_centre[i, dim] = np.random.choice(choices, replace=True)

    if return_numeric_bounds:
        if search_space.num_numeric > 0:
            num_lb, num_ub = get_tr_bounds_from_model(
                numeric_centre, 
                tr_manager=tr_manager, 
                search_space=search_space, 
                model=model
            )

            return x_centre, num_lb, num_ub
        else:
            return x_centre, None, None
    else:
        return x_centre

def sample_perm_within_trust_region(
        tr_centre: Tensor, 
        search_space: SearchSpace, 
        tr_manager: TrManagerBase, 
        n_points: int,
        custom_radius: Optional[float] = None,
    ):
        if custom_radius:
            rad = custom_radius
        else:
            rad = tr_manager.get_perm_radius()

        if tr_centre.ndim == 1:
            x_perm = tr_centre.reshape(1, -1).repeat(n_points, 1)     
        else:
            x_perm = tr_centre

        max_pairs = (search_space.num_permutation_dims * (search_space.num_permutation_dims - 1)) / 2
        num_neighbor_swaps = np.floor(rad * max_pairs).astype(int)
        # we sequentially swap the neighbors at locations perm_seq, perm_seq + 1
        permute_sequence = np.random.choice(x_perm.shape[1] - 1, size=(num_neighbor_swaps, n_points))
        
        # inplace f**s things up
        x_perm_new = x_perm.clone()
        mask = np.arange(len(x_perm_new))
        for perm in permute_sequence:
            x_perm_new[mask, perm+1], x_perm_new[mask, perm] = x_perm_new[mask, perm], x_perm_new[mask, perm+1]
        return x_perm_new 


def sample_nominal_within_trust_region(
        tr_centre: Tensor, 
        search_space: SearchSpace, 
        tr_manager: TrManagerBase, 
        n_points: int, 
        nominal_dims: Optional[List] = None, # exists simply if we happen to have sequence dims
        custom_radius: Optional[float] = None,
    ):
        if nominal_dims is None:
            nominal_dims = search_space.nominal_dims
                
        if len(nominal_dims) == 0:
            return torch.zeros((n_points, 0))
        
        if custom_radius:
            rad = custom_radius
        else:
            rad = tr_manager.get_nominal_radius()
            
        nominal_params = [search_space.params[search_space.param_names[dim]] for dim in nominal_dims]

        if tr_centre.ndim == 1:
            x_nominal = tr_centre.reshape(1, -1).repeat(n_points, 1)     
        else:
            x_nominal = tr_centre

        n_params = len(nominal_params)
        choices = np.array([[param.lb, param.ub] for param in nominal_params]) + 1
        # this sampling routine is not uniform in the search space
        # since there is much larger density on the TR center (and configs close to it
        # than any other (1/tr_radius of the density)
        # nevertheless, sampling uniformly within the trust region is undesirable since a param
        # with a large number of options will disproportionally be chosen
        # but, offsetting more parameters should be (exponentially) more common than offsetting few
        
        n_perturb_nominal = np.random.binomial(rad, n_params / (rad + n_params), size=(n_points, 1))
        dims_to_change = np.random.uniform(size=(n_points, n_params)).argsort() < n_perturb_nominal
        
        # to ensure the actual move, we use modulo --> move the categorical to an offset
        offset = np.random.randint(*choices.T, size=(n_points, len(choices)))
        x_offset = (x_nominal + dims_to_change * offset)
        x_nominal_new = x_offset % choices[:, 1]
        return x_nominal_new        


def get_tr_bounds_from_model(
    tr_centre: Tensor, 
    tr_manager: TrManagerBase, 
    search_space: SearchSpace,
    model: ModelBase = None,      
):
    if (model is not None) and isinstance(model, ExactGPModel):
        kernel = model.gp.kernel
    else:
        kernel = None

    is_numeric = search_space.num_numeric > 0
    is_mixed = is_numeric and search_space.num_nominal > 0
    num_lb, num_ub = get_num_tr_bounds(tr_centre, tr_manager, is_numeric, is_mixed, kernel)
    return num_lb, num_ub


def sample_numeric_within_trust_region(
        tr_centre: Tensor, 
        search_space: SearchSpace, 
        tr_manager: TrManagerBase, 
        n_points: int,
        discrete_choices: List[Tensor],
        max_n_perturb_num: Optional[int] = 20,
        model: Optional[ModelBase] = None,
        custom_radius: Optional[float] = None,
    ):
        num_lb, num_ub = get_tr_bounds_from_model(
            tr_centre,
            tr_manager,
            search_space,
            model,
        )
        
        x_num = tr_centre.reshape(1, -1).repeat(n_points, 1)   
        seed = np.random.randint(int(1e6))
        sobol_engine = SobolEngine(search_space.num_numeric, scramble=True, seed=seed)
        pert = sobol_engine.draw(n_points).to(x_num)
        pert = (num_ub - num_lb) * pert + num_lb

        perturb_prob = min(max_n_perturb_num / search_space.num_numeric, 1.0)
        mask = torch.rand(n_points, search_space.num_numeric) <= perturb_prob
        ind = torch.where(torch.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, search_space.num_numeric, size=len(ind))] = 1
        x_num = torch.where(mask.to(x_num.device), pert, x_num)
        x_num_new = round_discrete_vars(x_num, search_space.disc_dims, discrete_choices)

        return x_num_new



def filter_within_trust_region(
    x_observed: Tensor, 
    tr_centre: Tensor, 
    search_space: SearchSpace,
    radii: Dict[str, float], 
    distances: Dict[str, Callable],
    return_mask: bool = False,
    ) -> Tensor:
    """Filters data from 'x_observed' to only include data that is located within the trust region
    with center 'tr_centre', using the specified distance metrics and radii.

    Args:
        x_observed (Tensor): _description_
        tr_centre (Tensor): _description_
        search_space (SearchSpace): _description_
        distances (Dict[Callable]): _description_

    Returns:
        Tensor: _description_
    """ 
    dim_masks = {}   
    tr_centre = tr_centre.reshape(1, -1)
    dim_masks['numeric'] = search_space.cont_dims + search_space.disc_dims
    dim_masks['nominal'] = search_space.nominal_dims
    dim_masks['perm'] = search_space.all_perm_dims
    inside_tr = torch.ones((len(x_observed)))
    for vartype, subset_dims in dim_masks.items():
        if len(subset_dims) > 0:
            vartype_mask = distances[vartype](x_observed[:, subset_dims], tr_centre[:, subset_dims]) < radii[vartype]
            inside_tr = inside_tr * vartype_mask

    if return_mask:
        return inside_tr.bool()
    
    return x_observed[inside_tr.bool()]    

