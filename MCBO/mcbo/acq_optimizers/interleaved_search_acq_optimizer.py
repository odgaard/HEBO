# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
from typing import Optional, Callable, Dict, List, Union

import numpy as np
import torch
from torch.autograd import grad
from mcbo.acq_funcs.acq_base import AcqBase
from mcbo.acq_optimizers.acq_optimizer_base import AcqOptimizerBase
from mcbo.models import ModelBase
from mcbo.search_space import SearchSpace
from mcbo.trust_region import TrManagerBase
from mcbo.trust_region.proxy_tr_manager import ProxyTrManager
from mcbo.trust_region.tr_utils import (
    sample_within_tr,
    sample_nominal_within_trust_region,
    sample_perm_within_trust_region,
)
from mcbo.utils.discrete_vars_utils import get_discrete_choices
from mcbo.utils.discrete_vars_utils import round_discrete_vars
from mcbo.utils.distance_metrics import hamming_distance, kendall_tau_distance
from mcbo.utils.model_utils import add_hallucinations_and_retrain_model
from mcbo.utils.plot_resource_utils import COLORS_SNS_10, get_color


class InterleavedSearchAcqOptimizer(AcqOptimizerBase):
    color_1: str = get_color(ind=1, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return InterleavedSearchAcqOptimizer.color_1

    @property
    def name(self) -> str:
        return "IS"

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 n_iter: int = 100,
                 n_raw: int = 512, 
                 n_restarts: int = 10,
                 max_n_perturb_num: int = 20,
                 num_optimizer: str = 'adam',
                 num_lr: Optional[float] = None,
                 nominal_tol: int = 100,
                 gd_iters: int = 20,
                 max_nominal_swap: int = 2,
                 nominal_iters: int = 10,
                 max_perm_swap: int = 2,
                 perm_iters: int = 10,
                 batch_limit: int = 256,
                 dtype: torch.dtype = torch.float64
                 ):
        super(InterleavedSearchAcqOptimizer, self).__init__(
            search_space=search_space,
            dtype=dtype,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals
        )
        assert search_space.num_cont + search_space.num_disc + search_space.num_nominal + search_space.num_permutation_dims == search_space.num_dims, \
            'Incorrect number of dimensions'

        self.n_iter = n_iter
        self.n_raw = n_raw
        self.batch_limit = batch_limit
        self.n_restarts = n_restarts
        self.max_n_perturb_num = max_n_perturb_num
        self.max_nominal_swap = max_nominal_swap

        max_perms = max(1, len(search_space.all_perm_dims) * (len(search_space.all_perm_dims) - 1) / 2) 
        self.max_perm_swap = max_perm_swap / max_perms
        self.num_optimizer = num_optimizer
        self.nominal_tol = nominal_tol
        self.nominal_iters = nominal_iters
        self.perm_iters = perm_iters
        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims
        self.gd_iters = gd_iters
        # Dimensions of discrete variables in tensors containing only numeric variables
        self.disc_dims_in_numeric = [i + len(self.search_space.cont_dims) for i in
                                     range(len(self.search_space.disc_dims))]

        self.discrete_choices = get_discrete_choices(search_space)
        # all perm dims
        self.inverse_mapping = [(self.numeric_dims + self.search_space.nominal_dims + self.search_space.all_perm_dims).index(i) for i in
                                range(self.search_space.num_dims)]

        # Determine the learning rate used to optimize numeric variables if needed
        if len(self.numeric_dims) > 0:
            if num_lr is None:
                if self.search_space.num_disc > 0:
                    num_lr = 1 / (len(self.discrete_choices[0]) - 1)
                else:
                    num_lr = 0.1
            else:
                assert 0 < num_lr < 1, \
                    'Numeric variables are normalised in the range [0, 1]. The learning rate should not exceed 1'
            self.num_lr = num_lr

    def optimize(self,
                 x: torch.Tensor,
                 n_suggestions: int,
                 x_observed: torch.Tensor,
                 model: ModelBase,
                 acq_func: AcqBase,
                 acq_evaluate_kwargs: dict,
                 tr_manager: Optional[TrManagerBase],
                 **kwargs
                 ) -> torch.Tensor:

        # if TR manager is None: we set TR to be the entire space
        if tr_manager is None:
            tr_manager = ProxyTrManager(search_space=self.search_space, dtype=self.search_space.dtype,
                                        obj_dims=self.obj_dims, out_constr_dims=self.out_constr_dims,
                                        out_upper_constr_vals=self.out_upper_constr_vals)
            if self.search_space.num_numeric > 0:
                tr_manager.register_radius('numeric', 0, 1, 1)
            if self.search_space.num_nominal > 0:
                tr_manager.register_radius('nominal', min_radius=0, max_radius=self.search_space.num_nominal + 1,
                                           init_radius=self.search_space.num_nominal + 1)
            if self.search_space.num_permutation_dims > 0:
                tr_manager.register_radius('perm', min_radius=1 / len(self.search_space.all_perm_dims), max_radius=1,
                                           init_radius=1)
            tr_manager.set_center(center=self.search_space.transform(self.search_space.sample()))
        
        if n_suggestions == 1:
            return self._optimize(
                x=x, n_suggestions=1, x_observed=x_observed,
                model=model, acq_func=acq_func,
                acq_evaluate_kwargs=acq_evaluate_kwargs, tr_manager=tr_manager
            )
        else:
            x_next = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)
            model = copy.deepcopy(
                model)  # create a local copy of the model to be able to retrain it  # TODO this fails when using the BOiLS model
            x_observed = x_observed.clone()

            for i in range(n_suggestions):
                x_ = self._optimize(
                    x=x, n_suggestions=1, x_observed=x_observed,
                    model=model, acq_func=acq_func, acq_evaluate_kwargs=acq_evaluate_kwargs,
                    tr_manager=tr_manager
                )
                x_next = torch.cat((x_next, x_), dim=0)

                # No need to add hallucinations during last iteration as the model will not be used
                if i < n_suggestions - 1:
                    x_observed = torch.cat((x_observed, x_), dim=0)
                    add_hallucinations_and_retrain_model(model, x_[0])

            return x_next

    def _optimize(self,
                  x: torch.Tensor,
                  n_suggestions: int,
                  x_observed: torch.Tensor,
                  model: ModelBase,
                  acq_func: AcqBase,
                  acq_evaluate_kwargs: dict,
                  tr_manager: Optional[TrManagerBase],
                  ) -> torch.Tensor:

        if n_suggestions > 1:
            import warnings
            warnings.warn('Interleaved search acquisition optimizer was not extended to batched setting')

        dtype, device = model.dtype, model.device

        # Sample initialisation
        x_centre = x.reshape(1, -1)
        x_feasible = torch.zeros((0, x_centre.shape[1]))

        while len(x_feasible) < self.n_raw:
            x_raw, numeric_lb, numeric_ub = sample_within_tr(
                x_centre=x_centre,
                search_space=self.search_space,
                tr_manager=tr_manager,
                n_points=self.n_raw,
                numeric_dims=self.numeric_dims,
                discrete_choices=self.discrete_choices,
                max_n_perturb_num=self.max_n_perturb_num,
                model=model,
                return_numeric_bounds=True
            )
            is_feasible = self.input_eval_from_transfx(transf_x=x_raw)
            x_feasible = torch.cat((x_feasible, x_raw[is_feasible.flatten()]), dim=0)
            
        x_feasible = x_feasible[:self.n_raw]

        acq_raw = torch.empty(self.n_raw)
        for i in range(np.ceil(self.n_raw / self.batch_limit).astype(int)):
            batch_limits = i * self.batch_limit, min((i + 1) * self.batch_limit, self.n_raw)
            x_batch = x_feasible[batch_limits[0]:batch_limits[1]]

            acq_raw[batch_limits[0]:batch_limits[1]] = acq_func(x=x_batch.to(device, dtype), model=model, **acq_evaluate_kwargs)
        
        topk = torch.topk(acq_raw, k=self.n_restarts, largest=False).indices
        x0 = x_feasible[topk]
        
        x, acq = [], []
        x_numeric = x0[:, self.numeric_dims]
        x_nominal = x0[:, self.search_space.nominal_dims]
        x_perm = x0[:, self.search_space.all_perm_dims]
        converged = False
        for _ in range(self.n_iter) or converged:
            if x_numeric.shape[1] > 0:
                # only taking one step, should take more (i.e. optimize the entire thing. Rewrite!)
                # Optimise numeric variables
                x_numeric.requires_grad_(True)
                x_cand = self._reconstruct_x(x_numeric, x_nominal, x_perm)
                x_num_update = x_cand[:, self.numeric_dims].clone()
                for _ in range(self.gd_iters):
                    x_prev = x_num_update.clone()

                    # acq is negative, we want to minimize
                    # TODO double check that this does what is intended
                    x_num_update = x_num_update - self.num_lr * grad(
                        acq_func(x_cand, model, **acq_evaluate_kwargs).sum(), x_cand)[0][:, self.numeric_dims]
                    x_num_update = torch.clip(x_num_update, numeric_lb, numeric_ub)
                    x_cand = self._reconstruct_x(x_num_update, x_nominal, x_perm)
                
                x_cand_copy = x_cand.clone()
                acq_x = acq_func(x=x_cand.to(device, dtype), model=model, **acq_evaluate_kwargs)

                with torch.no_grad():
                    x_numeric.data = round_discrete_vars(
                        x=x_numeric, discrete_dims=self.disc_dims_in_numeric,
                        choices=self.discrete_choices
                    )
                    x_numeric.data = torch.clip(x_numeric, min=numeric_lb, max=numeric_ub)
                    # check input constraints
                    if not np.all(self.input_eval_from_transfx(transf_x=x_cand)):
                        x_numeric = x_cand_copy[:, self.numeric_dims]

                x_numeric.requires_grad_(False)

            if x_nominal.shape[1] > 0:

                if self.search_space.num_numeric == 0: 
                    with torch.no_grad():
                        acq_x = acq_func(
                            x=self._reconstruct_x(x_nominal, x_perm).to(device, dtype), model=model,
                            **acq_evaluate_kwargs
                        )
                        
                swap_radius = min(tr_manager.get_nominal_radius(), self.max_nominal_swap)
        
                for _ in range(self.nominal_iters):
                    neighbours_nominal = sample_nominal_within_trust_region(
                        x_centre[:, self.search_space.nominal_dims],
                        search_space=self.search_space,
                        tr_manager=tr_manager,
                        n_points=self.n_restarts,
                        custom_radius=swap_radius, # only test maximally 2-swap neighborhoods
                    )
                    constraint_eval = np.all(self.input_eval_from_transfx(transf_x=self._reconstruct_x(x_numeric, neighbours_nominal, x_perm)), axis=1
                    )

                    with torch.no_grad():
                        x_cand = self._reconstruct_x(x_numeric, neighbours_nominal, x_perm)
                        acq_neighbour = acq_func(x=x_cand.to(device, dtype), model=model, **acq_evaluate_kwargs)

                    # acquisition is negative which is a little bit crayzay
                    improved = ((acq_neighbour < acq_x) * constraint_eval).to(torch.bool)
                    x_nominal[improved] = neighbours_nominal[improved]
                    acq_x[improved] = acq_neighbour[improved]
                
            if x_perm.shape[1] > 0:

                if self.search_space.num_numeric == 0: 
                    with torch.no_grad():
                        acq_x = acq_func(
                            x=self._reconstruct_x(x_nominal, x_perm).to(device, dtype), model=model,
                            **acq_evaluate_kwargs
                        )
                swap_radius = min(tr_manager.get_perm_radius(), self.max_perm_swap)
                for _ in range(self.perm_iters):
                    neighbours_perm = sample_perm_within_trust_region(
                        x_centre[:, self.search_space.all_perm_dims],
                        search_space=self.search_space,
                        tr_manager=tr_manager,
                        n_points=self.n_restarts,
                        custom_radius=swap_radius, # only test maximally 2-swap neighborhoods
                    )
                    # TODO check constraint evaluation
                    constraint_eval = np.all(self.input_eval_from_transfx(
                        transf_x=self._reconstruct_x(x_numeric, x_nominal, neighbours_perm)), 
                        axis=1
                    )
                    with torch.no_grad():
                        x_cand = self._reconstruct_x(x_numeric, x_nominal, neighbours_perm)
                        acq_neighbour = acq_func(x=x_cand.to(device, dtype), model=model, **acq_evaluate_kwargs)

                    # acquisition is negative which is a little bit crayzay
                    improved = ((acq_neighbour < acq_x) * constraint_eval).to(torch.bool)
                    x_perm[improved] = neighbours_perm[improved]
                    acq_x[improved] = acq_neighbour[improved]

        best_cands = self._reconstruct_x(x_numeric, x_nominal, x_perm)
        indices = torch.argsort(acq_x)
        
        valid = False
        for idx in indices:
            x_ = best_cands[idx]
            if torch.logical_not((x_.unsqueeze(0) == x_observed).all(axis=1)).all():
                valid = True
                break

        if not valid:
            raise ValueError("None of the candidates were valid")
        else:
            x_ = x_.unsqueeze(0)

        return x_

    # TODO perms
    def _reconstruct_x(self, x_numeric: torch.FloatTensor, x_nominal: torch.FloatTensor, x_perm: torch.FloatTensor) -> torch.FloatTensor:
        return torch.cat((x_numeric, x_nominal, x_perm), dim=-1)[:, self.inverse_mapping]