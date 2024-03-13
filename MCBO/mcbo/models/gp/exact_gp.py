# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import copy
import warnings
from typing import Optional, List, Callable

import gpytorch
import math
import numpy as np
import torch
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.kernels import Kernel, MultitaskKernel, ScaleKernel
from gpytorch.likelihoods import (
    GaussianLikelihood, 
    MultitaskGaussianLikelihood, 
    BernoulliLikelihood
)
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
from gpytorch.priors import Prior, LogNormalPrior
from gpytorch.utils.errors import NotPSDError, NanError

from mcbo.models.gp.kernels import MixtureKernel, get_numeric_kernel_name
from mcbo.models.model_base import ModelBase
from mcbo.search_space import SearchSpace
from mcbo.utils.training_utils import subsample_training_data, remove_repeating_samples



class ExactGPModel(ModelBase, torch.nn.Module):
    supports_cuda = True

    support_grad = True
    support_multi_output = True

    @property
    def name(self) -> str:
        name = "GP"
        kernel = self.kernel
        if isinstance(kernel, ScaleKernel):
            kernel = kernel.base_kernel
        if isinstance(kernel, MixtureKernel):
            kernel_name = kernel.name
        elif self.search_space.num_params == self.search_space.num_numeric:
            kernel_name = get_numeric_kernel_name(kernel)
        else:
            kernel_name = kernel.name
        name += f" ({kernel_name})"
        return name
    def __init__(
            self,
            search_space: SearchSpace,
            kernel: Kernel,
            num_out: int,
            noise_prior: Optional[Prior] = None,
            noise_constr: Optional[Interval] = None,
            noise_lb: float = 1e-5,
            pred_likelihood: bool = True,
            lr: float = 3e-3,
            num_epochs: int = 100,
            optimizer: str = 'adam',
            max_cholesky_size: int = 2000,
            max_training_dataset_size: int = 1000,
            max_batch_size: int = 1000,
            verbose: bool = False,
            print_every: int = 10,
            binary_classification: bool = False,
            dtype: torch.dtype = torch.float64,
            device: torch.device = torch.device('cpu'),
            impute_invalid: Callable = torch.mean,
    ):

        super(ExactGPModel, self).__init__(search_space=search_space, dtype=dtype, num_out=num_out, device=device)
        self.kernel = copy.deepcopy(kernel)
        self.lr = lr
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.max_cholesky_size = max_cholesky_size
        self.max_training_dataset_size = max_training_dataset_size
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        self.print_every = print_every
        self.binary_classification = binary_classification
        self.impute_invalid_fun = impute_invalid
        if noise_prior is None:
            noise_prior = LogNormalPrior(-4.63, 0.5)
        else:
            assert isinstance(noise_prior, Prior)

        if noise_constr is None:
            assert noise_lb is not None
            self.noise_lb = noise_lb
            noise_constr = GreaterThan(noise_lb)
        else:
            assert isinstance(noise_constr, Interval)

        # Model settings
        self.pred_likelihood = pred_likelihood
        self.batch_shape = torch.Size([num_out]) if num_out > 1 else torch.Size([])
        
        if binary_classification:
            self.likelihood = BernoulliLikelihood()
        else:
            self.likelihood = GaussianLikelihood(noise_constraint=noise_constr, noise_prior=noise_prior, batch_shape=self.batch_shape)

        self.gp = None

    def fit_y_to_y(self, fit_y: torch.Tensor) -> torch.Tensor:
        return fit_y * self.y_std.to(fit_y) + self.y_mean.to(fit_y)

    def y_to_fit_y(self, y: torch.Tensor) -> torch.Tensor:
        # Normalise target values
        fit_y = (y - self.y_mean.to(y)) / self.y_std.to(y)

        # Add a small amount of noise to prevent training instabilities
        fit_y += 1e-6 * torch.randn_like(fit_y)
        return fit_y

    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> List[float]:
        assert x.ndim == 2
        assert x.shape[0] == y.shape[0]
        invalid_indices = torch.any(torch.isnan(y), dim=1)
        if torch.any(torch.isnan(y)):
            # if there are no valid indices
            if invalid_indices.sum() == len(y):
                y = torch.zeros_like(y)
            else:
                impute_value = self.impute_invalid_fun(y[~invalid_indices], dim=0)
                y[invalid_indices] = impute_value
                

        # Remove repeating data points 
        # Determine if the dataset is not too large
        if len(y) > self.max_training_dataset_size:
            x, y = subsample_training_data(x, y, self.max_training_dataset_size)

        self.x = x.to(dtype=self.dtype, device=self.device)
        self._y = y.to(dtype=self.dtype, device=self.device)

        self.fit_y = self.y_to_fit_y(y=self._y)
        self.fit_y = self.fit_y.reshape(*(self.batch_shape + (-1, 1)))

        if self.binary_classification:
            self.gp = GPyTorchClassificationModel(self.x, self.fit_y, self.kernel, self.likelihood).to(self.x)
        else:
            self.gp = GPyTorchGPModel(self.x, self.fit_y, self.kernel, self.likelihood).to(self.x)

        # Attempt to make a local copy of the class to possibly later recover from a ValueError Exception
        self_copy = None
        try:
            self_copy = copy.deepcopy(self)
        except:
            pass

        self.gp.train()
        self.likelihood.train()
        if self.binary_classification:
            mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp, self._y.numel())
            
        else:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

        if self.optimizer == 'adam':
            opt = torch.optim.Adam([{'params': mll.parameters()}], lr=self.lr)
        else:
            raise NotImplementedError(f'Optimiser {self.optimizer} was not implemented.')
        losses = []
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):

            for epoch in range(self.num_epochs):
                def closure(append_loss=True):
                    opt.zero_grad()
                    dist = self.gp(self.x)
                    if self.binary_classification:
                        loss = -1 * mll(dist, self._y.squeeze())
    
                    else:
                        loss = -1 * mll(dist, self.fit_y.squeeze()).sum()
                        
                    loss.backward()
                    if append_loss:
                        losses.append(loss.item())
        
                    return loss

                try:
                    opt.step(closure)
                except (NotPSDError, NanError):
                    warnings.warn('\n\nMatrix is singular during GP training. Resetting GP parameters to ' + \
                                  'what they were before gp training and moving to the next BO iteration. Possible ' + \
                                  ' solutions: \n\n - Consider changing to double precision \n - Decrease the number of ' + \
                                  'GP training epochs per BO iteration or the GP learning rate to avoid overfitting.\n')

                    self.__class__ = copy.deepcopy(self_copy.__class__)
                    self.__dict__ = copy.deepcopy(self_copy.__dict__)

                    break
                except ValueError as e:
                    kernel_any_nan = False
                    mean_any_nan = False
                    likelihood_any_nan = False
                    for _, param in self.gp.mean.named_parameters():
                        if torch.isnan(param).any():
                            mean_any_nan = True
                            break
                    for _, param in self.gp.kernel.named_parameters():
                        if torch.isnan(param).any():
                            kernel_any_nan = True
                            break
                    for _, param in self.likelihood.named_parameters():
                        if torch.isnan(param).any():
                            likelihood_any_nan = True
                            break

                    if (mean_any_nan or kernel_any_nan or likelihood_any_nan) and (self_copy is not None):
                        warnings.warn(f'\n\nSome parameters (mean: {mean_any_nan} | kernel: {kernel_any_nan} | '
                                      f'likelihood: {likelihood_any_nan}) became NaN. Resetting GP parameters to ' + \
                                      'what they were before gp training and moving to the next BO iteration.\n\n')
                        self.__class__ = copy.deepcopy(self_copy.__class__)
                        self.__dict__ = copy.deepcopy(self_copy.__dict__)
                        break
                    else:
                        raise e

                if self.verbose and ((epoch + 1) % self.print_every == 0 or epoch == 0):
                    print('After %d epochs, loss = %g' % (epoch + 1, closure(append_loss=False).item()), flush=True)

        self.gp.eval()
        self.likelihood.eval()
        
        return losses


    def predict(self, x: torch.Tensor, **kwargs) -> (torch.Tensor, torch.Tensor):
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
            x = x.to(device=self.device, dtype=self.dtype)
            #try:
            pred = self.gp(x)
            #except:
            #    breakpoint()
            if self.pred_likelihood:
                pred = self.likelihood(pred)
            mu_ = pred.mean.reshape(-1, self.num_out)
            var_ = pred.variance.reshape(-1, self.num_out)

        return mu_, var_.clamp(min=torch.finfo(var_.dtype).eps)

    def sample_y(self, x: torch.FloatTensor, n_samples=1) -> torch.FloatTensor:
        """
        Should return (n_samples, Xc.shape[0], self.num_out)
        """
        x = x.to(dtype=self.dtype, device=self.device)
        with gpytorch.settings.debug(False):
            pred = self.gp(x)
            if self.pred_likelihood:
                pred = self.likelihood(pred)
            sample = pred.rsample(torch.Size((n_samples,))).view(n_samples, x.shape[0], 1)
            sample = self.y_std * sample + self.y_mean
            return sample

    @property
    def noise(self) -> torch.Tensor:
        if self.num_out == 1:
            return (self.gp.likelihood.noise * self.y_std ** 2).view(self.num_out).detach()
        else:
            return (self.gp.likelihood.task_noises * self.y_std ** 2).view(self.num_out).detach()

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Function used to move model to target device and dtype. Note that this should also change self.dtype and
        self.device

        Args:
            device: target device
            dtype: target dtype

        Returns:
            self
        """
        self.device = device
        self.dtype = dtype
        if self.gp is not None:
            self.gp.to(device=device, dtype=dtype)
        return super().to(device=device, dtype=dtype)


class GPyTorchGPModel(ExactGP):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, kernel: Kernel, likelihood: GaussianLikelihood):
        super(GPyTorchGPModel, self).__init__(x, y.squeeze(), likelihood=likelihood)
        self.mean = ConstantMean(batch_shape=kernel.batch_shape)
        self.kernel = kernel
        
    def forward(self, x: torch.FloatTensor) -> MultivariateNormal:
        mean = self.mean(x)
        cov = self.kernel(x)
        return MultivariateNormal(mean, cov)



class GPyTorchClassificationModel(ApproximateGP):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, kernel: Kernel, likelihood: BernoulliLikelihood):
        variational_distribution = CholeskyVariationalDistribution(x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, x, variational_distribution, learn_inducing_locations=False
        )
        super(GPyTorchClassificationModel, self).__init__(variational_strategy)
        self.mean_module = ConstantMean(batch_shape=kernel.batch_shape)
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        return latent_pred
    
    def set_train_data(self, inputs=None, targets=None, strict=True):
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            if strict:
                for input_, t_input in length_safe_zip(inputs, self.train_inputs or (None,)):
                    for attr in {"shape", "dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                            msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                            raise RuntimeError(msg)
            self.train_inputs = inputs
        if targets is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                        raise RuntimeError(msg)
            self.train_targets = targets
        self.prediction_strategy = None
