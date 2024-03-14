from typing import Optional

import numpy as np
import torch
import fire
from mcbo import task_factory

from mcbo.optimizers.bo_builder import BoBuilder
from mcbo.optimizers.bo_builder import BO_ALGOS

from mcbo.utils.experiment_utils import get_task_from_id
from eval_tools.recorded_trajectory import RecordedTrajectory
from eval_tools.method_registry import METHOD_REGISTRY



def main(
        task_name: str = 'spmm', 
        method_name: Optional[str] = None, 
        use_perms: bool = True, 
        model_id: str = 'gp_hed', 
        use_tr: bool = False, 
        perm_kernel: str = "mallows",
        seed: int = 999,
        experiment_path: str = "bacobench_test",
    ):
    if method_name == "random":
        n_initial_samples = 200
        n_samples = 200
    else:
        n_initial_samples = 10
        n_samples = 200
    
    enable_permutation = use_perms
    
    if method_name is not None:
        method = METHOD_REGISTRY[method_name]
        use_perms = method["use_perms"]
        model_id = method["model_id"]
        use_tr = method["use_tr"]
        perm_kernel = method["perm_kernel"]

    else:
        method_name = "noname"

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    task = task_factory(task_name=task_name, enable_permutation=enable_permutation, objectives=["compute_time", "energy"])
    search_space = task.get_search_space()
    tracked_task = RecordedTrajectory(task, task_name, method_name, seed, experiment_path)
    
    tr_id = "basic" if use_tr else None
    model_kwargs=dict(perm_kernel_name=perm_kernel)
    acq_func_kwargs = dict(num_constr=1)
    bo_builder = BoBuilder(
        model_id=model_id,
        acq_opt_id='is',
        acq_func_id='cei',
        tr_id=tr_id,
        init_sampling_strategy="sobol_scramble",
        model_kwargs=model_kwargs,
        acq_func_kwargs=acq_func_kwargs
    )
    # bo_builder = BO_ALGOS["BOSS"]
    device = torch.device("cpu")
    optimizer = bo_builder.build_bo(
        search_space=search_space,
        n_init=n_initial_samples,
        device=device,
        obj_dims=[0, 1],
        out_constr_dims=[2],
        input_constraints=task.input_constraints
    )

    for i in range(n_samples):
        x = optimizer.suggest(1)
        y = tracked_task(x)

        optimizer.observe(x, y)
        
        print(f'Iteration {i}/{n_samples} - {task_name} = {optimizer.best_y}')

    # Access history of suggested points and black-box values
    all_x = search_space.inverse_transform(optimizer.data_buffer.x)
    all_y = optimizer.data_buffer.y.cpu().numpy()


if __name__ == '__main__':
    fire.Fire(main)

