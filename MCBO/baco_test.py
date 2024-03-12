import torch
import fire
from mcbo import task_factory

from mcbo.optimizers.bo_builder import BoBuilder
from mcbo.optimizers.bo_builder import BO_ALGOS

from mcbo.utils.experiment_utils import get_task_from_id


def main(task_name: str = 'spmm', use_perms: bool = True, model_id: str = 'gp_to', use_tr: bool = True, perm_kernel: str = "kendalltau"):
    n_initial_samples = 10
    n_samples = 200
    use_permutations = True
    print(task_name)
    #task = get_task_from_id(task_name)
    task = task_factory(task_name=task_name, use_permutations=use_permutations, objectives=["compute_time"])
    search_space = task.get_search_space()
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
        #input_constraints=task.input_constraints()
    )

    for i in range(n_samples):
        x = optimizer.suggest(1)
        y = task(x)

        optimizer.observe(x, y)

        print(f'Iteration {i}/{n_samples} - {task_name} = {optimizer.best_y.round(decimals=2)}')

    # Access history of suggested points and black-box values
    all_x = search_space.inverse_transform(optimizer.data_buffer.x)
    all_y = optimizer.data_buffer.y.cpu().numpy()


if __name__ == '__main__':
    fire.Fire(main)

