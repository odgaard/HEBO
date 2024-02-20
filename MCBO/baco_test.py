import torch
from mcbo import task_factory
from mcbo.optimizers.bo_builder import BoBuilder

if __name__ == '__main__':
    task_name = 'ttv'
    n_initial_samples = 20
    n_samples = 30

    task = task_factory(task_name=task_name)
    search_space = task.get_search_space()
    bo_builder = BoBuilder(
        model_id='gp_to', acq_opt_id='is', acq_func_id='ei', tr_id='basic', init_sampling_strategy="sobol_scramble"
    )

    optimizer = bo_builder.build_bo(search_space=search_space, n_init=n_initial_samples, device=torch.device("cuda"))

    for i in range(n_samples):
        x = optimizer.suggest(1)
        y = task(x)
        optimizer.observe(x, y)
        print(f'Iteration {i + 1:3d}/{n_samples:3d} - {task_name} = {optimizer.best_y:.3f}')

    # Access history of suggested points and black-box values
    all_x = search_space.inverse_transform(optimizer.data_buffer.x)
    all_y = optimizer.data_buffer.y.cpu().numpy()
