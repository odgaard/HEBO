import torch
from mcbo import task_factory
from mcbo.optimizers.bo_builder import BoBuilder

if __name__ == '__main__':
    task = task_factory(task_name='SpMM')
    search_space = task.get_search_space()
    bo_builder = BoBuilder(
        model_id='gp_ssk', acq_opt_id='mp', acq_func_id='ei', tr_id=None, init_sampling_strategy="uniform"
    )

    optimizer = bo_builder.build_bo(search_space=search_space, n_init=20, device=torch.device("cuda"))

    for i in range(30):
        x = optimizer.suggest(1)
        y = task(x)
        optimizer.observe(x, y)
        print(f'Iteration {i + 1:3d}/{100:3d} - SpMM = {optimizer.best_y:.3f}')

    # Access history of suggested points and black-box values
    all_x = search_space.inverse_transform(optimizer.data_buffer.x)
    all_y = optimizer.data_buffer.y.cpu().numpy()
