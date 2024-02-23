from itertools import permutations
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import pandas as pd

import bacobench as bb

from mcbo.tasks import TaskBase


taco_tasks = ['spmm', 'spmv', 'sddmm', 'ttv', 'mttkrp']
rise_tasks = ['asum']

class BacoTaskBase(TaskBase):
    def __init__(self, benchmark_name: str, dataset_id: str = '10k', objectives: List[str] = ['compute_time', 'energy']):
        super(BacoTaskBase, self).__init__()
        self.benchmark_name = benchmark_name.lower()
        self.dataset_id = dataset_id
        self.objectives = objectives
        self.tabular = True if benchmark_name in taco_tasks else False
        self.tabular = False
        self.bench = bb.benchmark(benchmark_name, dataset=self.dataset_id, tabular=self.tabular, objectives=self.objectives)

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        results = np.zeros((len(x), self.objective_count()))
        for i in range(len(x)):
            results[i] = self.evaluate_single_point(x.iloc[i])
        return results

    def evaluate_single_point(self, x: pd.Series) -> np.ndarray:
        d = x.to_dict().copy()
        t = []
        permutation_is_permutation_variable = False
        if permutation_is_permutation_variable:
            for k, v in d.copy().items():
                if 'permutation' in k:
                    t.append(int(v))
                    del d[k]

            d['permutation'] = str(tuple(t))
        #for k, v in d.items():
        #    if self.benchmark_name in rise_tasks:
        #        d[k] = 2**v

        print("Evaluate:", d)

        query_result = self.bench.query([d])[0]
        r = query_result['compute_time']
        if r == 0.0:
            r = 1e+6
        print(r)

        return np.array(r)

    def objective_count(self) -> int:
        #return len(self.objectives)
        return 1

    @property
    def name(self) -> str:
        return self.benchmark_name

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def input_constraints(self) -> Optional[List[Callable[[Dict], bool]]]:
        # By default, there are no constraints. Override in subclass if needed.
        return None

    def generate_valid_permutations(self) -> List[tuple]:
        # Implement in subclasses if needed
        return []


class AsumTask(BacoTaskBase):
    def __init__(self):
        super(AsumTask, self).__init__('Asum')

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        """
        return [
            {'name': 'tuned_sp0', 'type': 'int', 'lb': 1, 'ub': 16777216},
            {'name': 'tuned_gs0', 'type': 'int', 'lb': 1, 'ub': 1024},
            {'name': 'tuned_stride', 'type': 'int', 'lb': 1, 'ub': 16777216},
            {'name': 'tuned_sp1', 'type': 'int', 'lb': 1, 'ub': 16777216},
            {'name': 'tuned_ls0', 'type': 'int', 'lb': 1, 'ub': 1024},
        ]
        """
        return [
            {'name': 'tuned_sp0', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 16777216},
            {'name': 'tuned_gs0', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 1024},
            {'name': 'tuned_stride', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 16777216},
            {'name': 'tuned_sp1', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 16777216},
            {'name': 'tuned_ls0', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 1024},
        ]
        """
        return [
            {'name': 'tuned_sp0', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 24},
            {'name': 'tuned_gs0', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 10},
            {'name': 'tuned_stride', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 24},
            {'name': 'tuned_sp1', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 24},
            {'name': 'tuned_ls0', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 10},
        ]
        """

    def input_constraints(self) -> Optional[List[Callable[[Dict], bool]]]:
        return [
        #    lambda x: (x['tuned_sp0'] & (x['tuned_sp0'] - 1)) == 0,
        #    lambda x: (x['tuned_gs0'] & (x['tuned_gs0'] - 1)) == 0,
        #    lambda x: (x['tuned_stride'] & (x['tuned_stride'] - 1)) == 0,
        #    lambda x: (x['tuned_sp1'] & (x['tuned_sp1'] - 1)) == 0,
        #    lambda x: (x['tuned_ls0'] & (x['tuned_ls0'] - 1)) == 0,
            lambda x: x['tuned_sp0'] % x['tuned_sp1'] == 0 and (x['tuned_sp0']*(1/x['tuned_sp1'])) >= (1) and x['tuned_sp0'] >= x['tuned_sp1'] * x['tuned_stride'] and x['tuned_gs0'] % x['tuned_ls0'] == 0
        ]
        """
        return [
            lambda x: x['tuned_sp0'] >= 0,
            lambda x: 2**x['tuned_sp0'] % 2**x['tuned_sp1'] == 0,
            lambda x: (2**x['tuned_sp0']*(1/(2**x['tuned_sp1']))) >= (1),
            lambda x: 2**x['tuned_sp0'] >= 2**x['tuned_sp1'] * 2**x['tuned_stride'],
            lambda x: 2**x['tuned_gs0'] >= 0,
            lambda x: 2**x['tuned_gs0'] % 2**x['tuned_ls0'] == 0
        ]
        """

class TTVTask(BacoTaskBase):
    def __init__(self):
        super(TTVTask, self).__init__('TTV')

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return [
            {'name': 'chunk_size_i', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 8},
            {'name': 'chunk_size_fpos', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 8},
            {'name': 'chunk_size_k', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 8},
            {'name': 'omp_chunk_size', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 5},
            {'name': 'omp_num_threads', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 4, 'log_scale': True},
            {'name': 'omp_scheduling_type', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_monotonic', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_dynamic', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_proc_bind', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'permutation', 'type': 'nominal', 'categories': self.generate_valid_permutations()},
        ]

    def generate_valid_permutations(self) -> List[tuple]:
        # TTV-specific permutation logic
        valid_conditions = [
            lambda p: p[0] == 4,
            lambda p: (p[0] == 0 and p[1] == 1),
            lambda p: (p[0] == 1 and p[1] == 0),
            lambda p: (p[0] == 0 and p[1] == 4),
            lambda p: (p[0] == 1 and p[1] == 4)
        ]
        all_perms = list(permutations(range(5)))
        valid_perms = [perm for perm in all_perms if any(cond(perm) for cond in valid_conditions)]
        return [str(perm) for perm in valid_perms]


class SpMMTask(BacoTaskBase):
    def __init__(self):
        super(SpMMTask, self).__init__('SpMM')

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return [
            {'name': 'chunk_size', 'type': 'int_exponent', 'base': 2, 'lb': 2, 'ub': 64},
            {'name': 'unroll_factor', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 8},
            {'name': 'omp_chunk_size', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 64},
            {'name': 'omp_num_threads', 'type': 'int', 'lb': 1, 'ub': 12},
            {'name': 'omp_scheduling_type', 'type': 'nominal', 'categories': [0, 1, 2]},
            {'name': 'omp_monotonic', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_dynamic', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_proc_bind', 'type': 'nominal', 'categories': [0, 1, 2]},
            {'name': 'permutation', 'type': 'nominal', 'categories': self.generate_valid_permutations()},
            #{'name': 'permutation', 'type': 'permutation', 'length': 5},
        ]

    def generate_valid_permutations(self) -> List[tuple]:
        all_perms = list(permutations(range(5)))
        valid_perms = [perm for perm in all_perms if perm[0] < perm[3] and
                       perm[1] < perm[3] and
                       perm[0] < perm[2] and
                       perm[1] < perm[2] and
                       (perm[3] < perm[2] or perm[3] < perm[4])]
        return [str(perm) for perm in valid_perms]

    def input_constraints(self) -> Optional[List[Callable[[Dict], bool]]]:
        return [
                lambda x: x['omp_num_threads'] % 2 == 0
        ]
        #return [
      #      lambda x: x['permutation'][0] < x['permutation'][3] and
      #      x['permutation'][1] < x['permutation'][3] and
      #      x['permutation'][0] < x['permutation'][2] and
      #      x['permutation'][1] < x['permutation'][2] and
      #      (x['permutation'][3] < x['permutation'][2] or x['permutation'][3] < x['permutation'][4])
        #]

class SpMVTask(BacoTaskBase):
    def __init__(self):
        super(SpMVTask, self).__init__('SpMV')

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return [
            {'name': 'chunk_size', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 10},
            {'name': 'chunk_size2', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 5},
            {'name': 'chunk_size3', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 5},
            {'name': 'omp_chunk_size', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 8},
            {'name': 'omp_num_threads', 'type': 'int', 'lb': 4, 'ub': 20},
            {'name': 'omp_scheduling_type', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_monotonic', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_dynamic', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_proc_bind', 'type': 'nominal', 'categories': [0, 1, 2]},
            {'name': 'permutation', 'type': 'nominal', 'categories': self.generate_valid_permutations()},
        ]

    def generate_valid_permutations(self) -> List[tuple]:
        # Adjusted to new constraints: permutation_4 must be 4
        valid_perms = [perm for perm in permutations(range(5)) if perm[4] == 4]
        return [str(perm) for perm in valid_perms]

class SDDMMTask(BacoTaskBase):
    def __init__(self):
        super(SDDMMTask, self).__init__('SDDMM')

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return [
            {'name': 'chunk_size', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 8},
            {'name': 'unroll_factor', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 6},
            {'name': 'omp_chunk_size', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 8},
            {'name': 'omp_num_threads', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 4, 'log_scale': True},
            {'name': 'omp_scheduling_type', 'type': 'nominal', 'categories': [0, 1, 2]},
            {'name': 'omp_monotonic', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_dynamic', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_proc_bind', 'type': 'nominal', 'categories': [0, 1, 2]},
            {'name': 'permutation', 'type': 'nominal', 'categories': self.generate_valid_permutations()},
        ]

    def input_constraints(self) -> Optional[List[Callable[[Dict], bool]]]:
        return [
            lambda x: x['unroll_factor'] < x['chunk_size'] and x['unroll_factor'] % 2 == 0,
            lambda x: (x['omp_chunk_size'] % 2 == 0) or (x['omp_chunk_size'] == 1),
        ]

    def generate_valid_permutations(self) -> List[tuple]:
        # Implement the logic for generating valid permutations based on the constraint provided
        valid_perms = []
        all_perms = list(permutations(range(5)))
        for perm in all_perms:
            if (((perm[2] < perm[4]) and (perm[0] < perm[2]) and (perm[1] < perm[2])) or
               ((perm[4] < perm[2]) and (perm[0] < perm[4]) and (perm[1] < perm[4]))):
                valid_perms.append(perm)
        return [str(perm) for perm in valid_perms]

class MTTKRPTask(BacoTaskBase):
    def __init__(self):
        super(MTTKRPTask, self).__init__('MTTKRP')

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return [
            {'name': 'chunk_size', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 10},
            {'name': 'unroll_factor', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 10},
            {'name': 'omp_chunk_size', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 8},
            {'name': 'omp_num_threads', 'type': 'int_exponent', 'base': 2, 'lb': 1, 'ub': 6},
            {'name': 'omp_scheduling_type', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_monotonic', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_dynamic', 'type': 'nominal', 'categories': [0, 1]},
            {'name': 'omp_proc_bind', 'type': 'nominal', 'categories': [0, 1, 2]},
            {'name': 'permutation', 'type': 'nominal', 'categories': self.generate_valid_permutations()},
        ]

    def input_constraints(self) -> Optional[List[Callable[[Dict], bool]]]:
        return [
            lambda x: x['unroll_factor'] < x['chunk_size'] and x['unroll_factor'] % 2 == 0,
        ]
