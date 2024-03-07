from itertools import permutations
from typing import Any, Dict, List, Optional, Callable
import time

import numpy as np
import pandas as pd

from interopt.parameter import ParamType, Constraint
import bacobench as bb

from mcbo.tasks import TaskBase


taco_tasks = ['spmm', 'spmv', 'sddmm', 'ttv', 'mttkrp']
rise_tasks = ['asum', 'harris', 'kmeans', 'mm', 'scal', 'stencil']

class BacoTaskBase(TaskBase):
    def __init__(self, benchmark_name: str, dataset_id: str = '10k',
                 objectives: List[str] = ['compute_time', 'energy']):
        super(BacoTaskBase, self).__init__()
        self.benchmark_name = benchmark_name.lower()
        self.dataset_id = dataset_id
        self.objectives = objectives
        self.current_time = time.time()
        enable_model = self.benchmark_name in taco_tasks
        enable_tabular = True
        port = 50050
        interopt_server = 'localhost'
        self.bench = bb.benchmark(
            self.benchmark_name, dataset=self.dataset_id,
            enable_tabular=enable_tabular, enable_model=enable_model,
            objectives=self.objectives, port=port,
            server_addresses=[interopt_server]
        )

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

        query_result = self.bench.query(d)
        r = query_result['compute_time']
        if r == 0.0:
            r = 1e+6
        with open(f'results-{self.current_time}.txt', 'a', encoding='utf-8') as f:
            f.write(f"Query: {d}\n")
            f.write(f"Result: {r}\n")

        return np.array(r)

    def objective_count(self) -> int:
        #return len(self.objectives)
        return 1

    @property
    def name(self) -> str:
        return self.benchmark_name

    def map_type_to_mcbo_type(self, type_enum):
        return {
            ParamType.REAL: 'real',
            ParamType.INTEGER_EXP: 'int_exponent',
            ParamType.INTEGER: 'int',
            ParamType.CATEGORICAL: 'nominal',
            #ParamType.PERMUTATION: 'permutation'
            ParamType.PERMUTATION: 'nominal'
        }[type_enum]

    def get_search_space_params(self) -> list[dict[str, Any]]:
        mcbo_params = []
        for param in self.bench.definition.search_space.params:
            type_enum = param.param_type_enum
            d = {
                'name': param.name,
                'type': self.map_type_to_mcbo_type(type_enum),
                'default': param.default
            }
            if type_enum in (ParamType.INTEGER, ParamType.REAL, ParamType.INTEGER_EXP):
                d['lb'] = param.lower
                d['ub'] = param.upper
                if type_enum == ParamType.INTEGER_EXP:
                    d['base'] = param.base
            if type_enum == ParamType.CATEGORICAL:
                d['categories'] = param.categories
            if type_enum == ParamType.PERMUTATION:
                #d['length'] = param.length
                # TODO: Temporary categorical implementation of PERMUTATION
                l = list(permutations(range(param.length)))
                d['categories'] = [str(perm) for perm in l]
            mcbo_params.append(d)
        return mcbo_params

    def input_constraints(self) -> Optional[list[Callable[[dict], bool]]]:
        variable_names = [param.name for param in self.bench.definition.search_space.params]
        lambda_constraints = []
        for constraint in self.bench.definition.search_space.constraints:
            dict_string: str = Constraint._as_dict_string(constraint.constraint, variable_names)

            # TODO: Temporary fix for permutation constraints
            dict_string = dict_string.replace("x['permutation']", "eval(x['permutation'])")

            lambda_constraints.append(Constraint._string_as_lambda(dict_string))

        return lambda_constraints
