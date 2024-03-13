from typing import Any, List, Optional, Callable
import time
import re

import numpy as np
import pandas as pd
import bacobench as bb

from mcbo.tasks import TaskBase
from interopt.parameter import ParamType, Constraint

taco_tasks = ['spmm', 'spmv', 'sddmm', 'ttv', 'mttkrp']
rise_tasks = ['asum', 'harris', 'kmeans', 'mm', 'scal', 'stencil']

class BacoTaskBase(TaskBase):
    def __init__(self, benchmark_name: str, dataset_id: str = '10k',
                 objectives: List[str] = ['compute_time', 'energy'], enable_permutation=True):
        super(BacoTaskBase, self).__init__()
        self.benchmark_name = benchmark_name.lower()
        self.dataset_id = dataset_id
        self.objectives = objectives
        self.current_time = time.time()
        self.constraints = [1]
        enable_model = self.benchmark_name in taco_tasks
        #enable_model = False
        enable_tabular = True
        self.enable_permutation = enable_permutation
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
        if self.enable_permutation:
            for k, v in d.copy().items():
                if 'permutation' in k:
                    t.append(int(v))
                    del d[k]

            d['permutation'] = str(tuple(t))

        query_result = self.bench.query(d)
        compute_time = query_result['compute_time']
        valid = 1.0
        if compute_time == 0.0:
            valid = 0.0
            for objective in self.objectives:
                if objective in query_result:
                    query_result[objective] = np.nan
        results = []
        for objective in self.objectives:
            if objective in query_result:
                results.append(query_result[objective])
        results.append(valid)

        output_results = False
        if output_results:
            with open(f'results-{self.current_time}.txt', 'a', encoding='utf-8') as f:
                f.write(f"Query: {d}\n")
                f.write(f"Result: {results}\n")

        results[-1] = np.random.choice(2)
        results = np.array(results)
        if results[-1] == 0:
            results[:-1] = np.nan
        print("result: ".upper(), results)
        return results

    def objective_count(self) -> int:
        #return len(self.objectives)
        return len(self.objectives) + len(self.constraints)

    @property
    def name(self) -> str:
        return self.benchmark_name

    def map_type_to_mcbo_type(self, type_enum):
        return {
            ParamType.REAL: 'real',
            ParamType.INTEGER_EXP: 'int_exponent',
            ParamType.INTEGER: 'int',
            ParamType.CATEGORICAL: 'nominal',
            ParamType.PERMUTATION: 'permutation' if self.enable_permutation else 'nominal'
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
                d['length'] = param.length
                # TODO: Temporary categorical implementation of PERMUTATION
                #l = list(permutations(range(param.length)))
                #d['categories'] = [str(perm) for perm in l]
            mcbo_params.append(d)
        return mcbo_params

    def input_constraints(self) -> Optional[list[Callable[[dict], bool]]]:
        variable_names = [param.name for param in self.bench.definition.search_space.params]
        lambda_constraints = []
        for constraint in self.bench.definition.search_space.constraints:
            dict_string: str = Constraint._as_dict_string(constraint.constraint, variable_names)

            # TODO: Temporary fix for permutation constraints
            if self.enable_permutation:
                print(f"Enable: {dict_string}")
                # Replace permutation[i] with permutation_i
                dict_string = re.sub(r"x\['permutation'\]\[(\d+)\]", r"x['permutation_\1']", dict_string)

                # Pandas vectorized operations
                dict_string = dict_string.replace(' or ', ' | ')
                dict_string = dict_string.replace(' and ', ' & ')

            else:
                dict_string = dict_string.replace("x['permutation']", "eval(x['permutation'])")
            print(dict_string)

            lambda_constraints.append(Constraint._string_as_lambda(dict_string))

        return lambda_constraints
