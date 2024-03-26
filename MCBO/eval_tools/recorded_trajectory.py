from typing import Union, List
import os
from os.path import dirname

from torch import Tensor
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class RecordedTrajectory:

    def __init__(
        self,
        function: callable,
        function_name: str,
        method_name: str,
        seed: int = 919,
        experiment_path: str = "bacobench_test",

    ) -> None:
        self.function = function
        self.output_cols = function.objectives + ["Feasibility"]
        self.input_cols = function.get_search_space().param_names
        empty_data = {
            col: [] for col in
                (self.input_cols + self.output_cols)
        }
        self.df = pd.DataFrame(empty_data)
        self.function_name = function_name
        self.save_path  = f'{experiment_path}/{function_name}/{method_name}/{method_name}_run_{seed}.csv'

    def _process_permutations(self, X: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        perm_cols = np.array(["permutation" in x_col for x_col in X.columns])
        if any(perm_cols):
            save_data = X.loc[:, ~perm_cols]
            perms = X.loc[:, ~perm_cols].to_numpy()
            save_data.loc[:, "permutation"] = [str(tuple(x)) for x in X.loc[:, perm_cols].to_numpy()]
        else:
            save_data = X

        return save_data

    def __call__(self, X: List[pd.Series]) -> np.array:
        save_data = self._process_permutations(X)
        res = self.function.evaluate(X)
        print(res)

        print(self.output_cols)
        for res_col, name in zip(res.T, self.output_cols):
            save_data.loc[:, name] = res_col

        self.save(save_data)
        return res

    def save(self, save_data: Union[pd.DataFrame, pd.Series]) -> None:
        self.df = pd.concat((self.df, save_data))
        os.makedirs(dirname(self.save_path), exist_ok=True)
        self.df.to_csv(self.save_path)
