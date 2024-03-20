import math
from os.path import join, dirname, abspath
from typing import Optional, List, Union
from glob import glob

import numpy as np
from scipy.stats import sem
import pandas as pd
from fire import Fire
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV


plt.rcParams['font.family'] = 'serif'

HYPERVOLUME_REFERENCES = {
    "spmm": {
        "energy": None,
        "compute_time": None,
    },
    "spmv": {
        "energy": 0.25,
        "compute_time": 2,
    },
    "sddmm": {
        "energy": None,
        "compute_time": None,
    },
    "asum": {
        "energy": None,
        "compute_time": None,
    },
}
COLORS = {
    "baco": "blue",
    "casmo_perm": "red",
    "bops": "green",
}

NICE_BENCH_NAMES = {
    "spmm": "spmm".upper(),
    "spmv": "spmv".upper(),
    "sddmm": "sddmm".upper(),
    "asum": "asum".upper(),
}
NICE_METHOD_NAMES = {
    "baco": "BACO",
}
MAX_PLOTS_PER_ROW = 4
N_ERROR = 1

def _typecheck_list(inp: Union[str, list]):
    if isinstance(inp, str):
        return [inp]
    return inp

def filter_benchmark_or_method(paths: List[str], to_keep: Union[List, str]):
    paths_to_keep = []
    if to_keep is not None:
        to_keep = _typecheck_list(to_keep)

        for p in paths:
            for tk in to_keep:
                if tk in p:
                    paths_to_keep.append(p)
                    break
    return paths_to_keep

def _get_run_seed(result_path: str):
    return int(result_path.split("_run_")[-1].replace(".csv", ""))


def _compute_hypervolume(results: np.array, hv_ref: np.array) -> np.array:
    hvi_compute = HV(np.zeros(len(hv_ref)))
    hv_ref = hv_ref.reshape(1, -1, 1)
    relative_results = np.nan_to_num(results - hv_ref, 0)
    all_hvi = np.zeros_like(results[:, 0, :])
    for res_idx, result in enumerate(relative_results):
        for iter_ in range(len(result.T)):
            hvi = hvi_compute(result.T[:iter_])
            all_hvi[res_idx, iter_] = hvi
            
    return all_hvi        


@Fire
def plot(
        path: str, 
        benchmarks: Optional[Union[str, list]] = None, 
        methods: Optional[Union[str, list]] = None, 
        metrics: Optional[Union[str, list]] = ["energy", "compute_time"],
        hv_ref: Optional[List] = None, 
        use_length: str = "min",
        plot_name: str = "bacobench_results"
    ) -> None:
    metrics = _typecheck_list(metrics)
    all_results = {}
    bench_paths = glob(join(path, "*"))
    if benchmarks is not None:
        bench_paths = filter_benchmark_or_method(bench_paths, benchmarks)
    
    for bench in bench_paths:
        bench_name = bench.split("/")[-1]
        #bench_name = NICE_BENCH_NAMES.get(bench_name, bench_name)
        all_results[bench_name] = {}
        method_paths = glob(join(bench, "*"))
        if methods is not None:
            method_paths = filter_benchmark_or_method(method_paths, methods)

        for method in method_paths:
            method_name =  method.split("/")[-1]
            #method_name = NICE_METHOD_NAMES.get(method_name, method_name)
            csvs = glob(join(method, "*.csv"))
            csv_lengths = [len(pd.read_csv(csv)) for csv in csvs]
            result_length = eval(use_length)(csv_lengths)
            result_array = np.zeros((len(csvs), len(metrics), result_length)) * np.nan
            
            for r_idx, csv in enumerate(sorted(csvs, key=lambda x: _get_run_seed(x))):
                df = pd.read_csv(csv).iloc[:result_length, :]
                for m_idx, metric in enumerate(metrics):
                    res = df.loc[:, metric]
                    result_array[r_idx, m_idx, :len(res)] = res
            
            all_results[bench_name][method_name] = result_array
    
    assert len(bench_paths) > 0, "All benchmarks are filtered out."
    nrows, ncols = math.ceil(len(bench_paths) / MAX_PLOTS_PER_ROW), min(len(bench_paths), MAX_PLOTS_PER_ROW) 
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5))
    axes = np.array(axes).reshape(nrows, -1)
    for bench_idx, b_name in enumerate(all_results.keys()):
        ax = axes[bench_idx // MAX_PLOTS_PER_ROW, bench_idx % MAX_PLOTS_PER_ROW]
        methods_for_bench = all_results[b_name]
        for m_name, m_result in methods_for_bench.items():
            refs = np.array([HYPERVOLUME_REFERENCES[b_name][m] for m in metrics])
            hv = _compute_hypervolume(m_result, refs)
            mean, std = hv.mean(axis=0), sem(hv, axis=0)
            X_plot = np.arange(0, hv.shape[-1])
            ax.plot(X_plot, mean, label=NICE_METHOD_NAMES.get(m_name, m_name), color=COLORS[m_name])
            ax.fill_between(X_plot, mean - N_ERROR * std, mean + N_ERROR * std, alpha=0.15, color=COLORS[m_name])
            ax.plot(X_plot, mean + N_ERROR * std, alpha=0.3, color=COLORS[m_name])
            ax.plot(X_plot, mean - N_ERROR * std, alpha=0.3, color=COLORS[m_name])
    
        ax.legend(fontsize=16)
        ax.set_ylabel("Hypervolume", fontsize=18)
        ax.set_xlabel("Iteration", fontsize=18)
        ax.tick_params(axis='both', labelsize=15)
        ax.set_title(NICE_BENCH_NAMES.get(b_name, b_name), fontsize=20)
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_name}.pdf")
    plt.show()

    # RESULTS HAVE NOW BEEN GATHERED
                    
#if __name__ == "__main__":
#    plot()