from itertools import chain, cycle, repeat
from pathlib import Path
import pandas as pd
from functools import partial
from time import perf_counter
import numpy as np
import multiprocessing as mp
from psutil import cpu_count
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

import gillespie as gill
import gillespie_newranges as gill_new
from gillespie import (
    make_matrices_for_ssa,
    _rescale,
)


def get_uniform_samples(n_samples, seed, param_ranges):
    n_params = len(param_ranges)
    rg = np.random.default_rng(seed)
    lh_sampler = LatinHypercube(n_params, seed=rg)
    uniform_samples = lh_sampler.random(n_samples)
    return uniform_samples


def run_gillespie_random_params(idx_params_ssa, pop0):
    idx, (params, ssa) = idx_params_ssa
    start = perf_counter()
    ssa.run_with_params(pop0, params)
    end = perf_counter()
    return idx, end - start


def main(
    save_dir: Path,
    n_samples,
    n_threads=None,
    seed=2023,
    save: bool = False,
):
    # # Activator-inhibitor circuit
    # # 2 components A and B. AAa_ABa_BAi
    # nc = 2
    # Am, Rm, U = make_matrices_for_ssa(
    #     nc, activations=[[0, 0], [0, 1]], inhibitions=[[1, 0]]
    # )
    # ssa_kw = dict(
    #     seed=0,
    #     n_species=nc,
    #     activation_mtx=Am,
    #     inhibition_mtx=Rm,
    #     update_mtx=U,
    #     dt=20.0,
    #     nt=100,
    #     mean_mRNA_init=10.0,
    #     DEFAULT_PARAMS=gill.DEFAULT_PARAMS,
    # )

    # Activator-inhibitor-quencher circuit
    # 3 components A and B. AAa_ABa_BAi_CBi
    nc = 3
    Am, Rm, U = make_matrices_for_ssa(
        nc, activations=[[0, 0], [0, 1]], inhibitions=[[1, 0], [2, 1]]
    )
    ssa_kw = dict(
        seed=0,
        n_species=nc,
        activation_mtx=Am,
        inhibition_mtx=Rm,
        update_mtx=U,
        dt=20.0,
        nt=2000,
        mean_mRNA_init=10.0,
        DEFAULT_PARAMS=gill.DEFAULT_PARAMS,
    )

    if n_threads is None:
        n_threads = cpu_count(logical=True)
    else:
        n_threads = min(n_threads, cpu_count(logical=True))
    print(f"Assembling pool of {n_threads} processes")

    ssa_old = gill.GillespieSSA(**ssa_kw, PARAM_RANGES=gill.SAMPLING_RANGES)
    ssa_new = gill_new.GillespieSSA(**ssa_kw, PARAM_RANGES=gill_new.SAMPLE_RANGES)
    uniform_samples_old = get_uniform_samples(n_samples, seed, gill.SAMPLING_RANGES)
    uniform_samples_new = get_uniform_samples(n_samples, seed, gill_new.SAMPLE_RANGES)
    param_sets_old = np.array(
        [
            gill.convert_uniform_to_params(u, gill.SAMPLING_RANGES)
            for u in uniform_samples_old
        ]
    )
    param_sets_new = np.array(
        [
            gill_new.convert_uniform_to_params(u, gill_new.SAMPLE_RANGES)
            for u in uniform_samples_new
        ]
    )
    inputs = list(
        enumerate(
            [(p, ssa_old) for p in param_sets_old]
            + [(p, ssa_new) for p in param_sets_new]
        )
    )

    pop0 = np.zeros(ssa_old.n_species, dtype=np.int64)
    pop0[nc : nc * 2] = 10

    run_one_param_set = partial(run_gillespie_random_params, pop0=pop0)

    print(f"Running {n_samples} SSA samples each for old and new sampling ranges.")
    pbar = tqdm(total=2 * n_samples)

    # Processes should be initialized by first running an SSA to JIT-compile the model
    _init_thread = lambda: (
        run_one_param_set(inputs[0]),
        run_one_param_set(inputs[n_samples]),
    )
    with mp.Pool(n_threads, initializer=_init_thread) as pool:
        results = []
        for res in pool.imap_unordered(run_one_param_set, inputs):
            results.append(res)
            pbar.update(1)

    ...

    results_order, times = zip(*results)
    reorder = np.argsort(results_order)
    times = np.array(times)[reorder]
    times_old = times[:n_samples]
    times_new = times[n_samples:]

    sampled_quantities_old = np.array(
        [
            _rescale(u, lo, hi)
            for u, (lo, hi) in zip(uniform_samples_old.T, gill.SAMPLING_RANGES)
        ]
    )
    sampled_quantities_new = np.array(
        [
            _rescale(u, lo, hi)
            for u, (lo, hi) in zip(uniform_samples_new.T, gill_new.SAMPLE_RANGES)
        ]
    )

    data_old = dict(runtime_seconds=times_old) | dict(
        zip(gill.SAMPLED_VAR_NAMES, sampled_quantities_old)
    )
    data_new = dict(runtime_seconds=times_new) | dict(
        zip(gill_new.SAMPLE_VAR_NAMES, sampled_quantities_new)
    )
    df_old = pd.DataFrame(data_old).sort_values(by="runtime_seconds", ascending=False)
    df_new = pd.DataFrame(data_new).sort_values(by="runtime_seconds", ascending=False)

    if save:
        fpath_old = (
            Path(save_dir)
            .joinpath(f"2023-06-15_random_sample_runtimes_oldranges.csv")
            .resolve()
            .absolute()
        )
        print(f"Writing results to: {fpath_old}")
        df_old.to_csv(fpath_old, index=False)

        fpath_new = (
            Path(save_dir)
            .joinpath(f"2023-06-15_random_sample_runtimes_newranges.csv")
            .resolve()
            .absolute()
        )
        print(f"Writing results to: {fpath_new}")
        df_new.to_csv(fpath_new, index=False)

    ...


if __name__ == "__main__":
    save_dir = Path("data/oscillation/gillespie_runtime")
    main(save_dir=save_dir, n_samples=1500, n_threads=None, save=True)
