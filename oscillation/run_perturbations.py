from datetime import datetime
from itertools import cycle, islice, product
import h5py
from functools import partial
from multiprocessing.pool import Pool
import numpy as np
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from tf_network import TFNetworkModel
from gillespie import (
    DEFAULT_PARAMS,
    DEFAULT_TIME_STEP_SECONDS,
    DEFAULT_N_TIME_POINTS,
    MEAN_INITIAL_POPULATION,
)


def run_ssa_perturbed(
    args: tuple[str, float],
    genotype: str,
    params: np.ndarray,
    dt_secs: float,
    nt: int,
    nchunks: int = 1,
    init_mean: float = MEAN_INITIAL_POPULATION,
) -> dict:
    run_idx, sigma_params, replicate, seed = args
    model = TFNetworkModel(genotype, initialize=True, seed=seed, dt=dt_secs, nt=nt)
    (
        y_t,
        where_minimum,
        frequency,
        acf_minimum,
    ) = model.run_with_params_perturbation_and_get_results(
        params, sigma_params, init_mean=init_mean, nchunks=nchunks
    )
    result = (
        genotype,
        model.t,
        y_t,
        where_minimum,
        acf_minimum,
        frequency,
    )
    return run_idx, (sigma_params, replicate, seed), result


def main(
    genotype: str,
    sigma_params: Iterable[float],
    params: np.ndarray = DEFAULT_PARAMS,
    n_replicates: int = 10,
    seed: int = 2023,
    nchunks: int = 1,
    dt_secs: float = DEFAULT_TIME_STEP_SECONDS,
    nt: int = DEFAULT_N_TIME_POINTS,
    init_mean: float = MEAN_INITIAL_POPULATION,
    nprocs: int = 1,
    progress: bool = False,
    save: bool = False,
    save_dir: Optional[Path] = None,
    suffix: str = "",
):
    # Run SSA
    run_one_ssa = partial(
        run_ssa_perturbed,
        genotype=genotype,
        params=params,
        dt_secs=dt_secs,
        nt=nt,
        nchunks=nchunks,
        init_mean=init_mean,
    )

    # Generate arguments for each run
    n_sigmas = len(sigma_params)
    n_runs = n_replicates * n_sigmas
    run_indices = np.arange(n_runs)
    run_seeds = np.random.SeedSequence(seed).generate_state(n_runs)
    sigma_idx, run_replicates = np.divmod(run_indices, n_replicates)
    run_sigmas = np.array(sigma_params)[sigma_idx]
    run_args = list(zip(run_indices, run_sigmas, run_replicates, run_seeds))

    # Run SSAs
    if nprocs > 1:
        if progress:
            from tqdm import tqdm

            pbar = tqdm(total=len(run_args), desc="Running SSAs")

        results = []
        with Pool(nprocs) as pool:
            for result in pool.imap_unordered(run_one_ssa, run_args):
                results.append(result)
                if progress:
                    pbar.update()

    else:
        iterator = run_args
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Running SSAs")
        results = [run_one_ssa(args) for args in iterator]

    indices, inputs, outputs = zip(*results)
    sigmas, replicates, seeds = zip(*inputs)
    metadata = pd.DataFrame(
        {
            "run_idx": indices,
            "sigma_param": sigmas,
            "replicate": replicates,
            "seed": seeds,
        }
    )

    # Sort results and format for saving
    metadata = metadata.set_index("run_idx").sort_index()
    outputs = [out for _, out in sorted(zip(indices, outputs), key=lambda x: x[0])]
    (
        genotypes,
        ts,
        y_ts_tup,
        where_minima_tup,
        acf_minima_tup,
        frequencies_tup,
    ) = zip(*outputs)
    genotype = genotypes[0]
    t = ts[0]
    y_ts = np.array(y_ts_tup, dtype=np.int64)
    where_minima = np.array(where_minima_tup, dtype=np.int64)
    acf_minima = np.array(acf_minima_tup, dtype=np.float64)
    frequencies = np.array(frequencies_tup, dtype=np.float64)

    print(f"Done! {len(genotypes)} simulations completed for perturbation mutants of")
    print(f"\tGenotype: {genotype}\n")
    if save:
        today = datetime.today().strftime("%y%m%d")
        hdf_path = Path(save_dir) / f"{today}_perturbation_data{suffix}.hdf5"
        print(f"Writing to: {hdf_path.resolve().absolute()}")
        with h5py.File(hdf_path, "w") as f:
            f.attrs["genotype"] = genotype
            f.create_dataset("t_secs", data=t)
            f.create_dataset("params", data=params)
            f.create_dataset("y_ts", data=y_ts)
            f.create_dataset("where_minima", data=where_minima)
            f.create_dataset("acf_minima", data=acf_minima)
            f.create_dataset("frequencies", data=frequencies)

        metadata.to_hdf(hdf_path, key="metadata", mode="a")

    ...

if __name__ == "__main__":
    # 5-component oscillator with 4/5 FT
    genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_BDa_CBi_CCa_CDi_CEi_DAi_DBi_DCa_EAi_EEa"
    params = DEFAULT_PARAMS

    today = datetime.today().strftime("%y%m%d")
    save_dir = Path(f"data/oscillation/{today}_5TF_FTO3_perturbations")
    save_dir.mkdir(exist_ok=True)

    main(
        genotype=genotype,
        params=params,
        seed=2023,
        # nprocs=1,
        nprocs=6,
        nchunks=10,
        sigma_params=[0.0, 1.0],
        n_replicates=3,
        # nprocs=14,
        # sigma_params=np.logspace(-2, 0, 21),
        # sigma_params=np.linspace(0, 0.2, 21),
        # n_replicates=10,
        progress=True,
        save=True,
        save_dir=save_dir,
    )
