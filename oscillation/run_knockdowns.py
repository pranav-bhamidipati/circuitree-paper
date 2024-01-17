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
)


def run_ssa(
    args: tuple[str, float],
    genotype: str,
    params: np.ndarray,
    dt_secs: float,
    nt: int,
    nchunks: int = 1,
) -> dict:
    run_idx, knockdown_tf, knockdown_coeff, seed = args
    model = TFNetworkModel(genotype, initialize=True, seed=seed, dt=dt_secs, nt=nt)
    (
        y_t,
        where_minimum,
        frequency,
        acf_minimum,
    ) = model.run_with_params_knockdown_and_get_results(
        params, knockdown_tf, knockdown_coeff, seed=seed, nchunks=nchunks
    )
    result = (
        genotype,
        model.t,
        y_t,
        where_minimum,
        acf_minimum,
        frequency,
    )
    return run_idx, (knockdown_tf, knockdown_coeff, seed), result


def main(
    genotype: str,
    knockdown_coeffs: Iterable[float] = np.linspace(1, 0, 11),
    params: np.ndarray = DEFAULT_PARAMS,
    n_replicates: int = 10,
    knockdown_tfs: Optional[list[str]] = None,
    ACF_threshold: float = -0.4,
    seed: int = 2023,
    nchunks: int = 1,
    dt_secs: float = DEFAULT_TIME_STEP_SECONDS,
    nt: int = DEFAULT_N_TIME_POINTS,
    nprocs: int = 1,
    progress: bool = False,
    save: bool = False,
    save_dir: Optional[Path] = None,
    suffix: str = "",
):
    knockdown_tfs = knockdown_tfs or list(genotype.strip("*").split("::")[0])
    knockdown_coeffs = np.array(knockdown_coeffs)
    n_kds = len(knockdown_tfs)
    n_coeffs = len(knockdown_coeffs)

    # Run SSA
    run_one_ssa = partial(
        run_ssa,
        genotype=genotype,
        params=params,
        dt_secs=dt_secs,
        nt=nt,
        nchunks=nchunks,
    )

    run_seeds = np.random.SeedSequence(seed).generate_state(
        n_replicates * n_kds * n_coeffs
    )
    iter_run_seeds = iter(run_seeds)
    kd_tfs_and_coeffs = list(product(knockdown_tfs, knockdown_coeffs))
    run_args = []
    i = 0
    for kd_tf, kd_coeff in kd_tfs_and_coeffs:
        for seed in islice(iter_run_seeds, n_replicates):
            run_args.append((i, kd_tf, kd_coeff, seed))
            i += 1

    # for kd_tf, kd_coeff in kd_tfs_and_coeffs:
    #     for seed in islice(iter_run_seeds, n_replicates):
    #         run_args.append((len(run_args), kd_tf, kd_coeff, seed))

    # for i, seed in enumerate(run_seeds):
    #     kd_tf, kd_coeff = kd_tfs_and_coeffs[i % (n_kds * n_coeffs)]
    #     run_args.append((i, kd_tf, kd_coeff, seed))

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
    kd_tfs, kd_coeffs, seeds = zip(*inputs)
    replicates = np.array(indices) % n_replicates
    metadata = pd.DataFrame(
        {
            "run_idx": indices,
            "knockdown_tf": kd_tfs,
            "knockdown_coeff": kd_coeffs,
            "seed": seeds,
            "replicate": replicates,
        }
    )

    # Sort results
    metadata = metadata.set_index("run_idx").sort_index()
    outputs = [out for _, out in sorted(zip(indices, outputs), key=lambda x: x[0])]

    # Save results
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

    print(
        f"Done! {len(genotypes)} simulations completed for KD mutants of:\n\t{genotype}\n"
    )
    print("Found the following results (mean +/- std)...")
    for i, kd_tf in enumerate(knockdown_tfs):
        print(f"Knockdown: {kd_tf}")
        print("=" * len(f"Knockdown: {kd_tf}"))
        for j, kd_coeff in enumerate(knockdown_coeffs):
            oscillated_mask = acf_minima[i, j] < ACF_threshold
            if oscillated_mask.sum() == 0:
                print("\tNo oscillations")
            else:
                print(f"\t{oscillated_mask.sum()} / {n_replicates} oscillated")
                acf_min_mean = acf_minima[i, j][oscillated_mask].mean()
                acf_min_std = acf_minima[i, j][oscillated_mask].std()
                freq_mean = frequencies[i, j][oscillated_mask].mean()
                freq_std = frequencies[i, j][oscillated_mask].std()
                print(f"KD coeff: {kd_coeff:.2f}")
                print(
                    f"\tACF min = {acf_min_mean:.3f} +/- {acf_min_std:.3f}, freq (1/min) = {freq_mean:.3f} +/- {freq_std:.3f}"
                )
        print("\n")

    if save:
        today = datetime.today().strftime("%y%m%d")
        hdf_path = Path(save_dir) / f"{today}_knockdowns_data{suffix}.hdf5"
        print(f"Writing to: {hdf_path.resolve().absolute()}")
        with h5py.File(hdf_path, "w") as f:
            f.attrs["genotype"] = genotype
            f.create_dataset("t_secs", data=t)
            # f.create_dataset("knockdown_tfs", data=knockdown_tfs)
            # f.create_dataset("knockdown_coeffs", data=knockdown_coeffs)
            # f.create_dataset("replicate_seeds", data=run_seeds)
            f.create_dataset("params", data=params)
            f.create_dataset("y_ts", data=y_ts)
            f.create_dataset("where_minima", data=where_minima)
            f.create_dataset("acf_minima", data=acf_minima)
            f.create_dataset("frequencies", data=frequencies)

        metadata.to_hdf(hdf_path, key="metadata", mode="a")


if __name__ == "__main__":
    # 5-component oscillator with 4/5 FT
    genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_BDa_CBi_CCa_CDi_CEi_DAi_DBi_DCa_EAi_EEa"
    params = DEFAULT_PARAMS

    today = datetime.today().strftime("%y%m%d")
    save_dir = Path(f"data/oscillation/{today}_5TF_FTO3_knockdowns")
    save_dir.mkdir(exist_ok=True)

    main(
        genotype=genotype,
        params=params,
        seed=2023,
        # nprocs=1,
        # nchunks=10,
        # knockdown_coeffs=[0.0, 1.0],
        # n_replicates=3,
        nprocs=14,
        knockdown_coeffs=np.linspace(1, 0, 21),
        n_replicates=10,
        progress=True,
        save=True,
        save_dir=save_dir,
    )
