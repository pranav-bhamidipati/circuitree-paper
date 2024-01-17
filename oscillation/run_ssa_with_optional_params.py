from circuitree import SimpleNetworkGrammar
from datetime import datetime
import h5py
from functools import partial
from multiprocessing.pool import Pool
import numpy as np
from pathlib import Path
from typing import Optional

from tf_network import TFNetworkModel
from gillespie import (
    DEFAULT_PARAMS,
    MEAN_INITIAL_POPULATION,
    DEFAULT_TIME_STEP_SECONDS,
    DEFAULT_N_TIME_POINTS,
)


def run_ssa(
    genotype_and_init_proteins: tuple[str, np.ndarray],
    seed: int,
    dt_secs: float,
    nt: int,
    params: np.ndarray = DEFAULT_PARAMS,
) -> dict:
    genotype, init_proteins = genotype_and_init_proteins
    model = TFNetworkModel(genotype, initialize=True, seed=seed, dt=dt_secs, nt=nt)
    pop0 = model.ssa.population_from_proteins(init_proteins)
    t = model.t
    y_t = model.run_ssa_with_params(pop0, params=params)
    prots_t = y_t[..., model.m : model.m * 2]
    frequency, acf_minimum = model.get_acf_minima_and_results(
        t, prots_t, freqs=True, abs=False
    )
    return genotype, t, y_t, acf_minimum, frequency


def main(
    genotypes: list[str],
    seed: int = 2023,
    dt_secs: float = DEFAULT_TIME_STEP_SECONDS,
    nt: int = DEFAULT_N_TIME_POINTS,
    nprocs: int = 1,
    pool: Optional[Pool] = None,
    progress: bool = False,
    save: bool = False,
    save_dir: Optional[Path] = None,
    suffix: str = "",
    params: np.ndarray = DEFAULT_PARAMS,
):
    # Draw initial conditions
    init_seed, ssa_seed = np.random.SeedSequence(seed).generate_state(2)
    rg = np.random.default_rng(init_seed)

    # Protein counts are Poisson distributed
    components, *_ = SimpleNetworkGrammar.parse_genotype(genotypes[0])
    init_proteins_single = rg.poisson(
        MEAN_INITIAL_POPULATION, size=len(components)
    ).tolist()
    init_proteins = [np.array(init_proteins_single, dtype=np.int64)] * len(genotypes)

    # Run SSA
    run_one_ssa = partial(run_ssa, seed=ssa_seed, dt_secs=dt_secs, nt=nt, params=params)
    args = list(zip(genotypes, init_proteins))

    if len(genotypes) > 1 and (nprocs != 1 or pool is not None):
        if progress:
            from tqdm import tqdm

            pbar = tqdm(total=len(genotypes), desc="Running SSAs")

        results = []
        if pool is not None:
            for result in pool.imap_unordered(run_one_ssa, args):
                results.append(result)
                if progress:
                    pbar.update()
        else:
            with Pool(nprocs) as pool:
                for result in pool.imap_unordered(run_one_ssa, args):
                    results.append(result)
                    if progress:
                        pbar.update()

    else:
        iterator = args
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(genotypes), desc="Running SSAs")

        results = [run_one_ssa(g) for g in iterator]

    # Save results
    run_genotypes, ts, y_ts, acf_minima, frequencies = zip(*results)
    t = ts[0]
    gen_to_result = dict(zip(run_genotypes, zip(y_ts, acf_minima, frequencies)))
    y_ts = [gen_to_result[g][0] for g in genotypes]
    acf_minima = np.array([gen_to_result[g][1] for g in genotypes])
    frequencies = np.array([gen_to_result[g][2] for g in genotypes])

    print(f"Done! {len(genotypes)} simulations completed.")
    print(f"Found ACF minima:")
    for gen, acfmin in zip(genotypes, acf_minima):
        print(f"\t{gen}: {acfmin:.3f}")

    data = zip(genotypes, y_ts, acf_minima, frequencies)

    if save:
        today = datetime.today().strftime("%y%m%d")
        hdf_path = Path(save_dir) / f"{today}_simulation_data{suffix}.hdf5"
        print(f"Writing to: {hdf_path.resolve().absolute()}")
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("t_secs", data=t)
            f.attrs["seed"] = seed
            for genotype, y_t, acf_min, freq_per_sec in data:
                f.create_dataset(genotype, data=y_t)
                f[genotype].attrs["ACF_min"] = acf_min
                f[genotype].attrs["frequency_per_sec"] = freq_per_sec


if __name__ == "__main__":
    # Variations of activator-inhibitor w/w/o additions like PAR, suppressor, etc.
    genotypes = [
        "*ABC::ABa_BAi",
        "*ABC::AAa_ABa_BAi",
        "*ABC::ABa_BAi_CBi",
        "*ABC::ABa_BAi_CAa",
        "*ABC::AAa_ABa_BAi_CBi",
    ]
    suffix = "_3TF_AI_variations"

    today = datetime.today().strftime("%y%m%d")
    save_dir = Path(f"data/oscillation/{today}_AI_variations")
    save_dir.mkdir(exist_ok=True)

    params = (
        8.760e-02,  # k_on
        9.991e01,  # k_off_1
        5.672e00,  # k_off_2
        1.699e-01,  # km_unbound
        8.149e00,  # km_act
        5.480e-04,  # km_rep
        1.699e-01,  # km_act_rep
        8.419e-02,  # kp
        1.608e-02,  # gamma_m
        1.103e-02,  # gamma_p
    )

    main(
        genotypes=genotypes,
        params=params,
        seed=2024,
        nprocs=5,
        progress=True,
        save=True,
        save_dir=save_dir,
        suffix=suffix,
    )
