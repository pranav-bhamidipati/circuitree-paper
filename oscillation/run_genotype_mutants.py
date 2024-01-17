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
) -> dict:
    genotype, init_proteins = genotype_and_init_proteins
    model = TFNetworkModel(genotype, initialize=True, seed=seed, dt=dt_secs, nt=nt)
    pop0 = model.ssa.population_from_proteins(init_proteins)
    t = model.t
    y_t = model.run_ssa_with_params(pop0, params=DEFAULT_PARAMS)
    prots_t = y_t[..., model.m : model.m * 2]
    frequency, acf_minimum = model.get_acf_minima_and_results(
        t, prots_t, freqs=True, abs=False
    )
    return genotype, t, y_t, acf_minimum, frequency


def main(
    genotype: str,
    seed: int = 2023,
    dt_secs: float = DEFAULT_TIME_STEP_SECONDS,
    nt: int = DEFAULT_N_TIME_POINTS,
    nprocs: int = 1,
    pool: Optional[Pool] = None,
    progress: bool = False,
    save: bool = False,
    save_dir: Optional[Path] = None,
    suffix: str = "",
):
    components_wt, interactions_joined = genotype.strip("*").split("::")

    # Generate all deletion mutants
    genotypes = [genotype]
    for component in components_wt:
        mut_components = components_wt.replace(component, "")
        mut_interactions = "_".join(
            [i for i in interactions_joined.split("_") if component not in i]
        )
        genotypes.append(f"*{mut_components}::{mut_interactions}")

    # Draw initial conditions
    init_seed, ssa_seed = np.random.SeedSequence(seed).generate_state(2)
    rg = np.random.default_rng(init_seed)

    # Protein counts are Poisson distributed
    init_proteins_wt = rg.poisson(
        MEAN_INITIAL_POPULATION, size=len(components_wt)
    ).tolist()

    # All other species are zero
    init_proteins = [init_proteins_wt] + [
        init_proteins_wt[:i] + init_proteins_wt[i + 1 :]
        for i in range(len(components_wt))
    ]
    init_proteins = [np.array(p, dtype=np.int64) for p in init_proteins]

    # Run SSA
    run_one_ssa = partial(run_ssa, seed=ssa_seed, dt_secs=dt_secs, nt=nt)
    args = list(zip(genotypes, init_proteins))

    if nprocs > 1 or pool is not None:
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
    mutations = ["WT"] + list(components_wt)

    data = zip(
        mutations,
        genotypes,
        y_ts,
        acf_minima,
        frequencies,
    )
    print(f"Done! {len(genotypes)} simulations completed for mutants of:\n\t{genotype}")
    print(f"Found ACF minima:")
    for mut, gen, acfmin in zip(mutations, genotypes, acf_minima):
        print(f"\t{mut}: {acfmin:.3f}")

    if save:
        today = datetime.today().strftime("%y%m%d")
        hdf_path = Path(save_dir) / f"{today}_mutants_data{suffix}.hdf5"
        print(f"Writing to: {hdf_path.resolve().absolute()}")
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("t_secs", data=t)
            f.attrs["seed"] = seed
            for mut, genotype, y_t, acf_min, freq_per_sec in data:
                dset = f"{mut}"
                f.create_dataset(dset, data=y_t)
                f[dset].attrs["genotype"] = genotype
                f[dset].attrs["ACF_min"] = acf_min
                f[dset].attrs["frequency_per_sec"] = freq_per_sec


if __name__ == "__main__":
    # genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_BCa_CBi_CCa_CDi_CEi_DAi"
    # suffix = "_FTO"

    # genotype = "*ABCDE::AAa_ABa_ACa_ADi_BAi_BBa_BCi_BEi_CDi_DAa_DBi_DDa_ECi"
    # suffix = "_FTO2"

    # genotype = "*ABCDE::AAa_ABa_ACa_ADi_BAi_BBa_BCi_BEi_CDi_DAa_DBi_DCi_DDa_EAi_ECi"
    # suffix = "_FTO3"

    # genotype = "*ABCDE::AAa_ABi_ADi_BBa_BCi_CCa_CAi_DDa_DCi_DEi_EEa_EAi"
    # suffix = "_Matt_FTO"

    # genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_CBi_CCa_CDi_DAi_DDa_DEi_ECi_EEa"
    # suffix = "_Matt_FTO2"

    # genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_CBi_CCa_CDi_DAi_DEi_ECi"
    # suffix = "_Matt_FTO3"

    # genotype = "*ABCDEF::AAa_ACi_BAi_BBa_CBi_CCa_CDi_DAi_DDa_DEi_ECi_EEa_EFi_FDi_FFa"
    # suffix = "_Matt_FTO_6TF"

    # genotype = "*ABCDE::AAa_ACi_BAi_BBa_CBi_CCa_CDi_DAi_DDa_DEi_EDa"
    # suffix = "_PB_FTO"

    # genotype = "*ABCDE::AAa_ACi_BAi_BBa_CBi_CCa_CDi_DAi_DDa_DEa_EDi"
    # suffix = "_PB_FTO2"

    # genotype = "*ABCDE::AAa_ABa_BAi_BBa_BCi_CCa_CDi_DBi_DDa_DEi_ECi_EEa"
    # suffix = "_PB_FTO3"

    # # 5/5 fault-tolerant!!
    # genotype = "*ABCDE::AAa_ABa_BAi_BCi_BDi_BEi_CCa_CDi_DDa_DEi_ECi_EEa"
    # suffix = "_PB_FTO4"

    ### Three interesting cases from the 5TF fault-tolerant oscillator search

    # # Highest Q overall
    # genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_CBi_CCa_CDi_CEi_DAi_DBi_DDa_EAi_ECa"
    # suffix = "_5TF_FTO_highly_fault_tolerant"

    # # Highest Q among samples without mutation
    # genotype = "*ABCDE::AAa_ABa_ACi_ADi_BAi_BBa_BEi_CAi_CBa_CCa_DBi_DCi_DDa_DEi_EAi"
    # suffix = "_5TF_FTO_medium_fault_tolerant"

    # # Largest difference between Q with and without mutation
    # # genotype = "*ABCDE::AAa_ABa_ACi_ADi_AEi_BAi_BBa_BCa_CBi_CCa_DBi_DCa_DDa_EBi_ECi"
    # # genotype = "*ABCDE::AAa_ABa_ACi_ADi_BAi_BBa_CAi_CDa_DBi_DCi_DDa_DEi_EAi_ECa_EEa"
    # # genotype = "*ABCDE::AAa_ABa_ACa_ADi_AEi_BAi_BBa_CAi_CCa_DAa_DBi_DCi_DDa_DEi_EAi"
    # genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_BCi_CBi_CCa_CDi_CEi_DAi_DBi_DDa_EAi_EEa"
    # suffix = "_5TF_FTO_fragile_to_mut"

    # # Compare to normal oscillator w/out fault-tolerance (2/5)
    # genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_CBi_CCa"
    # suffix = "_5TF_AI+Rep"

    # # 4-component oscillator with 3/5 FT
    # genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_CBi_CCa_CDi_DAi_DDa"
    # suffix = "_5TF_AI+2Rep"

    # # 5-component oscillator with 3/5 FT
    # genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_CBi_CCa_CDi_CEi_DAi_DDa_EAi"
    # suffix = "_5TF_AI+3Rep"

    # 5-component oscillator with 4/5 FT
    genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_BDa_CBi_CCa_CDi_CEi_DAi_DBi_DCa_EAi_EEa"
    suffix = "_5TF_3AI+3Rep"

    ### Try some particular cases of possible 4/5 FT
    # genotype = "*ABCDE::AAa_ABa_ACi_ADi_BBa_BCi_BDi_CAa_CCa_CEi_DCi_EAi_EBi_EDi_EEa"
    # suffix = "_5TF_test1"
    # genotype = "*ABCDE::AAa_ABa_ACi_ADi_BAi_BBa_CAi_CDa_DBi_DCi_DDa_DEi_EAi_ECa_EEa"
    # suffix = "_5TF_test2"
    # genotype = "*ABCDE::AAa_ABa_ACa_ADi_BAi_BBa_CAi_CCa_DBi_DCi_DDa_DEi_EAi_EDa_EEa"
    # suffix = "_5TF_test3"
    # genotype = "*ABCDE::AAa_ABa_ACa_ADi_BAi_BBa_BDa_CAi_CCa_DBi_DCi_DDa_DEi_EAi_EBa"
    # suffix = "_5TF_test4"
    # genotype = "*ABCDE::AAa_ABa_ACi_BAi_BBa_BDa_CBi_CCa_CDi_CEi_DAi_DBi_DCa_EAi_EEa"
    # suffix = "_5TF_test5"

    # save_dir = Path("data/oscillation/231120_5TF_FTO_mutants")
    save_dir = Path("data/oscillation/231121_5TF_FTO_mutants")
    save_dir.mkdir(exist_ok=True)

    main(
        genotype=genotype,
        seed=2023,
        # nprocs=1,
        nprocs=6,
        progress=True,
        save=True,
        save_dir=save_dir,
        suffix=suffix,
    )
