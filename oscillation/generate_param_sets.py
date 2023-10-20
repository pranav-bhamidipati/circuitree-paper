from datetime import datetime
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats.qmc import LatinHypercube

from gillespie import (
    SAMPLING_RANGES,
    PARAM_NAMES,
    MEAN_INITIAL_POPULATION,
    convert_uniform_to_params,
)


def main(
    n_samples: int,
    save_dir: Path,
    n_components: int,
    seed: int = 2023,
    save: bool = False,
    suffix: str = "",
):
    n_params = len(SAMPLING_RANGES)

    # Draw random seeds for each sample
    seed_seq = np.random.SeedSequence(seed)
    prng_seeds = list(map(int, seed_seq.generate_state(n_samples + 1)))

    # Draw parameter sets and initial conditions
    rg = np.random.default_rng(prng_seeds.pop(0))
    lh_sampler = LatinHypercube(n_params, seed=rg)
    uniform_samples = lh_sampler.random(n_samples)
    param_sets = np.array(
        [convert_uniform_to_params(u, SAMPLING_RANGES) for u in uniform_samples]
    )
    initial_conditions = rg.poisson(
        MEAN_INITIAL_POPULATION, size=(n_samples, n_components)
    )

    seed_data = pd.DataFrame({"prng_seed": prng_seeds})
    seed_data.index.name = "param_index"
    init_columns = [f"{chr(ord('A') + i)}_0" for i in range(n_components)]
    init_data = pd.DataFrame(initial_conditions, columns=init_columns)
    init_data.index.name = "param_index"
    param_data = pd.DataFrame(param_sets, columns=PARAM_NAMES)
    param_data.index.name = "param_index"
    data = pd.concat([seed_data, init_data, param_data], axis=1)

    if save:
        today = datetime.now().strftime("%y%m%d")
        fname = Path(save_dir) / f"{today}_param_sets_{n_samples}{suffix}.csv"
        fname = fname.resolve().absolute()
        print(f"Writing parameter queue to {fname}")
        data.to_csv(fname)


if __name__ == "__main__":
    save_dir = Path("data/oscillation")
    save_dir.mkdir(exist_ok=True)
    main(
        n_samples=10_000,
        save_dir=save_dir,
        save=True,
        n_components=5,
        suffix="_5tf",
        # n_components=3,
        # suffix="_3tf",
    )
