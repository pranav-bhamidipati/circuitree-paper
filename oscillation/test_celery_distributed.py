from functools import partial
from typing import Optional
from circuitree.parallel import ParameterTable
from oscillation_parallel_celery import OscillationTreeCelery
import pandas as pd
from pathlib import Path
from gillespie import PARAM_NAMES
from multiprocessing import Pool


def run_circuit_search(
    seed: int,
    n_steps: int,
    param_sets_csv: Path,
    init_cols: list[str],
    index_col: str,
    **tree_kw,
):
    param_df = pd.read_csv(param_sets_csv, index_col=index_col)
    param_df.index.name = "visit"
    param_table = ParameterTable.from_dataframe(
        param_df,
        seed_col="visit",
        init_cols=init_cols,
        param_cols=PARAM_NAMES,
    )

    save_dir = master_save_dir.joinpath(f"tree_{seed}")
    save_dir.mkdir(exist_ok=True)
    tree = OscillationTreeCelery(
        parameter_table=param_table, seed=seed, save_dir=save_dir, **tree_kw
    )
    tree.search_mcts(n_steps)


def main(
    param_sets_csv: Path,
    master_save_dir: Path,
    n_trees=12,
    batch_size=5,
    start_seed=2023,
    n_steps=1_000_000,
    index_col: str = "sample_num",
    max_interactions: Optional[int] = None,
):
    tree_kw = dict(
        n_steps=n_steps,
        param_sets_csv=param_sets_csv,
        index_col=index_col,
        root="ABCDE::",
        components=["A", "B", "C", "D", "E"],
        interactions=["activates", "inhibits"],
        max_interactions=max_interactions,
        init_cols=["A_0", "B_0", "C_0", "D_0", "E_0"],
        dt=20.0,  # seconds
        nt=2000,
        batch_size=batch_size,
    )

    # Uncomment for 3-component oscillation
    tree_kw |= dict(
        root="ABC::",
        components=["A", "B", "C"],
        interactions=["activates", "inhibits"],
        init_cols=["A_0", "B_0", "C_0"],
    )

    seeds = range(start_seed, start_seed + n_trees)
    run_search_in_parallel = partial(run_circuit_search, **tree_kw)

    print(f"Making {n_trees} search trees with batch size {batch_size}.")
    print(f"Reading parameter table from {param_sets_csv}")
    print(f"Saving results to {master_save_dir}")

    if n_trees == 1:
        run_search_in_parallel(seeds[0])
    else:
        with Pool(n_trees) as pool:
            print()
            print("Starting tree search...")
            pool.map(run_search_in_parallel, seeds)


if __name__ == "__main__":
    from datetime import date

    today = date.today().strftime("%y%m%d")
    param_sets_csv = Path(
        # "~/git/circuitree-paper/data/oscillation/param_sets_queue_10000_5tf.csv"
        "~/git/circuitree-paper/data/oscillation/param_sets_queue_10000.csv"  # uncomment for 3-TF oscillation
    ).expanduser()
    master_save_dir = Path(
        # f"~/git/circuitree-paper/data/oscillation/mcts/{today}_distributed"
        f"~/git/circuitree-paper/data/oscillation/mcts/{today}_3tf_localredis"  # uncomment for 3-TF oscillation
    ).expanduser()
    master_save_dir.mkdir(exist_ok=True)

    main(
        param_sets_csv=param_sets_csv,
        master_save_dir=master_save_dir,
        n_trees=25,
        batch_size=5,
        start_seed=2023,
        n_steps=1_000_000,
    )
