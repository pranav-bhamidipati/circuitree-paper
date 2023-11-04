from functools import partial
from typing import Optional
from old_scripts.oscillation_local_celery import OscillationTreeCeleryLocal
from pathlib import Path
from multiprocessing import Pool


def run_circuit_search(seed: int, n_steps: int, **tree_kw):
    save_dir = master_save_dir.joinpath(f"tree_{seed}")
    save_dir.mkdir(exist_ok=True)
    tree = OscillationTreeCeleryLocal(seed=seed, save_dir=save_dir, **tree_kw)
    tree.search_mcts(n_steps)


def progress_callback(tree: OscillationTreeCeleryLocal, *args, **kwargs):
    """Callback function to report progress of MCTS search."""
    _progress_msg = f"Iteration {tree.iteration_counter} complete for tree {tree.seed}"
    tree.logger.info(_progress_msg)
    print(_progress_msg)


def main(
    param_sets_csv: Path,
    master_save_dir: Path,
    n_trees=12,
    batch_size=5,
    start_seed=2023,
    n_steps=1_000_000,
    index_col: str = "sample_num",
    max_interactions: Optional[int] = None,
    time_limit: int = 120,  # seconds per batch
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
        time_limit=time_limit,
    )

    # # Uncomment for 3-component oscillation
    # tree_kw |= dict(
    #     root="ABC::",
    #     components=["A", "B", "C"],
    #     interactions=["activates", "inhibits"],
    #     init_cols=["A_0", "B_0", "C_0"],
    # )

    seeds = range(start_seed, start_seed + n_trees)
    run_search_in_parallel = partial(run_circuit_search, **tree_kw)

    print(f"Making {n_trees} search trees with batch size {batch_size}.")
    print(f"Reading parameter table from {param_sets_csv}")
    print(f"Saving results to {master_save_dir}")

    print()
    print("Starting tree search...")
    if n_trees == 1:
        run_search_in_parallel(seeds[0])
    else:
        with Pool(n_trees) as pool:
            pool.map(run_search_in_parallel, seeds)


if __name__ == "__main__":
    from datetime import date

    today = date.today().strftime("%y%m%d")
    param_sets_csv = Path(
        "~/git/circuitree-paper/data/oscillation/param_sets_queue_10000_5tf.csv"
    ).expanduser()
    master_save_dir = Path(
        f"~/git/circuitree-paper/data/oscillation/mcts/{today}_distributed"
    ).expanduser()
    master_save_dir.mkdir(exist_ok=True)

    # # Uncomment for 3-TF oscillation
    # param_sets_csv = Path(
    #     "~/git/circuitree-paper/data/oscillation/param_sets_queue_10000.csv"
    # ).expanduser()
    # master_save_dir = Path(
    #     f"~/git/circuitree-paper/data/oscillation/mcts/{today}_3tf_localredis"
    # ).expanduser()
    # master_save_dir.mkdir(exist_ok=True)

    main(
        param_sets_csv=param_sets_csv,
        master_save_dir=master_save_dir,
        max_interactions=12,
        # n_trees=1, # For testing
        n_trees=20,
        batch_size=5,
        start_seed=2023,
        n_steps=1_000_000,
    )
