from functools import partial
from itertools import cycle
from typing import Iterable
from circuitree import DefaultFactoryDict
from circuitree.parallel import TranspositionTable, TableBuffer
from dask.distributed import Lock, Client, LocalCluster, as_completed, Future
import numpy as np
from pathlib import Path

from oscillation import TFNetworkModel
from oscillation_parallel import OscillationParallelTree


def run_one_mcts_step(tree: OscillationParallelTree):
    tree.traverse()
    return tree


def search_indefinite(
    trees: Iterable[OscillationParallelTree], client: Client, n_iters=None
):
    """Acts like a while True loop, but each tree runs in parallel, moving to the next
    iteration of MCTS upon finishing the last one.
    """

    n_trees = len(trees)

    if n_iters is None:
        maxiter = np.inf
    else:
        maxiter = n_iters * n_trees

    try:
        futures = client.map(run_one_mcts_step, trees)
        trees_as_completed = as_completed(futures, with_results=True)
        k = 0
        while k < maxiter:
            _, tree = next(trees_as_completed)
            future = client.submit(run_one_mcts_step, tree)
            trees_as_completed.add(future)
            k += 1

    except KeyboardInterrupt:
        print("Exiting search")
        all_futures = [Future(k) for ks in client.has_what().values() for k in ks]
        client.cancel(all_futures)


def search_indefinite_noparallel(
    trees: Iterable[OscillationParallelTree], n_iters=None
):
    n_trees = len(trees)

    if n_iters is None:
        maxiter = np.inf
    else:
        maxiter = n_iters * n_trees

    endless_tree_iterator = cycle(trees)

    try:
        for k in range(maxiter):
            tree = next(endless_tree_iterator)
            tree.traverse()
    except KeyboardInterrupt:
        print("Exiting search")


def initialize_model(genotype: str, dt: float, nt: int, init_mean: float):
    model = TFNetworkModel(genotype)
    model.initialize_ssa(dt, nt, init_mean)
    return model


def main(
    save_dir: Path,
    n_trees: int,
    n_iters_per_tree: int,
    nt: int = 2000,
    dt: float = 20.0,
    init_mean: float = 10.0,
    buffersize: int = 1000,
    n_workers: int = 10,
    extension: str = "csv",
):
    components = ["A", "B", "C"]
    interactions = ["activates", "inhibits"]
    root = "ABC::"
    init_condition_names = ["A_0", "B_0", "C_0"]
    param_names = [
        "k_on",
        "k_off_1",
        "k_off_2",
        "km_unbound",
        "km_act",
        "km_rep",
        "km_act_rep",
        "kp",
        "gamma_m",
        "gamma_p",
    ]
    columns = ["reward"] + init_condition_names + param_names

    time_points = np.linspace(0, nt * dt, nt, endpoint=False)
    init_model = partial(initialize_model, dt=dt, nt=nt, init_mean=init_mean)

    model_table = DefaultFactoryDict(default_factory=init_model)
    txp_table = TranspositionTable(results_colnames=columns)
    buffer = TableBuffer(columns=columns, save_dir=save_dir, maxsize=10)

    parallel_trees = []
    for seed in range(n_trees):
        tree = OscillationParallelTree(
            components=components,
            interactions=interactions,
            root=root,
            seed=seed,
            time_points=time_points,
            success_threshold=0.005,
            autocorr_threshold=0.5,
            init_mean=10.0,
            columns=columns,
            model_table=model_table,
            transposition_table=txp_table,
            buffer=buffer,
            save_dir=save_dir,
            maxsize=buffersize,
            extension=extension,
        )
        parallel_trees.append(tree)

    cluster = LocalCluster(n_workers=n_workers)
    with Client(cluster) as client:
        print("Link to dashboard:", client.dashboard_link)
        search_indefinite(parallel_trees, client, n_iters=n_iters_per_tree)

        # Single-threaded version (for debug)
        # search_indefinite_noparallel(parallel_trees, n_iters=n_iters_per_tree)


if __name__ == "__main__":
    save_dir = Path("/tmp/data/circuitree/oscillation")
    main(
        save_dir=save_dir,
        n_trees=10,
        n_iters_per_tree=10,
        buffersize=1,
        n_workers=10,
        # extension="parquet",
    )
