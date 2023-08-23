from collections import Counter
from functools import partial
from multiprocessing import Pool, Semaphore, active_children, current_process
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from psutil import cpu_count
from typing import Iterable, Optional

from tqdm import tqdm

from circuitree.metrics import (
    mcts_reward_and_nodes_visited,
    # mcts_reward_nodes_and_modularity_estimate,
)
from circuitree.parallel import TranspositionTable
from oscillation_parallel import OscillationTreeParallel

from time import perf_counter

### Use the param sets and select just the low-gamma regime
### Should save the runtime as well as param sets, rewards, etc.
### Use the reaction rate limit from the analysis of the 3-TF BFS data


def run_search_and_save(
    batch_size_and_seed: tuple[int],
    n_steps_total: int,
    transposition_table: TranspositionTable,
    save_columns: Iterable[str],
    save_dir: Path,
    progress_bar: bool = True,
    pbar_position: Optional[int] = None,
    pbar: Optional[tqdm] = None,
    **kwargs,
):
    batch_size, seed = batch_size_and_seed
    ot = OscillationTreeParallel(
        pool=None,
        transposition_table=transposition_table,
        read_only=True,
        batch_size=batch_size,
        seed=seed,
        **kwargs,
    )
    n_steps = n_steps_total // ot.batch_size
    results = ot.search_mcts(
        n_steps,
        metric_func=mcts_reward_and_nodes_visited,
        progress_bar=progress_bar,
        pbar_position=pbar_position,
        pbar=pbar,
    )

    res_data = pd.DataFrame(
        results, columns=save_columns, index=pd.RangeIndex(0, n_steps + 1, name="step")
    )
    res_data["selected_node"] = pd.Categorical(res_data["selected_node"])
    res_data["simulated_node"] = pd.Categorical(res_data["simulated_node"])

    stem = f"oscillation_circuit_tree_batchsize{batch_size}_seed{seed}_{uuid4()}"
    save_dir = Path(save_dir)
    gml_target = save_dir.joinpath(f"{stem}.gml")
    attrs_target = save_dir.joinpath(f"{stem}.json")
    df_target = save_dir.joinpath(f"{stem}.csv")

    ot.to_file(gml_target, attrs_target)
    res_data.to_csv(df_target)

    return batch_size


def run_search_in_parallel(
    batch_size_seed_and_task: tuple[int],
    n_steps_total: int,
    progress_bar: bool = True,
    pbar_positions: dict[int, tqdm] = None,
    **kwargs,
):
    pid = current_process().pid
    if pbar_positions is None:
        pbar = None
    else:
        position = pbar_positions[pid]
        batch_size, seed, task_num, task_total = batch_size_seed_and_task
        pbar = tqdm(
            range(n_steps_total // batch_size),
            position=position,
            desc=f"Process {pid} - Task {task_num}/{task_total}",
            leave=False,
        )

    batch_size = run_search_and_save(
        batch_size_and_seed=batch_size_seed_and_task[:2],
        n_steps_total=n_steps_total,
        transposition_table=table,  # declared as a global during process initialization
        progress_bar=progress_bar,
        pbar=pbar,
        **kwargs,
    )

    return pid, batch_size


def _load_table_in_process(
    semaphore: Semaphore,
    data_path: Path,
    load_kw: dict,
):
    "Use a semaphore to limit the number of file-loads that can happen simultaneously"
    global table

    with semaphore:
        table = _load_table(
            data_path=data_path,
            load_kw=load_kw,
        )


def _load_table(
    data_path: Path,
    load_kw: dict,
):
    if data_path.is_dir():
        pqfiles = list(data_path.glob("state_*/*.parquet"))
        if not pqfiles:
            raise ValueError(f"No parquet files found in data directory: {data_path}")
        table = TranspositionTable.from_parquet(
            pqfiles,
            load_kw=load_kw,
        )
    elif data_path.exists():
        if data_path.suffix != ".parquet":
            raise ValueError(f"Data file has invalid extension: '{data_path.suffix}'")
        table = TranspositionTable.from_parquet(
            data_path,
            load_kw=load_kw,
        )
    else:
        raise FileNotFoundError(f"Data file does not exist: {data_path.absolute()}")

    return table


def main(
    data_loc: Path,
    save_loc: Path,
    batch_sizes: Iterable[int],
    seeds: Iterable[int],
    n_steps_total: int = 1_000_000,
    parallel: bool = True,
    init_columns: Optional[list[str]] = None,
    param_names: Optional[list[str]] = None,
    components: Optional[list[str]] = None,
    interactions: Optional[list[str]] = None,
    root: str = "ABC::",
    nt: int = 100,
    dt: float = 20.0,
    reward_column: str = "reward",
    load_kw: dict = {},
    progress_bar: bool = True,
    n_workers: Optional[int] = None,
    max_readers_of_table: int = 12,
):
    if components is None:
        components = ["A", "B", "C"]
    if interactions is None:
        interactions = ["activates", "inhibits"]

    columns = [reward_column]
    if init_columns is not None:
        columns.extend(init_columns)
    if param_names is not None:
        columns.extend(param_names)

    # Read a transposition table from a file (or multiple files)
    data_path = Path(data_loc)
    load_kw["columns"] = ["state"] + columns

    save_dir = Path(save_loc)
    table_kw = dict(
        data_path=data_path,
        load_kw=load_kw,
    )

    order = np.argsort(batch_sizes)[::-1]
    n_batch_sizes = order.size
    batch_sizes_sorted = [batch_sizes[i] for i in order]
    seeds_sorted = [seeds[i] for i in order]
    save_columns = ["rewards", "selected_node", "simulated_node"]

    search_args = zip(batch_sizes_sorted, seeds_sorted)
    replicate_counter = Counter()
    if parallel:
        if n_workers is None:
            n_workers = cpu_count()
        n_workers = min(n_batch_sizes, n_workers)

        bs_str = ", ".join(map(str, batch_sizes_sorted))
        print(f"Running batch sizes {bs_str} in parallel with {n_workers} workers")

        semaphore = Semaphore(max_readers_of_table)
        _init_table = partial(_load_table_in_process, semaphore, **table_kw)

        with Pool(n_workers, initializer=_init_table) as pool:
            # Get the progress bar location for each process
            children = active_children()
            pbar_positions = {c.pid: i for i, c in enumerate(children)}

            # Construct the keyword arguments for the search
            kw = dict(
                n_steps_total=n_steps_total,
                progress_bar=progress_bar,
                columns=columns,
                nt=nt,
                dt=dt,
                components=components,
                interactions=interactions,
                root=root,
                bootstrap=True,
                save_columns=save_columns,
                save_dir=save_dir,
                pbar_positions=pbar_positions,
            )
            run_one_parallel = partial(run_search_in_parallel, **kw)

            # Run the searches in parallel
            parallel_args = [
                tuple(a) + (i + 1, n_batch_sizes) for i, a in enumerate(search_args)
            ]
            for pid, b in pool.imap_unordered(run_one_parallel, parallel_args):
                ...
            ...

    else:
        # Load the transposition table from file (or multiple files)
        print("Loading transposition table")
        table = _load_table(
            data_path,
            load_kw,
        )

        # Construct the keyword arguments for the search
        kw = dict(
            n_steps_total=n_steps_total,
            transposition_table=table,
            progress_bar=progress_bar,
            components=components,
            interactions=interactions,
            root=root,
            nt=nt,
            dt=dt,
            bootstrap=True,
            columns=columns,
            save_columns=save_columns,
            save_dir=save_dir,
        )

        for b, seed in search_args:
            replicate_counter[b] += 1
            print(
                f"Running batch size {b} - replicate {replicate_counter[b]}, seed {seed}"
            )
            _ = run_search_and_save((b, seed), **kw)

            ...

        ...

    print("Done")

    ...


if __name__ == "__main__":
    data_loc = Path("data/oscillation/230717_transposition_table_hpc.parquet")
    save_loc = Path("data/oscillation/230725_mcts_bootstrap_boolean2")
    save_loc.mkdir(exist_ok=True)

    # NOTE: Running many trees in parallel can be potentially memory intensive, since the
    #       transposition table can be very large and a copy must exist in each process.
    #       The problem is worst during the loading of the object, since it is not too
    #       efficient - there's a lot of memory copying.
    #       We reduce the memory burden by not reading in the initial conditions or
    #       simulation parameters, only the visit counts and rewards. Also, we use a
    #       file lock to ensure that only one process is loading the file at a time.
    #       Still, beware. Here lie dragons.

    n_replicates = 5
    batch_sizes = n_replicates * [1, 5, 10, 20, 50, 100]
    seeds = np.repeat(
        np.arange(n_replicates), len(batch_sizes) // n_replicates
    ).tolist()

    ...

    main(
        data_loc=data_loc,
        save_loc=save_loc,
        batch_sizes=batch_sizes,
        seeds=seeds,
        # parallel=False,
        n_workers=14,
    )
