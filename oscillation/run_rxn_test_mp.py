from functools import partial
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
from psutil import cpu_count
from time import perf_counter
from typing import Iterable, Optional
from uuid import uuid4

from oscillation import TFNetworkModel, OscillationTree
from gillespie import convert_params_to_sampled_quantities


def run_batch_and_save(
    input_data: pd.DataFrame,
    dt,
    nt,
    save_dir,
    param_names,
    log_dir,
    task_id=None,
    n_params: int = 8,
    save=True,
    ext: str = "parquet",
    init_cols: Iterable[str] = ["A_0", "B_0", "C_0"],
    param_cols: Iterable[str] = [
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
    ],
):
    start_logging(log_dir, task_id)

    sample_nums = input_data.index.values
    seeds = input_data["seed"].values
    prot0s = input_data[init_cols].values
    param_sets = input_data[param_cols].values
    genotypes = input_data["state"].values.tolist()
    batch_size = len(seeds)

    logging.info(f"Seeds: {seeds}")
    logging.info(f"Genotypes: {genotypes}")
    logging.info(f"Batch_size: {batch_size}")

    # If JIT compilation hasn't happened, dump the first simulation
    from gillespie import gillespie_trajectory_with_n_iter

    if not gillespie_trajectory_with_n_iter.nopython_signatures:
        logging.info("Running simulation once to trigger JIT compilation")
        _mod = TFNetworkModel(
            genotypes[0], seed=seeds[0], dt=dt, nt=nt, initialize=True
        )
        _pop0 = _mod.ssa.population_from_proteins(prot0s[0])
        _ = _mod.ssa.run_n_iter_with_params(_pop0, param_sets[0])
    logging.info("Proceeding with batch simulation")

    sampled_param_sets = np.zeros((batch_size, n_params))
    iters_per_timestep = np.zeros((batch_size,)).astype(np.int64)
    runtime_secs = np.zeros(batch_size).astype(np.float64)
    for i, (genotype, seed, prot0, params) in enumerate(
        zip(genotypes, seeds, prot0s, param_sets)
    ):
        model = TFNetworkModel(genotype, seed=seed, dt=dt, nt=nt, initialize=True)
        y0 = model.ssa.population_from_proteins(prot0)

        start_time = perf_counter()
        y_t, n_iter = model.ssa.run_n_iter_with_params(y0, params)
        end_time = perf_counter()
        sim_time = end_time - start_time

        sampled_param_sets[i] = convert_params_to_sampled_quantities(params)
        iters_per_timestep[i] = n_iter[-1] / n_iter.size
        runtime_secs[i] = sim_time

        logging.info(f" ===> {i+1} / {batch_size} took {sim_time:.3f} secs")

    logging.info("Completed batch simulation")

    n_iter_results = (
        sample_nums,
        genotype,
        sampled_param_sets,
        iters_per_timestep,
        runtime_secs,
    )

    if save:
        save_n_iter(
            n_iter_results=n_iter_results,
            save_dir=save_dir,
            param_names=param_names,
            ext=ext,
        )

    return n_iter_results


def save_n_iter(
    n_iter_results,
    save_dir,
    param_names,
    ext="parquet",
):
    (
        sample_nums,
        genotype,
        sampled_param_sets,
        iters_per_timestep,
        runtime_secs,
    ) = n_iter_results

    state_dir = Path(save_dir).joinpath(f"state_{genotype.strip('*')}")
    state_dir.mkdir(exist_ok=True)

    batch_size = len(sample_nums)
    data = dict(
        state=genotype,
        sample_num=sample_nums,
        runtime_secs=runtime_secs,
        iterations_per_timestep=iters_per_timestep,
    ) | dict(zip(param_names, sampled_param_sets.T))
    df = pd.DataFrame(data)
    df["state"] = df["state"].astype("category")

    fname = (
        state_dir.joinpath(f"batch{batch_size}_{uuid4()}.{ext}").resolve().absolute()
    )

    logging.info(f"Writing results to: {fname}")

    if ext == "csv":
        df.to_csv(fname, index=False)
    elif ext == "parquet":
        df.to_parquet(fname, index=False)
    else:
        raise ValueError(f"Unknown extension: {ext}")


def start_logging(log_dir, task_id=None):
    if task_id is None:
        task_id = uuid4().hex
    logger, logfile = _init_logger(task_id, log_dir)
    logger.info(f"Initialized logger for task {task_id}")
    logger.info(f"Logging to {logfile}")


def _init_logging(level=logging.INFO, mode="a"):
    fmt = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s --- %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(level)

    global _log_meta
    _log_meta = {"mode": mode, "fmt": fmt}


def _init_logger(task_id, log_dir: Path):
    logger = logging.getLogger()
    logger.handlers = []  # remove all handlers
    logfile = Path(log_dir).joinpath(f"task_{task_id}.log")
    fh = logging.FileHandler(logfile, mode=_log_meta["mode"])
    fh.setFormatter(_log_meta["fmt"])
    logger.addHandler(fh)
    return logger, logfile


def prompt_before_wiping_logs(log_dir):
    while True:
        decision = input(f"Delete all files in log directory?\n\t{log_dir}\n[Y/n]: ")
        if decision.lower() in ("", "y", "yes"):
            import shutil

            shutil.rmtree(log_dir)
            log_dir.mkdir()
            break
        elif decision.lower() in ("n", "no"):
            import sys

            print("Exiting...")
            sys.exit(0)
        else:
            print(f"Invalid input: {decision}")


def main(
    log_dir: Path,
    save_dir: Path,
    ttable_parquet: Path,
    batch_size: int = 100,
    nt: int = 2000,
    dt_seconds: float = 20.0,
    n_samples: int | None = 10_000,
    n_workers: Optional[int] = None,
    print_every: int = 50,
    seed_start: int = 0,
    shuffle_seed: Optional[int] = None,
    prompt_before_wipe: bool = True,
):
    if prompt_before_wipe and any(log_dir.iterdir()):
        prompt_before_wiping_logs(log_dir)

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
    sampled_param_names = [
        "log10_kd_1",
        "kd_2_1_ratio",
        "km_unbound",
        "km_act",
        "nlog10_km_rep_unbound_ratio",
        "kp",
        "gamma_m",
        "gamma_p",
    ]
    components = ["A", "B", "C"]
    interactions = ["activates", "inhibits"]
    root = "ABC::"
    tree = OscillationTree(
        components=components,
        interactions=interactions,
        root=root,
        dt=dt_seconds,
        nt=nt,
    )
    tree.grow_tree()

    if n_workers is None:
        n_workers = cpu_count(logical=True)

    # Cycle through nodes (in BFS order or shuffled)
    nodes_arr = np.array([n for n in tree.bfs_iterator() if tree.is_terminal(n)])
    if shuffle_seed is not None:
        rg = np.random.default_rng(shuffle_seed)
        rg.shuffle(nodes_arr)

    # Read in transposition table (record of all samples taken during search)
    ttable = pd.read_parquet(ttable_parquet)

    # Keep a record of the original row order, then shuffle row order
    ttable = ttable.reset_index(drop=True).reset_index(names="sample_num")
    ttable = ttable.sample(frac=1, random_state=shuffle_seed)

    # Order the rows such that each genotype is sampled in order, in batches of size
    # `batch_size`, but the contents of each batch are shuffled
    ttable["state"] = pd.Categorical(ttable["state"], categories=nodes_arr.tolist())
    ttable["batch_num"] = ttable.groupby("state").cumcount() // batch_size
    ttable = ttable.sort_values(["batch_num", "state"])

    # Start logging
    logging.basicConfig(filename=Path(log_dir).joinpath("main.log"), level=logging.INFO)
    _msg = (
        f"Using {n_workers} workers to sample each genotype in batches of size "
        f"{batch_size}. Sampling will continue indefinitely."
    )
    logging.info(_msg)
    print(_msg)

    # Iterate over parameter sets and random seeds for each batch
    run_batch_job = partial(
        run_batch_and_save,
        dt=dt_seconds,
        nt=nt,
        save_dir=save_dir,
        param_names=sampled_param_names,
        log_dir=log_dir,
    )
    iter_input_data = (
        df for _, df in ttable.groupby(np.arange(len(ttable)) // batch_size)
    )

    # _init_logging(level=logging.DEBUG)
    # for k, (*_, runtime_secs) in enumerate(map(run_batch_job, iter_input_data)):
    #     if k % print_every == 0:
    #         print(f"Finished {k} batches")
    #         logging.info(f"Finished {k} batches")
    #         print(f"Runtimes (seconds):")
    #         print("\t" + ", ".join([f"{r:.2f}" for r in runtime_secs]))

    with Pool(n_workers, initializer=_init_logging, initargs=(logging.DEBUG,)) as pool:
        k = 0
        start_time = perf_counter()
        for _ in pool.imap_unordered(run_batch_job, iter_input_data):
            k += 1
            if k % print_every == 0:
                end_time = perf_counter()
                _prog = (
                    f"Finished batch {k}. Took {end_time - start_time:.2f} seconds "
                    f"for {print_every} batches."
                )
                print(_prog)
                logging.info(_prog)
                start_time = end_time


if __name__ == "__main__":
    save_dir = Path("data/oscillation/n_iter")
    save_dir.mkdir(exist_ok=True)

    log_dir = Path("logs/oscillation/n_iter")
    log_dir.mkdir(exist_ok=True)

    ttable_parquet = Path("data/oscillation/230717_transposition_table_hpc.parquet")

    main(
        save_dir=save_dir,
        log_dir=log_dir,
        ttable_parquet=ttable_parquet,
        batch_size=10,
        nt=360,
        dt_seconds=20.0,
        n_workers=14,
        print_every=10,
        shuffle_seed=2023,
        # prompt_before_wipe=False,
    )
