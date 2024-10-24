import datetime
from functools import partial
import h5py
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
from psutil import cpu_count
from time import perf_counter
from typing import Literal, Optional
from uuid import uuid4

from gillespie import PARAM_NAMES
from tf_network import TFNetworkModel
from oscillation import OscillationTree


def run_batch_and_save(
    args,
    dt,
    nt,
    save_dir,
    log_dir,
    task_id=None,
    ext="csv",
    nchunks=5,
):
    start_time = perf_counter()

    start_logging(log_dir, task_id)

    (
        seeds,
        genotype,
        indices,
        batch_init_condition,
        batch_param_sets,
    ) = args

    batch_num = min(indices) // len(indices)

    logging.info(f"Batch #: {batch_num}")
    logging.info(f"Param set(s): {min(indices)} - {max(indices)}")
    logging.info(f"Genotype: {genotype}")

    start_sim_time = perf_counter()
    model = TFNetworkModel(genotype)
    prots_t, acf_minima = model.run_ssa_asymmetric_and_get_acf_minima(
        dt=dt,
        nt=nt,
        init_proteins=batch_init_condition,
        params=batch_param_sets,
        seed=seeds,
        nchunks=nchunks,
        freqs=False,
        indices=False,
        abs=False,
    )

    end_sim_time = perf_counter()
    sim_time = end_sim_time - start_sim_time
    logging.info(f"Simulation took {sim_time:.2f}s ")

    save_run(
        genotype, indices, prots_t, acf_minima, sim_time, save_dir=save_dir, ext=ext
    )
    end_time = perf_counter()
    total_time = end_time - start_time
    logging.info(f"Total time {total_time:.2f}s")

    return batch_num, genotype, sim_time, total_time


def save_run(
    genotype,
    param_indices,
    prots_t,
    acf_minima,
    sim_time,
    save_dir,
    ext="csv",
):
    start = min(param_indices)
    stop = max(param_indices)
    descriptor = f"{start}-{stop}_state_{genotype.strip('*')}"

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    results_file = save_dir.joinpath(f"results/{descriptor}.{ext}")
    data_file = save_dir.joinpath(f"data/{descriptor}.hdf5")

    df = pd.DataFrame(
        dict(
            state=genotype,
            param_idx=param_indices,
            acf_min=acf_minima,
            simulation_time_seconds=sim_time,
            data_file=data_file.name,
        )
    )
    df["state"] = df["state"].astype("category")

    logging.info(f"Writing results to: {results_file.resolve().absolute()}")

    results_file.parent.mkdir(exist_ok=True)
    if ext == "csv":
        df.to_csv(results_file, index=False)
    elif ext == "parquet":
        df.to_parquet(results_file, index=False)

    logging.info(f"Writing data to: {data_file.resolve().absolute()}")

    data_file.parent.mkdir(exist_ok=True)
    with h5py.File(str(data_file), "w") as f:
        f.create_dataset("prots_t", data=prots_t)
    df.to_hdf(data_file, key="metadata", mode="a", format="table")


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
            return True
        elif decision.lower() in ("n", "no"):
            import sys

            print("Exiting...")
            return False
        else:
            print(f"Invalid input: {decision}")


def generate_batched_args(
    init_conditions,
    param_sets,
    genotypes,
    batch_size,
    root_seed,
    seeds,
    samples_per_genotype=None,
):
    """
    Returns a generator that yields tuples of:

        (batch_seeds, genotype, indices, batch_init_condition, batch_param_sets)

    If ``maxiter`` is specified, it cycles through ``genotypes``, drawing samples
    until ``maxiter`` is reached. Otherwise, it runs indefinitely. Each call
    generates a batch of input arguments for simulations of a single genotype. Each
    simulation is seeded uniquely with a sequence of integers of the form

        (param_idx, genotype_idx, root_seed)

    """
    param_idx = 0
    while samples_per_genotype is None or param_idx < samples_per_genotype:
        indices = np.arange(param_idx, param_idx + batch_size)
        batch_init_condition = init_conditions[indices]
        batch_param_sets = param_sets[indices]

        for g_idx, genotype in enumerate(genotypes):
            batch_seeds = [(p_idx, g_idx, root_seed) for p_idx in indices]
            yield (
                batch_seeds,
                genotype,
                indices,
                batch_init_condition,
                batch_param_sets,
            )
        param_idx += batch_size


def main(
    param_sets_csv: Path,
    save_dir: Path,
    log_dir: Path,
    batch_size: int = 100,
    nt: int = 2160,
    dt_seconds: float = 20.0,
    n_samples: Optional[int] = None,
    n_workers: Optional[int] = None,
    print_every: int = 50,
    # oscillation_thresh: float = 0.35,
    root_seed: int = 0,
    results_file_ext: Literal["csv", "parquet"] = "csv",
    nchunks: int = 5,
):
    if any(log_dir.iterdir()):
        run_main = prompt_before_wiping_logs(log_dir)
        if not run_main:
            return

    # Define the components and interactions of the system
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

    # Enumerate the search space by "growing" the tree of possible genotypes
    tree.grow_tree()

    # Define the initial columns and parameter names for output dataframes
    init_columns = [f"{c}_0" for c in components]
    param_names = [f"{c}_{p}" for c in components for p in PARAM_NAMES]

    if n_workers is None:
        n_workers = cpu_count(logical=True)

    genotypes = np.array(
        [n for n in tree.bfs_iterator() if tree.grammar.is_terminal(n)]
    )

    # Start logging
    logging.basicConfig(filename=Path(log_dir).joinpath("main.log"), level=logging.INFO)

    if n_samples is None:
        _msg = (
            f"Using {n_workers} workers to sample each genotype in batches of size "
            f"{batch_size}. Sampling will continue indefinitely."
        )
    else:
        if n_samples % batch_size != 0:
            raise ValueError(
                f"n_samples={n_samples} must be divisible by batch_size={batch_size}"
            )
        _msg = (
            f"Using {n_workers} workers to take {n_samples} samples of each genotype, "
            f"in batches of size {batch_size}."
        )

    # Read in the table of parameter sets and initial conditions
    param_table = pd.read_csv(param_sets_csv)
    n_param_sets = len(param_table)
    seeds = param_table["prng_seed"].values.astype(int)
    init_conditions = param_table[init_columns].values.astype(int)
    param_sets = (
        param_table[param_names].values.astype(float).reshape(n_param_sets, 3, -1)
    )

    batched_args = generate_batched_args(
        init_conditions,
        param_sets,
        genotypes,
        batch_size,
        root_seed,
        seeds,
        samples_per_genotype=n_samples,
    )

    logging.info(_msg)
    print(_msg)

    run_one_job = partial(
        run_batch_and_save,
        dt=dt_seconds,
        nt=nt,
        save_dir=save_dir,
        log_dir=log_dir,
        ext=results_file_ext,
        nchunks=nchunks,
    )

    n_batches_total = n_samples * len(genotypes) // batch_size
    pool_start_time = perf_counter()

    if n_workers == 1:
        _init_logging(logging.INFO)
        k = 0
        for args in batched_args:
            k += 1
            batch_num, genotype, simulation_time, total_time = run_one_job(args)
            if k % print_every == 0:
                pool_elapsed = perf_counter() - pool_start_time
                estimated_wait = (pool_elapsed / k) * (n_batches_total - k)
                _msg = (
                    f"{datetime.timedelta(seconds=pool_elapsed)} -- "
                    f"Finished batch: {k} of {n_batches_total} ({k/n_batches_total:.2%}) -- "
                    f"Estimated time remaining: {datetime.timedelta(seconds=estimated_wait)} -- "
                    f"Avg time per simulation: {pool_elapsed * n_workers/(k * batch_size):.2f}s"
                )
                print(_msg)
                logging.info(_msg)

            logging.info(
                f"Batch {batch_num} took {total_time:.2f}s"
                f" ({simulation_time:.2f}s of simulation) for state {genotype}"
            )
    else:
        with Pool(
            n_workers, initializer=_init_logging, initargs=(logging.INFO,)
        ) as pool:
            k = 1
            for batch_num, genotype, simulation_time, total_time in pool.imap_unordered(
                run_one_job, batched_args
            ):
                if k % print_every == 0:
                    pool_elapsed = perf_counter() - pool_start_time
                    estimated_wait = (pool_elapsed / k) * (n_batches_total - k)
                    _msg = (
                        f"{datetime.timedelta(seconds=pool_elapsed)} -- "
                        f"Finished batch: {k} of {n_batches_total} ({k/n_batches_total:.2%}) -- "
                        f"Estimated time remaining: {datetime.timedelta(seconds=estimated_wait)} -- "
                        f"Avg time per simulation: {pool_elapsed * n_workers/(k * batch_size):.2f}s"
                    )
                    print(_msg)
                    logging.info(_msg)

                logging.info(
                    f"Batch {batch_num} took {total_time:.2f}s"
                    f" ({simulation_time:.2f}s of simulation) for state {genotype}"
                )
                k += 1

    pool_total_time = perf_counter() - pool_start_time
    time_per_sim = (pool_total_time * n_workers) / (n_samples * len(genotypes))
    print(
        f"""DONE!
            -- Total time elapsed: {datetime.timedelta(seconds=pool_total_time)}
            -- Avg time per simulation: {time_per_sim:.2f}s
        """
    )
    logging.info(_msg)


if __name__ == "__main__":

    run_name = "exhaustive_small_with_sim_time"
    param_sets_csv = Path(
        "~/git/circuitree-paper/data/oscillation_asymmetric_params/"
        "241017_param_sets_100_3tf_small.csv"
    )
    today = datetime.datetime.now().strftime("%y%m%d")

    save_dir = Path(f"data/oscillation_asymmetric_params/exhaustive/{today}_{run_name}")
    save_dir.mkdir(exist_ok=True)

    log_dir = Path(f"logs/oscillation_asymmetric_params/exhaustive/{today}_{run_name}")
    log_dir.mkdir(exist_ok=True)
    main(
        param_sets_csv=param_sets_csv,
        save_dir=save_dir,
        log_dir=log_dir,
        n_samples=100,
        batch_size=20,
        nt=2160,
        dt_seconds=20.0,
        n_workers=186,
        nchunks=5,
        print_every=50,
        # oscillation_thresh=0.35,
    )
