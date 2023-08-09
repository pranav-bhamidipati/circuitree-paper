from functools import partial
import h5py
from itertools import cycle, chain, repeat
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
from psutil import cpu_count
from time import perf_counter
from typing import Iterable, Optional
from uuid import uuid4

from oscillation.oscillation import TFNetworkModel, OscillationTree
from gillespie import convert_params_to_sampled_quantities


def run_batch_and_save(
    seed_and_genotype,
    batch_size,
    dt,
    nt,
    save_dir,
    param_names,
    log_dir,
    task_id=None,
    n_params: int = 8,
    ext: str = "parquet",
):
    start_logging(log_dir, task_id)

    seed, genotype = seed_and_genotype

    logging.info(f"Seed: {seed}")
    logging.info(f"Genotype: {genotype}")
    logging.info(f"Batch_size: {batch_size}")

    model = TFNetworkModel(genotype, seed=seed, dt=dt, nt=nt, initialize=True)

    param_sets = np.zeros((batch_size, n_params)).astype(np.float64)
    iters_per_timestep = np.zeros((batch_size,)).astype(np.int64)
    runtime_secs = np.zeros(batch_size).astype(np.float64)
    for i in range(batch_size):
        start_time = perf_counter()
        pop0, params, y_t, n_iter = model.ssa.run_n_iter_random()
        sampled_params = convert_params_to_sampled_quantities(params)
        end_time = perf_counter()
        sim_time = end_time - start_time

        iter_per_timestep = n_iter[-1] / n_iter.size

        param_sets[i] = sampled_params
        iters_per_timestep[i] = iter_per_timestep
        runtime_secs[i] = sim_time

        ...

        logging.info(f" ===> {i+1} / {batch_size} took {sim_time:.3f} secs")

    ...

    n_iter_results = seed, genotype, param_sets, iters_per_timestep, runtime_secs

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
    seed, genotype, param_sets, iters_per_timestep, runtime_secs = n_iter_results

    state_dir = Path(save_dir).joinpath(f"state_{genotype.strip('*')}")
    state_dir.mkdir(exist_ok=True)

    # n_iters = np.atleast_2d(n_iters)
    # n_iter_cols = [f"n_iter_{i}" for i in range(n_iters.shape[1])]

    data = (
        dict(state=genotype, seed=seed, runtime_secs=runtime_secs)
        | dict(zip(param_names, np.atleast_2d(param_sets).T))
        # | dict(zip(n_iter_cols, n_iters.T))
        | dict(iterations_per_timestep=iters_per_timestep)
    )
    df = pd.DataFrame(data)
    df["state"] = df["state"].astype("category")

    fname = state_dir.joinpath(f"{uuid4()}.{ext}").resolve().absolute()

    logging.info(f"Writing results to: {fname}")

    if ext == "csv":
        df.to_csv(fname, index=False)
    elif ext == "parquet":
        df.to_parquet(fname, index=False)
    else:
        raise ValueError(f"Unknown extension: {ext}")


# def save_pop_data(
#     pop_data_dir,
#     genotype,
#     y_t,
#     pop0s,
#     param_sets,
#     rewards,
#     init_columns,
#     param_names,
#     thresh,
# ):
#     save_idx = np.where(rewards > thresh)[0]
#     data = (
#         dict(state=genotype, reward=rewards[save_idx])
#         | dict(zip(init_columns, np.atleast_2d(pop0s[save_idx]).T))
#         | dict(zip(param_names, np.atleast_2d(param_sets[save_idx]).T))
#     )
#     df = pd.DataFrame(data)

#     state_no_asterisk = genotype.strip("*")
#     fname = pop_data_dir.joinpath(f"state_{state_no_asterisk}_ID#{uuid4()}.hdf5")
#     fname = fname.resolve().absolute()

#     logging.info(f"\tWriting all data for {len(save_idx)} runs to: {fname}")

#     with h5py.File(fname, "w") as f:
#         f.create_dataset("y_t", data=y_t[save_idx])
#     df.to_hdf(fname, key="metadata", mode="a", format="table")


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
    batch_size: int = 100,
    nt: int = 2000,
    dt_seconds: float = 20.0,
    n_samples: int | None = 10_000,
    n_workers: Optional[int] = None,
    print_every: int = 50,
    seed_start: int = 0,
    shuffle_seed: Optional[int] = None,
):
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

    # Split up the BFS into rounds
    # In each round, we will run n_samples simulations for each genotype.
    # Samples will be run and results saved in batches of size ``save_every``.
    # This ensures that workers with shorter simulations can steal work periodically.
    if n_workers is None:
        n_workers = cpu_count(logical=True)

    # Cycle through nodes in BFS order, taking n_batches_per_cycle batches of samples
    # from each node. n_cycles is set by balancing two factors. More cycles (fewer
    # samples per cycle) allows us to gradually accumulate data on all genotypes, rather
    # than one-by-one. However, it also means that for every cycle, we will end up JIT-
    # compiling the models again.
    bfs_arr = np.array([n for n in tree.bfs_iterator() if tree.is_terminal(n)])
    if shuffle_seed is not None:
        rg = np.random.default_rng(shuffle_seed)
        rg.shuffle(bfs_arr)

    # # Start logging
    # logging.basicConfig(filename=Path(log_dir).joinpath("main.log"), level=logging.INFO)

    # Run indefinitely or until a fixed number of samples
    if n_samples is None:
        bfs = cycle(bfs_arr.tolist())
        n_batches = None
        iter_seeds_and_genotypes = enumerate(cycle(bfs), start=seed_start)
        _msg = (
            f"Using {n_workers} workers to sample each genotype in batches of size "
            f"{batch_size}. Sampling will continue indefinitely."
        )
    else:
        n_batches, mod = divmod(n_samples, batch_size)
        if mod != 0:
            raise ValueError(
                f"n_samples ({n_samples}) must be divisible by batch_size ({batch_size})"
            )
        bfs = chain.from_iterable(repeat(bfs_arr.tolist(), n_batches))
        iter_seeds_and_genotypes = enumerate(
            chain.from_iterable(repeat(bfs, n_batches)), start=seed_start
        )
        _msg = (
            f"Using {n_workers} workers to make {n_samples} samples for each genotype "
            f"({n_batches} batches of size {batch_size})."
        )

    # logging.info(_msg)
    print(_msg)

    run_batch_job = partial(
        run_batch_and_save,
        batch_size=batch_size,
        dt=dt_seconds,
        nt=nt,
        save_dir=save_dir,
        param_names=sampled_param_names,
        log_dir=log_dir,
    )

    # for k, (seed, genotype, param_sets, n_iters, runtime_secs) in enumerate(
    #     map(run_batch_job, iter_seeds_and_genotypes)
    # ):
    #     if k % print_every == 0:
    #         print(f"Finished {k} batches")
    #         logging.info(f"Finished {k} batches")
    #         print(f"Runtimes (seconds):")
    #         print("\t" + ", ".join([f"{r:.2f}" for r in runtime_secs]))

    with Pool(n_workers, initializer=_init_logging, initargs=(logging.DEBUG,)) as pool:
        k = 0
        start_time = perf_counter()
        for seed, genotype, param_sets, n_iters, runtime_secs in pool.imap_unordered(
            run_batch_job, iter_seeds_and_genotypes
        ):
            k += 1
            if k % print_every == 0:
                end_time = perf_counter()
                _prog = f"Finished batch {k}. Took {end_time - start_time:.2f} seconds for {print_every} batches."
                print(_prog)
                logging.info(_prog)
                start_time = end_time


if __name__ == "__main__":
    save_dir = Path("data/oscillation/n_iter")
    save_dir.mkdir(exist_ok=True)

    log_dir = Path("logs/oscillation/n_iter")
    log_dir.mkdir(exist_ok=True)

    if any(log_dir.iterdir()):
        prompt_before_wiping_logs(log_dir)

    main(
        save_dir=save_dir,
        log_dir=log_dir,
        # n_samples=10000,
        batch_size=10,
        nt=360,
        dt_seconds=20.0,
        # n_workers=4,
        print_every=10,
        shuffle_seed=2023,
    )
