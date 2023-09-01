from asyncio.exceptions import TimeoutError as AsyncTimeoutError
from circuitree.parallel import TranspositionTable
from dask.distributed import Client, LocalCluster, get_client, wait
import h5py
from functools import partial
import logging
import numpy as np
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional

from tf_network import TFNetworkModel
from oscillation_parallel import OscillationTreeParallel


def save_ttable_every_n_iters(tree: OscillationTreeParallel, *args, **kwargs):
    """Callback function to save the transposition table every n iterations."""
    tree.iteration_counter += 1
    i = tree.iteration_counter
    if i % tree.save_ttable_every == 0:
        ttable_fpath = Path(tree.save_dir) / f"iter{i}_trans_table.csv"
        sim_table_fpath = Path(tree.save_dir) / f"iter{i}_simtime_table.csv"
        logging.info(
            f"Saving transposition table and simulation times to {tree.save_dir}"
        )
        tree.ttable.to_csv(ttable_fpath, **kwargs)
        tree.simtime_table.to_csv(sim_table_fpath, **kwargs)


# def _init_logging(level=logging.INFO, mode="a"):
#     fmt = logging.Formatter(
#         "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s --- %(message)s"
#     )
#     logger = logging.getLogger()
#     logger.setLevel(level)

#     global _log_meta
#     _log_meta = {"mode": mode, "fmt": fmt}


# def start_logging(log_dir, task_id=None):
#     if task_id is None:
#         task_id = uuid4().hex
#     logger, logfile = _init_logger(task_id, log_dir)
#     logger.info(f"Initialized logger for task {task_id}")
#     logger.info(f"Logging to {logfile}")


# def _init_logger(task_id, log_dir: Path):
#     logger = logging.getLogger()
#     logger.handlers = []  # remove all handlers
#     logfile = Path(log_dir).joinpath(f"task_{task_id}.log")
#     fh = logging.FileHandler(logfile, mode=_log_meta["mode"])
#     fh.setFormatter(_log_meta["fmt"])
#     logger.addHandler(fh)
#     return logger, logfile


def run_ssa_and_save(
    seed: int,
    prots0: Iterable[int],
    params: Iterable[float],
    state: str,
    dt: float,
    nt: int,
    max_iter_per_timestep: int,
    autocorr_threshold: float,
    save_dir: Path,
    save_nans: bool,
    exist_ok: bool,
):
    # start_logging()
    logging.info(f"Initializing model and running SSA for  with {seed=}, {state=}")
    start = perf_counter()
    model = TFNetworkModel(
        genotype=state,
        seed=seed,
        dt=dt,
        nt=nt,
        max_iter_per_timestep=max_iter_per_timestep,
        initialize=True,
    )
    prots_t, autocorr_min = model.run_with_params_and_get_acf_minimum(
        prots0=np.array(prots0),
        params=np.array(params),
        maxiter_ok=True,
        abs=False,
    )
    end = perf_counter()
    sim_time = end - start
    logging.info(f"Finished SSA in {sim_time:.4f}s")

    # Handle results and save to data dir on worker-side
    save_kw = dict(
        seed=seed,
        state=state,
        dt=dt,
        nt=nt,
        max_iter_per_timestep=max_iter_per_timestep,
        autocorr_threshold=autocorr_threshold,
        save_dir=save_dir,
        exist_ok=exist_ok,
        autocorr_min=autocorr_min,
        sim_time=sim_time,
        prots_t=prots_t,
    )
    if np.isnan(autocorr_min):
        logging.info("Simulation returned NaNs; maxiter_per_timestep exceeded.")
        if save_nans:
            _save_nan_results(prefix="nan_", **save_kw)
    else:
        logging.info(f"Finished with autocorr. minimum: {autocorr_min:.4f}")
        if -autocorr_min > autocorr_threshold:
            logging.info("Oscillation detected, saving results.")
            _save_results(prefix="osc_", **save_kw)
        else:
            logging.info("No oscillations detected.")

    return autocorr_min, sim_time


def _save_results(
    seed: int,
    state: str,
    dt: float,
    nt: int,
    max_iter_per_timestep: int,
    autocorr_threshold: float,
    save_dir: Path,
    exist_ok: bool,
    prefix: str = "",
    autocorr_min: float | np.float64 = np.nan,
    sim_time: float | np.float64 = np.nan,
    prots_t: np.ndarray = np.array([]),
    **kwargs,
):
    fname = f"{prefix}state_{state}_seed{seed}.hdf5"
    fpath = Path(save_dir) / fname
    if fpath.exists():
        if not exist_ok:
            raise FileExistsError(fpath)
    logging.info(f"Saving results to {fpath}")
    with h5py.File(fpath, "w") as f:
        f.create_dataset("y_t", data=prots_t)
        f.attrs["autocorr_min"] = autocorr_min
        f.attrs["state"] = state
        f.attrs["seed"] = seed
        f.attrs["dt"] = dt
        f.attrs["nt"] = nt
        f.attrs["max_iter_per_timestep"] = max_iter_per_timestep
        f.attrs["autocorr_threshold"] = autocorr_threshold
        f.attrs["sim_time"] = sim_time


def _save_nan_results(
    seed: int,
    state: str,
    dt: float,
    nt: int,
    max_iter_per_timestep: int,
    prots_t: np.ndarray,
    sim_time: float | np.float64,
    save_dir: Path,
    exist_ok: bool,
    prefix: str = "",
    **kwargs,
):
    fname = f"{prefix}state_{state}_seed{seed}.hdf5"
    fpath = Path(save_dir) / fname
    if fpath.exists():
        if not exist_ok:
            raise FileExistsError(fpath)
    logging.info(f"Saving results to {fpath}")
    with h5py.File(fpath, "w") as f:
        f.create_dataset("y_t", data=prots_t)
        f.attrs["state"] = str(state)
        f.attrs["seed"] = int(seed)
        f.attrs["dt"] = float(dt)
        f.attrs["nt"] = int(nt)
        f.attrs["max_iter_per_timestep"] = int(max_iter_per_timestep)
        f.attrs["sim_time"] = float(sim_time)


class OscillationTreeDask(OscillationTreeParallel):
    def __init__(
        self,
        *,
        time_limit: float = 60.0,
        save_nans: bool = True,
        save_ttable_every: int = 25,
        sim_time_table: Optional[TranspositionTable] = None,
        client: Optional[Client] = None,
        cluster: Optional[LocalCluster] = None,
        cluster_kwargs: dict[str, Any] = {},
        exist_ok: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._simulation_time_table = TranspositionTable(sim_time_table)
        self.time_limit = time_limit

        self.iteration_counter = 0
        self.save_ttable_every = save_ttable_every

        self.add_done_callback(save_ttable_every_n_iters)

        self._cluster = LocalCluster(**cluster_kwargs) if cluster is None else cluster
        self._client = Client(self.cluster) if client is not None else get_client()
        self.n_procs = self.client.ncores()

        self.save_nans = save_nans
        self.exist_ok = exist_ok

        # Specify any attributes that should not be serialized when dumping to file
        self._non_serializable_attrs.extend(["_simulation_time_table", "_client"])

    @property
    def simtime_table(self):
        return self._simulation_time_table

    @property
    def cluster(self) -> Client:
        return self._cluster

    @property
    def client(self) -> Client:
        return self._client

    def save_results(self, data: dict[str, Any]) -> None:
        """Save results to the transposition table"""
        self.ttable[data["state"]].extend(data["reward"])

    @property
    def run_kwargs(self) -> dict[str, Any]:
        return dict(
            dt=float(self.dt),
            nt=int(self.nt),
            max_iter_per_timestep=int(self.max_iter_per_timestep),
            autocorr_threshold=float(self.autocorr_threshold),
            save_dir=str(self.save_dir),
            save_nans=self.save_nans,
            exist_ok=self.exist_ok,
        )

    def simulate_visits(self, state, visits) -> tuple[list[float], dict[str, Any]]:
        # Specify the keyword arguments
        do_task = partial(
            run_ssa_and_save,
            state=str(state),
            **self.run_kwargs,
        )

        # Submit the tasks using args that are JSON-serializable
        input_args = zip(*(self.param_table[v] for v in visits))

        # for v in visits:
        #     seed, inits, params = self.param_table[v]
        #     args = (int(seed), list(map(int, inits)), list(map(float, params)))
        #     res = self.client.apply_async(do_task, args)
        #     results.append(res)

        # Wait for the tasks to finish. Enforces an overall timeout
        futures = self.client.map(do_task, *input_args)
        try:
            wait(futures, timeout=self.time_limit)
            autocorr_mins, sim_times = zip(*[f.result() for f in futures])
        except AsyncTimeoutError:
            logging.info(f"Batch timed out after {self.time_limit}s")

            autocorr_mins = []
            sim_times = []
            for f, visit in zip(futures, visits):
                if f.status == "finished":
                    autocorr_min, sim_time = f.result()
                    autocorr_mins.append(autocorr_min)
                    sim_times.append(sim_time)
                else:
                    _save_nan_results(
                        seed=int(visit),
                        state=str(state),
                        prots_t=np.nan,
                        sim_time=np.nan,
                        prefix="timeout_",
                        **self.run_kwargs,
                    )
                    autocorr_mins.append(np.nan)
                    sim_times.append(np.nan)
                    f.cancel()

        rewards = []
        for amin in autocorr_mins:
            if np.isnan(amin):
                rewards.append(np.nan)
            else:
                rewards.append(float(-amin > self.autocorr_threshold))

        self.simtime_table[state].extend(list(sim_times))
        data = {"state": state, "reward": autocorr_mins}

        # print(f"Iteration {self.iteration_counter}")
        # print(f"State {state}")
        # print(f"{np.nanmin(autocorr_mins)=}")
        # print(f"{np.nanmax(autocorr_mins)=}")

        return rewards, data
