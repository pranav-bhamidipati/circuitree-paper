from celery import Celery, group
from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger
from circuitree.parallel import TranspositionTable
import h5py
import numpy as np
from pathlib import Path
import ssl
from time import perf_counter
from typing import Any, Optional

from oscillation_parallel import OscillationTreeParallel
from tf_network import TFNetworkModel


_redis_url = (
    Path("~/git/circuitree-paper/oscillation/celery/.redis-url")
    .expanduser()
    .read_text()
    .strip()
)
app = Celery(
    "tasks",
    broker=_redis_url,
    broker_use_ssl=ssl.CERT_NONE,
    backend=_redis_url,
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    # broker_connection_retry_on_startup=False,
    broker_connection_retry_on_startup=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    task_compression="gzip",
)
app.conf["broker_transport_options"] = {
    "fanout_prefix": True,
    "fanout_patterns": True,
    "socket_keepalive": True,
}
task_logger = get_task_logger(__name__)


@app.task(soft_time_limit=90)
def run_ssa(
    seed: int,
    prots0: list[int],
    params: list[float],
    state: str,
    dt: float,
    nt: int,
    max_iter_per_timestep: int,
    autocorr_threshold: float,
    save_dir: Path,
    save_nans: bool = True,
    exist_ok: bool = False,
):
    kwargs = dict(
        seed=seed,
        prots0=prots0,
        params=params,
        state=state,
        dt=dt,
        nt=nt,
        max_iter_per_timestep=max_iter_per_timestep,
        autocorr_threshold=autocorr_threshold,
        save_dir=save_dir,
        save_nans=save_nans,
        exist_ok=exist_ok,
    )
    try:
        task_logger.info(f"Running SSA with {seed=}, {state=}")
        reward, sim_time = _run_ssa(**kwargs)
        return reward, sim_time

    except SoftTimeLimitExceeded:
        reward = sim_time = -1.0
        _save_results(prefix="timeout_", **kwargs)
        return -1.0, -1.0


def _run_ssa(
    seed: int,
    prots0: list[int],
    params: list[float],
    state: str,
    dt: float,
    nt: int,
    max_iter_per_timestep: int,
    autocorr_threshold: float,
    save_dir: Path,
    save_nans: bool,
    exist_ok: bool,
):
    start = perf_counter()
    task_logger.info("Initializing model")
    model = TFNetworkModel(
        genotype=state,
        seed=seed,
        dt=dt,
        nt=nt,
        max_iter_per_timestep=max_iter_per_timestep,
        initialize=True,
    )
    task_logger.info("Running SSA...")
    prots_t, autocorr_min = model.run_with_params_and_get_acf_minimum(
        prots0=np.array(prots0),
        params=np.array(params),
        maxiter_ok=True,
        abs=False,
    )
    end = perf_counter()
    sim_time = end - start
    task_logger.info(f"Finished SSA in {sim_time:.4f}s")

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
        task_logger.info("Simulation returned NaNs - maxiter_per_timestep exceeded.")
        reward = -1.0  # serializable, unlike np.nan
        if save_nans:
            _save_results(prefix="nan_", autocorr_min=reward, **save_kw)
            prefix = "nan_"
    else:
        task_logger.info(f"Finished with autocorr. minimum: {autocorr_min:.4f}")
        reward = float(-autocorr_min > autocorr_threshold)
        if reward:
            save = True
            prefix = "osc_"
            task_logger.info("Oscillation detected, saving results.")
        else:
            task_logger.info("No oscillations detected.")

    if save:
        _save_results(
            model=model,
            seed=seed,
            autocorr_min=reward,
            sim_time=sim_time,
            prots_t=prots_t,
            autocorr_threshold=autocorr_threshold,
            save_dir=save_dir,
            prefix=prefix,
            exist_ok=exist_ok,
        )
    return reward, sim_time

    # Handle results and save to data dir on worker-side
    save_kw = dict(
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
    prefix: str,
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
    task_logger.info(f"Saving results to {fpath}")
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


def save_ttable_every_n_iters(tree: OscillationTreeParallel, *args, **kwargs):
    """Callback function to save the transposition table every n iterations."""
    tree.iteration_counter += 1
    tree.logger.info(f"Iteration {tree.iteration_counter} complete.")
    i = tree.iteration_counter
    if i % tree.save_ttable_every == 0:
        ttable_fpath = Path(tree.save_dir) / f"iter{i}_trans_table.csv"
        sim_table_fpath = Path(tree.save_dir) / f"iter{i}_simtime_table.csv"
        tree.logger.info(
            f"Saving transposition table and simulation times to {tree.save_dir}"
        )
        tree.ttable.to_csv(ttable_fpath, **kwargs)
        tree.simtime_table.to_csv(sim_table_fpath, **kwargs)


class OscillationTreeCelery(OscillationTreeParallel):
    def __init__(
        self,
        save_ttable_every: int = 10,
        sim_time_table: Optional[TranspositionTable] = None,
        logger: Any = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.run_task = run_ssa
        self._simulation_time_table = TranspositionTable(sim_time_table)

        self.iteration_counter = 0
        self.save_ttable_every = save_ttable_every

        if logger is None:
            self.logger = task_logger
        elif isinstance(logger, str) or isinstance(logger, Path):
            import logging

            self.logger = logging.getLogger(__name__)
            self.logger.handlers.clear()
            self.logger.setLevel(logging.INFO)
            log_fmt = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            logfile = Path(logger)
            fh = logging.FileHandler(logfile, mode="w")
            fh.setFormatter(log_fmt)
            self.logger.addHandler(fh)
        else:
            self.logger = logger

        self.add_done_callback(save_ttable_every_n_iters)

        # self.run_task.soft_time_limit = time_limit
        # self.time_limit = time_limit

        # # Specify any attributes that should not be serialized when dumping to file
        self._non_serializable_attrs.extend(["_simulation_time_table", "logger"])

    @property
    def simtime_table(self):
        return self._simulation_time_table

    def save_results(self, data: dict[str, Any]) -> None:
        """Results are saved in the worker processes, so this method is a no-op."""
        return

    def simulate_visits(self, state, visits) -> tuple[list[float], dict[str, Any]]:
        # Make the input args JSON serializable
        input_args = []
        for v in visits:
            seed, inits, params = self.param_table[v]
            input_args.append(
                (int(seed), list(map(int, inits)), list(map(float, params)))
            )

        # Specify the keyword arguments
        kwargs = dict(
            state=str(state),
            dt=float(self.dt),
            nt=int(self.nt),
            autocorr_threshold=float(self.autocorr_threshold),
            max_iter_per_timestep=int(self.max_iter_per_timestep),
            save_dir=str(self.save_dir),
        )

        # Submit the tasks as a group and wait for them to finish, with a timeout
        task_group = group(self.run_task.s(*args, **kwargs) for args in input_args)
        group_result = task_group.delay()
        results = group_result.get()
        rewards, sim_times = zip(*results)
        self.simtime_table[state].extend(list(sim_times))

        return rewards, {}