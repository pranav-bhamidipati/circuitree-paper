import os
from billiard.exceptions import TimeLimitExceeded
from celery import Celery, group
from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger
from celery.result import AsyncResult, GroupResult
from circuitree.parallel import TranspositionTable
import h5py
import numpy as np
from pathlib import Path
from redis.exceptions import ResponseError
import ssl
from time import perf_counter, sleep
from typing import Any, Optional


from oscillation_parallel import OscillationTreeParallel
from tf_network import TFNetworkModel

_redis_url = os.environ.get("CELERY_BROKER_URL", "")
if _redis_url:
    ssl_kwargs = dict(
        broker_use_ssl=ssl.CERT_NONE,
        redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    )
else:
    # Some apps use redis:// url, no SSL needed
    _redis_url = os.environ["CELERY_BROKER_URL_INTERNAL"]
    ssl_kwargs = dict()

# if not _redis_url:
#     _redis_url = Path(
#         "~/git/circuitree-paper/oscillation/celery/.redis-url"
#     ).expanduser()

app = Celery(
    "tasks",
    broker=_redis_url,
    backend=_redis_url,
    # broker_connection_retry_on_startup=False,
    broker_connection_retry_on_startup=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    task_compression="gzip",
    **ssl_kwargs,
)
app.conf["broker_transport_options"] = {
    "fanout_prefix": True,
    "fanout_patterns": True,
    "socket_keepalive": True,
}
task_logger = get_task_logger(__name__)


@app.task(soft_time_limit=120, time_limit=150)
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
    exist_ok: bool = True,
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

    except Exception as e:
        if isinstance(e, SystemError) or isinstance(e, SoftTimeLimitExceeded):
            reward = sim_time = -1.0
            _save_results(prefix="timeout_", **kwargs)
            return -1.0, -1.0
        else:
            raise


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
            _save_results(prefix="nan_", **save_kw)
    else:
        task_logger.info(f"Finished with autocorr. minimum: {autocorr_min:.4f}")
        reward = float(-autocorr_min > autocorr_threshold)
        if reward:
            task_logger.info("Oscillation detected, saving results.")
            _save_results(prefix="osc_", **save_kw)
        else:
            task_logger.info("No oscillations detected.")

    return reward, sim_time


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
        if exist_ok:
            return
        else:
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

    def compute_tasks_and_handle_errors(
        self, iter_args, n_retries=5, **kwargs
    ) -> tuple[list[float], list[float]]:
        task_group = group(self.run_task.s(*args, **kwargs) for args in iter_args)
        submitted = False
        retries_left = n_retries
        while True:
            try:
                if not submitted:
                    group_result: AsyncResult | GroupResult = task_group.delay()
                    submitted = True
                rewards, sim_times = self.compute_tasks_and_collect(group_result)
                break
            except ResponseError:
                if retries_left == 0:
                    self.logger.warning(
                        f"Received {n_retries}/{n_retries} ResponseErrors. Giving up."
                    )
                    raise
                else:
                    retries_left -= 1

                    # Exponential backoff
                    retry_wait = 2 ** (n_retries - retries_left) + np.random.rand()
                    self.logger.info(
                        f"Received ResponseError. Too many requests? Retrying "
                        f"in {retry_wait:.3f} seconds."
                    )
                    sleep(retry_wait)

        return rewards, sim_times

    def compute_tasks_and_collect(
        self, group_result: AsyncResult | GroupResult
    ) -> tuple[list[float], list[float]]:
        result_indices = {res.id: i for i, res in enumerate(group_result.results)}
        rewards = [np.nan] * self.batch_size
        sim_times = [np.nan] * self.batch_size
        try:
            for result, val in group_result.collect():
                if (
                    isinstance(val, tuple)
                    and len(val) == 2
                    and isinstance(val[0], float)
                ):
                    reward, sim_time = val
                    i = result_indices[result.id]
                    if reward >= 0:  # negative reward indicates soft timeout
                        rewards[i] = reward
                        sim_times[i] = sim_time
        except TimeLimitExceeded:  # hard timeout
            # Cancel the tasks that are still running
            n_cancelled = self.batch_size - group_result.completed_count()
            self.logger.info(
                f"Hard time limit exceeded, canceling {n_cancelled}/{self.batch_size} tasks."
            )
            group_result.revoke(terminate=True)

        return rewards, sim_times

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
        rewards, sim_times = self.compute_tasks_and_handle_errors(input_args, **kwargs)

        self.simtime_table[state].extend(list(sim_times))

        return rewards, {}
