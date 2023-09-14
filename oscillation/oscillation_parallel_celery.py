from functools import cached_property
from billiard.exceptions import TimeLimitExceeded
from celery import Celery, Task, group
from celery.exceptions import SoftTimeLimitExceeded, TimeoutError
from celery.utils.log import get_task_logger
from celery.result import AsyncResult, GroupResult
from circuitree.parallel import TranspositionTable
import h5py
import numpy as np
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional

from oscillation_parallel import OscillationTreeParallel
from tf_network import TFNetworkModel

# from kombu.exceptions import OperationalError

# import os
# import ssl

# _redis_url = os.environ.get("CELERY_BROKER_URL", "")
# if _redis_url:
#     ssl_kwargs = dict(
#         broker_use_ssl=ssl.CERT_NONE,
#         redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
#     )
# else:
#     # Some apps use redis:// url, no SSL needed
#     _redis_url = os.environ["CELERY_BROKER_URL_INTERNAL"]
#     ssl_kwargs = dict()

# if not _redis_url:
#     _redis_url = Path(
#         "~/git/circuitree-paper/oscillation/celery/.redis-url"
#     ).expanduser()

app = Celery(
    "tasks",
    # broker=_redis_url,
    # backend=_redis_url,
    # broker_connection_retry_on_startup=False,
    # broker_connection_retry_on_startup=True,
    # worker_prefetch_multiplier=1,
    # worker_cancel_long_running_tasks_on_connection_loss=True,
    # task_compression="gzip",
    # **ssl_kwargs,
)
# app.conf["broker_transport_options"] = {
#     "fanout_prefix": True,
#     "fanout_patterns": True,
# "socket_keepalive": True,
# }
task_logger = get_task_logger(__name__)


@app.task(soft_time_limit=120, time_limit=300, queue="simulations")
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
        prots_t, autocorr_min, sim_time = _run_ssa(**kwargs)
        kwargs |= dict(
            autocorr_min=float(autocorr_min),
            sim_time=float(sim_time),
            prots_t=prots_t.tolist(),
        )

        if autocorr_min == 1.0:
            task_logger.info(
                "Simulation returned NaNs - maxiter_per_timestep exceeded."
            )
            reward = -1.0  # serializable, unlike np.nan
            if save_nans:
                _save_results.delay(prefix="nan_", **kwargs)
        else:
            task_logger.info(f"Finished with autocorr. minimum: {autocorr_min:.4f}")
            oscillated = -autocorr_min > autocorr_threshold
            reward = float(oscillated)
            if oscillated:
                task_logger.info("Oscillation detected, saving results.")
                _save_results.delay(prefix="osc_", **kwargs)
            else:
                task_logger.info("No oscillations detected.")

        return reward, sim_time

    except Exception as e:
        if isinstance(e, SystemError) or isinstance(e, SoftTimeLimitExceeded):
            task_logger.info(f"Soft time limit exceeded, writing metadata to file.")
            _save_results.delay(prefix="timeout_", **kwargs)
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
    **kwargs,
):
    start = perf_counter()
    task_logger.info("Initializing model and running SSA...")
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
        nchunks=5,  # simulation is split into 5 chunks - can be interrupted more easily
    )
    end = perf_counter()
    sim_time = end - start
    task_logger.info(f"Finished SSA in {sim_time:.4f}s")
    if np.isnan(autocorr_min):
        autocorr_min = 1.0  # positive value to indicate maxiter exceeded
    return prots_t, autocorr_min, sim_time


@app.task(queue="io", ignore_result=True)
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
    autocorr_min: float = 1.0,
    sim_time: float = -1.0,
    prots_t: list[tuple[float, ...]] = None,
    **kwargs,
):
    fname = f"{prefix}state_{state}_seed{seed}.hdf5"
    fpath = Path(save_dir) / fname
    if fpath.exists():
        if exist_ok:
            return
        else:
            raise FileExistsError(fpath)
    if prots_t is None:
        prots_t = []
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
        time_limit: int = 120,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._simulation_time_table = TranspositionTable(sim_time_table)
        self.iteration_counter = 0
        self.save_ttable_every = save_ttable_every
        self.time_limit = time_limit

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

        # Specify any attributes that should not be serialized when dumping to file
        self._non_serializable_attrs.extend(["_simulation_time_table", "logger"])

    @property
    def run_task(self) -> Task:
        return run_ssa

    @property
    def simtime_table(self):
        return self._simulation_time_table

    def save_results(self, data: dict[str, Any]) -> None:
        """Results are saved in the worker processes, so this method is a no-op."""
        return

    def compute_tasks_and_handle_errors(
        self, iter_args, n_retries: int = 5, **kwargs
    ) -> tuple[list[float], list[float]]:
        task_group = group(self.run_task.s(*args, **kwargs) for args in iter_args)
        group_result: GroupResult | AsyncResult = task_group.apply_async(
            retry=True, retry_policy={"max_retries": n_retries}
        )
        try:
            results = group_result.get(timeout=self.time_limit)
            rewards, sim_times = zip(*results)
        except Exception as e:
            # get() timed out or task received hard timeout
            if isinstance(e, (TimeoutError, TimeLimitExceeded)):
                rewards = []
                sim_times = []
                revoked = 0
                results: Iterable[AsyncResult] = group_result.results
                for result in results:
                    if result.ready():
                        reward, sim_time = result.get()
                        r = reward if reward >= 0 else np.nan
                        s = sim_time if reward >= 0 else np.nan
                    else:
                        # Cancel any running tasks - SIGINT signal is equivalent to
                        # KeyboardInterrupt and is a softer terminate signal than the
                        # default (SIGTERM)
                        revoked += 1
                        result.revoke(terminate=True, signal="SIGINT")
                        r = np.nan
                        s = np.nan
                    rewards.append(r)
                    sim_times.append(s)
                self.logger.info(f"Time limit exceeded, canceled {revoked} tasks.")

        return rewards, sim_times

    @cached_property
    def task_kwargs(self) -> dict[str, Any]:
        return dict(
            dt=float(self.dt),
            nt=int(self.nt),
            autocorr_threshold=float(self.autocorr_threshold),
            max_iter_per_timestep=int(self.max_iter_per_timestep),
            save_dir=str(self.save_dir),
        )

    def simulate_visits(self, state, visits) -> tuple[list[float], dict[str, Any]]:
        # Make the input args JSON serializable
        input_args = []
        for v in visits:
            seed, inits, params = self.param_table[v]
            input_args.append(
                (int(seed), list(map(int, inits)), list(map(float, params)))
            )

        # Submit the tasks and wait for them to finish, handling timeouts and retries
        rewards, sim_times = self.compute_tasks_and_handle_errors(
            input_args, state=str(state), **self.task_kwargs
        )
        self.simtime_table[state].extend(list(sim_times))

        return rewards, {}
