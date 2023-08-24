from celery import Celery, group
from celery.exceptions import SoftTimeLimitExceeded
import h5py
import numpy as np
from pathlib import Path
import ssl
from time import perf_counter, sleep
from typing import Any

from oscillation_parallel import OscillationTreeParallel
from tf_network import TFNetworkModel


_redis_url = (
    Path("/home/pbhamidi/git/circuitree-paper/oscillation/celery/.redis-url")
    .read_text()
    .strip()
)
app = Celery(
    "tasks",
    broker=_redis_url,
    broker_use_ssl=ssl.CERT_NONE,
    backend=_redis_url,
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    # broker_pool_limit=1,
    # broker_connection_retry_on_startup=False,
    broker_connection_retry_on_startup=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    broker_connection_max_retries=10,
    # redis_max_connections=1,
    task_compression="gzip",
)
app.conf["broker_transport_options"] = {
    "fanout_prefix": True,
    "fanout_patterns": True,
    "max_connections": 2,
    # "socket_keepalive": True,
}


@app.task()
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
    try:
        # for debugging
        sleep(20)
        return seed

        reward = _run_ssa(
            seed,
            prots0,
            params,
            state,
            dt,
            nt,
            max_iter_per_timestep,
            autocorr_threshold,
            save_dir,
            save_nans,
            exist_ok,
        )
        return reward

    except SoftTimeLimitExceeded:
        return np.nan


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
        abs=True,
    )
    end = perf_counter()
    sim_time = end - start

    # Handle results and save to data dir on worker-side
    state = model.genotype

    save = False
    prefix = ""
    if np.isnan(autocorr_min):
        if save_nans:
            save = True
            prefix = "nan_"
        reward = np.nan
    else:
        reward = float(-autocorr_min > autocorr_threshold)
        if reward:
            save = True
            prefix = "osc_"

    if save:
        fname = f"{prefix}state_{state}_seed{seed}.hdf5"
        fpath = Path(save_dir) / fname
        if fpath.exists():
            if not exist_ok:
                raise FileExistsError(fpath)
        with h5py.File(fpath, "w") as f:
            f.create_dataset("y_t", data=prots_t)
            f.attrs["reward"] = reward
            f.attrs["state"] = state
            f.attrs["seed"] = seed
            f.attrs["dt"] = model.dt
            f.attrs["nt"] = model.nt
            f.attrs["max_iter_per_timestep"] = model.max_iter_per_timestep
            f.attrs["autocorr_threshold"] = autocorr_threshold
            f.attrs["sim_time"] = sim_time

    return reward


class OscillationTreeCelery(OscillationTreeParallel):
    def __init__(
        self,
        time_limit: int = 600,  # seconds
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.run_task = run_ssa
        self.run_task.soft_time_limit = time_limit
        # self.time_limit = time_limit

        # # Specify any attributes that should not be serialized when dumping to file
        # self._non_serializable_attrs.append(...)

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
        task_group = group(
            [run_ssa.s(*args, **kwargs) for args in input_args],
            # soft_time_limit=self.time_limit,
        )
        print(
            f"Submitting {len(input_args)} tasks to Celery with time limit {self.time_limit}s."
        )
        group_result = task_group.delay()
        rewards = group_result.get()
        print(f"Got results: {rewards=}")

        raise NotImplementedError("Finished")

        return rewards, {}
