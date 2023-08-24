from concurrent.futures import wait
from celery import Celery, group
from celery.exceptions import TimeoutError as CeleryTimeoutError
from functools import partial
import h5py
import numpy as np
from pathlib import Path
import ssl
from time import perf_counter
from typing import Any
from uuid import uuid4

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
    redis_max_connections=50,
    task_compression="gzip",
)


@app.task()
def _run_one_ssa(
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
    return seed  # for debugging

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
            if exist_ok:
                return
            else:
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


class OscillationTreeCelery(OscillationTreeParallel):
    def __init__(
        self,
        timeout: float = 60 * 10,  # seconds
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.worker_timeout = timeout

        # Specify any attributes that should not be serialized when dumping to file
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
            expires=self.worker_timeout,
        )

        # Submit the tasks as a group and wait for them to finish, with a timeout
        task_group = group(_run_one_ssa.s(*args, **kwargs) for args in input_args)
        results = task_group.apply_async()
        rewards = []
        for result in results:
            if result.ready():
                rewards.append(result.get())
            else:
                rewards.append(np.nan)

        return rewards, {}
