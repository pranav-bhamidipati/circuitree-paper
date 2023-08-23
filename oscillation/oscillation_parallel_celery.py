from celery import Celery
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
    task_compression="gzip",
)


def save_ssa_result_to_hdf(
    model: TFNetworkModel,
    seed: int,
    reward: float,
    y_t: np.ndarray,
    autocorr_threshold: float,
    sim_time: float,
    save_dir: Path,
    prefix: str = "",
) -> None:
    state = model.genotype
    fname = f"{prefix}state_{state}_uid_{uuid4()}.hdf5"
    fpath = Path(save_dir) / fname
    with h5py.File(fpath, "w") as f:
        f.create_dataset("y_t", data=y_t)
        f.attrs["reward"] = reward
        f.attrs["state"] = state
        f.attrs["seed"] = seed
        f.attrs["dt"] = model.dt
        f.attrs["nt"] = model.nt
        f.attrs["max_iter_per_timestep"] = model.max_iter_per_timestep
        f.attrs["autocorr_threshold"] = autocorr_threshold
        f.attrs["sim_time"] = sim_time


@app.task()
def _run_one_ssa(
    seed: int,
    prots0: np.ndarray,
    params: np.ndarray,
    state: str,
    dt: float,
    nt: int,
    max_iter_per_timestep: int,
    autocorr_threshold: float,
    save_dir: Path,
):
    # # For debugging
    # return reward

    start = perf_counter()
    model = TFNetworkModel(
        genotype=state,
        dt=dt,
        nt=nt,
        max_iter_per_timestep=max_iter_per_timestep,
        initialize=True,
    )
    prots_t, autocorr_min = model.run_with_params_and_get_acf_minimum(
        prots0=prots0, params=params, seed=seed, maxiter_ok=True, abs=True
    )
    end = perf_counter()
    sim_time = end - start

    reward = float(-autocorr_min > autocorr_threshold)

    # Handle results and save to data dir on worker-side
    save_ssa_result_to_hdf(
        model=model,
        seed=seed,
        reward=reward,
        y_t=prots_t,
        autocorr_threshold=autocorr_threshold,
        sim_time=sim_time,
        save_dir=save_dir,
    )
    return reward


class OscillationTreeCelery(OscillationTreeParallel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Specify any attributes that should not be serialized when dumping to file
        # self._non_serializable_attrs.append(...)

    def save_results(self, data: dict[str, Any]) -> None:
        """Results are saved in the worker processes, so this method is a no-op."""
        return

    def simulate_visits(self, state, visits) -> tuple[list[float], dict[str, Any]]:
        input_args = [self.param_table[v] for v in visits]
        submit_task = partial(
            _run_one_ssa.delay,
            state=state,
            dt=self.dt,
            nt=self.nt,
            max_iter_per_timestep=self.max_iter_per_timestep,
            autocorr_threshold=self.autocorr_threshold,
            save_dir=self.save_dir,
        )
        futures = [submit_task(*args) for args in input_args]
        rewards = [f.get() for f in futures]
        return rewards, {}
