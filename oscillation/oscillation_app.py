from celery import Celery
from celery.utils.log import get_task_logger
import h5py
import json
import numpy as np
from pathlib import Path
import redis
from time import perf_counter, sleep
from tf_network import TFNetworkModel

INT64_MAXVAL = np.iinfo(np.int64).max

app = Celery("tasks")
app.config_from_object("celeryconfig")
task_logger = get_task_logger(__name__)
database: redis.Redis = redis.Redis.from_url(app.conf["broker_url"], db=0)
print(app.conf["broker_url"])
n_param_sets = database.hlen("parameter_table")


@app.task(queue="simulations")
def run_ssa_no_time_limit(
    state: str,
    param_index: int,
    dt: float,  # seconds
    nt: int,
    nchunks: int,
    autocorr_threshold: float,
    save_dir: str,
) -> tuple[float, tuple[bool, float]]:
    task_logger.info(f"Received {param_index=}, {state=}")
    state_key = "state_" + state.strip("*")
    existing_entry = database.hget(state_key, str(param_index))
    task_logger.info(f"Existing entry: {existing_entry}")
    if existing_entry is not None:
        autocorr_min, sim_time = json.loads(existing_entry)
        if (autocorr_min <= 0.0) and (sim_time >= 0.0):
            task_logger.info(
                f"Entry already exists. Returning autocorr. min of {autocorr_min} "
                f"for {param_index=}, {state=}"
            )
            return autocorr_min, (True, sim_time)

    # Use the parameter set corresponding to the visit number
    seed, prots0, params, (mutated_component,) = json.loads(
        database.hget("parameter_table", str(param_index))
    )

    # Remove the mutated component from the state string
    if mutated_component != "(none)":
        components, interactions_joined = state.strip("*").split("::")
        where_mutated = components.find(mutated_component)
        components = components.replace(mutated_component, "")
        interactions = [
            ixn
            for ixn in interactions_joined.split("_")
            if mutated_component not in ixn[:2]
        ]
        interactions_joined = "_".join(interactions)
        new_state = f"*{components}::{interactions_joined}"
        task_logger.info(
            f"Removing component {mutated_component}:  {state} -> {new_state}"
        )
        sim_state = new_state
        prots0 = prots0[:where_mutated] + prots0[where_mutated + 1 :]
    else:
        sim_state = state

    kwargs = dict(
        seed=seed,
        prots0=prots0,
        params=params,
        dt=dt,
        nt=nt,
        max_iter_per_timestep=INT64_MAXVAL,  # no limit on reaction rate
        nchunks=nchunks,
        autocorr_threshold=autocorr_threshold,
        save_dir=save_dir,
    )

    # Run the simulation
    prots_t, autocorr_min, sim_time = _run_ssa(state=sim_state, **kwargs)
    autocorr_min = float(autocorr_min)
    sim_time = float(sim_time)
    kwargs["sim_time"] = sim_time
    kwargs["prots_t"] = prots_t.tolist()

    # Register the state-index pair and its result in the database
    task_logger.info(
        f"Finished in {sim_time:.4f} secs with autocorr. min: {autocorr_min:.4f}. "
        "Entering result in database."
    )
    database.sadd("transposition_table_keys", state_key)
    database.hset(state_key, str(param_index), json.dumps([autocorr_min, sim_time]))

    # Save data for any oscillating simulations
    oscillated = -autocorr_min > autocorr_threshold
    if oscillated:
        task_logger.info(
            f"Oscillation detected, saving data for {param_index=}, {state=}."
        )
        save_results.delay(
            prefix="osc_",
            param_index=param_index,
            autocorr_min=autocorr_min,
            state=state,
            sim_state=sim_state,
            **kwargs,
        )

    return autocorr_min, (False, sim_time)


def _run_ssa(
    seed: int,
    prots0: list[int],
    params: list[float],
    state: str,
    dt: float,
    nt: int,
    max_iter_per_timestep: int,
    nchunks: int,
    **kwargs,
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
        abs=False,
        nchunks=nchunks,  # simulation is run in chunks so it can be interrupted if needed
    )
    end = perf_counter()
    sim_time = end - start
    return prots_t, autocorr_min, sim_time


@app.task(queue="io", ignore_result=True)
def save_results(
    param_index: int,
    seed: int,
    state: str,
    sim_state: str,
    dt: float,
    nt: int,
    max_iter_per_timestep: int,
    autocorr_threshold: float,
    save_dir: str,
    prefix: str,
    autocorr_min: float = 1.0,
    sim_time: float = -1.0,
    prots_t: list[tuple[float, ...]] = None,
    **kwargs,
):
    fname = f"{prefix}state_{state}_index{param_index}.hdf5"
    fpath = Path(save_dir) / fname
    if fpath.exists():
        task_logger.info(f"File {fpath} already exists. Skipping.")
    if prots_t is None:
        prots_t = []
    task_logger.info(f"Saving to {fpath}")
    with h5py.File(fpath, "w") as f:
        f.create_dataset("y_t", data=prots_t)
        f.attrs["param_index"] = param_index
        f.attrs["autocorr_min"] = autocorr_min
        f.attrs["state"] = state
        f.attrs["sim_state"] = sim_state
        f.attrs["seed"] = seed
        f.attrs["dt"] = dt
        f.attrs["nt"] = nt
        f.attrs["max_iter_per_timestep"] = max_iter_per_timestep
        f.attrs["autocorr_threshold"] = autocorr_threshold
        f.attrs["sim_time"] = sim_time
