from celery import Celery
from celery.utils.log import get_task_logger
import h5py
import json
import numpy as np
from pathlib import Path
import pandas as pd
import redis
from time import perf_counter, sleep
from tf_network import TFNetworkModel

INT64_MAXVAL = np.iinfo(np.int64).max

param_sets_csv = Path(
    "~/git/circuitree-paper/data/oscillation_asymmetric_params/"
    # "241023_param_sets_10000_5tf.csv"
    # "241023_param_sets_10000_3tf.csv"
    "241025_param_sets_10000_5tf_pmut0.5.csv"
).expanduser()
param_sets_csv = str(param_sets_csv.resolve().absolute())

app = Celery("tasks")
app.config_from_object("celeryconfig")
task_logger = get_task_logger(__name__)
database: redis.Redis = redis.Redis.from_url(app.conf["broker_url"], db=0)
print("Database address:", app.conf["broker_url"])


def read_params_from_table(
    row: int, n_components: int, param_sets_csv: str = param_sets_csv
) -> pd.DataFrame:

    # Use the parameter set corresponding to the visit number
    param_data = pd.read_csv(param_sets_csv).iloc[row]
    if "param_index" in param_data:
        del param_data["param_index"]

    # Extract whether a component is to be mutated (removed)
    if "component_mutation" in param_data:
        mutated_component = param_data.pop("component_mutation")
    else:
        mutated_component = "(none)"

    # Extract the random seed from the parameter set. If a root seed is provided, prepend
    # it to the seed from the parameter set.
    seed = int(param_data.pop("prng_seed").item())

    # Extract the initial protein counts and parameter values
    prots0 = param_data[:n_components].tolist()
    params = param_data[n_components:].values.reshape(n_components, -1).tolist()

    return prots0, params, seed, mutated_component


@app.task
def run_ssa_no_time_limit(
    state: str,
    param_index: int,
    dt: float,  # seconds
    nt: int,
    nchunks: int,
    autocorr_threshold: float,
    save_dir: str,
    param_sets_csv: str = param_sets_csv,
    root_seed: int | None = None,
) -> tuple[float, tuple[bool, float]]:
    task_logger.info(f"Received {param_index=}, {state=}")
    state_key = "state_" + state.strip("*")
    existing_entry = database.hget(state_key, str(param_index))
    if existing_entry is not None:
        autocorr_min, sim_time = json.loads(existing_entry)
        if (autocorr_min <= 0.0) and (sim_time >= 0.0):
            task_logger.info(
                f"Entry already exists. Returning autocorr. min of {autocorr_min} "
                f"for {param_index=}, {state=}"
            )
            return autocorr_min, (True, sim_time)

    # Parse the parameter set from the CSV file
    n_components = len(state.split("::")[0].strip("*"))
    prots0, params, seed, mutated_component = read_params_from_table(
        param_index, n_components, param_sets_csv
    )
    seed_phrase = (root_seed, seed) if root_seed is not None else (seed,)

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
        seed=seed_phrase,
        prots0=prots0,
        params=params,
        dt=dt,
        nt=nt,
        nchunks=nchunks,
        autocorr_threshold=autocorr_threshold,
        save_dir=save_dir,
    )

    # Run the simulation
    prots_t, autocorr_min, sim_time = _run_ssa_asymmetric(state=sim_state, **kwargs)
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
    oscillated = autocorr_min < autocorr_threshold
    if oscillated:
        task_logger.info(
            f"Oscillation detected, saving data for {param_index=}, {state=}."
        )
    else:
        task_logger.info(
            f"Oscillation not detected, saving data for {param_index=}, {state=}."
        )
    save_results.delay(
        prefix="data_",
        param_index=param_index,
        autocorr_min=autocorr_min,
        state=state,
        sim_state=sim_state,
        **kwargs,
    )

    return autocorr_min, (False, sim_time)


def _run_ssa_asymmetric(
    seed: int | tuple[int, int],
    prots0: list[int],
    params: list[float],
    state: str,
    dt: float,
    nt: int,
    nchunks: int,
    **kwargs,
):
    start = perf_counter()
    model = TFNetworkModel(genotype=state)
    prots_t, autocorr_min = model.run_ssa_asymmetric_and_get_acf_minima(
        dt=dt,
        nt=nt,
        init_proteins=np.array(prots0),
        params=np.array(params),
        seed=seed,
        nchunks=nchunks,
    )
    end = perf_counter()
    sim_time = end - start
    return prots_t, float(autocorr_min), float(sim_time)


@app.task(ignore_result=True)
def save_results(
    param_index: int,
    seed: int,
    state: str,
    sim_state: str,
    dt: float,
    nt: int,
    autocorr_threshold: float,
    save_dir: str,
    prefix: str,
    autocorr_min: float = 1.0,
    sim_time: float = -1.0,
    prots_t: list[tuple[float, ...]] = None,
    **kwargs,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    save_dir.joinpath("data").mkdir(exist_ok=True)
    fname = f"{prefix}state_{state}_index{param_index}.hdf5"
    fpath = save_dir / "data" / fname
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
        f.attrs["autocorr_threshold"] = autocorr_threshold
        f.attrs["sim_time"] = sim_time
