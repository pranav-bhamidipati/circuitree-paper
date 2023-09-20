from billiard.exceptions import TimeLimitExceeded
from celery import Celery, group
from celery.exceptions import SoftTimeLimitExceeded, TimeoutError
from celery.utils.log import get_task_logger
from celery.result import AsyncResult, GroupResult
from circuitree.parallel import ParallelTree
from collections import Counter
from functools import partial
import h5py
import json
import numpy as np
from pathlib import Path
import redis
from time import perf_counter
from typing import Iterable, Optional

from oscillation import OscillationTree
from tf_network import TFNetworkModel


########################################################################################
### A problem I am running into is that the Render Redis has too much latency and
### cannot be easily backed up/converted into a dataframe. So I made my own Redis that
### runs locally on the AWS instance.
###
### This time I will use Celery to distribute the tasks to the workers, and the
### same Redis database used by the workers will be used to store the transposition
### table and the parameter table.
###
### Data structures in the Redis database
### =====================================
###
###     parameter_table
###         A hash table that maps a parameter key (= visit number) to the corresponding
###         parameter set, stored as a JSON string.
###
###     transposition_table_keys
###         A set of terminal states that have been visited. Keys are equivalent to the
###         state strings shown below. This is used to keep track of which states have
###         been visited and store which keys in the database store the transposition
###         table, as opposed to other things such as Celery data structures.
###
###     state_[terminal state] (e.g. "state_*ABCDE::")
###         Each state string points to a Redis linked list of reward values. New rewards
###         are pushed to the list, and the length of the list is used to determine the
###         total number of visits to the state.
###
###         When a worker receives a state-visit pair, it checks if the state is in the
###         transposition table. If it is, it checks if the visit number is less than the
###         length of the list. If it is, then it gets the reward from the list. Otherwise,
###         then it runs the simulation, appends the reward to the list, saves any pertinent
###         (meta)data, and returns the reward.
###
########################################################################################


# app = Celery(
#     "tasks",
#     broker=_redis_url,
#     backend=_redis_url,
#     broker_connection_retry_on_startup=False,
#     broker_connection_retry_on_startup=True,
#     worker_prefetch_multiplier=1,
#     worker_cancel_long_running_tasks_on_connection_loss=True,
#     task_compression="gzip",
#     **ssl_kwargs,
# )
# app.conf["broker_transport_options"] = {
#     "fanout_prefix": True,
#     "fanout_patterns": True,
# "socket_keepalive": True,
# }

INT64_MAXVAL = np.iinfo(np.int64).max

app = Celery("tasks")
app.config_from_object("celeryconfig")
task_logger = get_task_logger(__name__)
database = redis.Redis(host="localhost", port=6379, db=0)
n_param_sets = database.hlen("parameter_table")

dt = 20.0  # seconds
nt = 2000
max_iter_per_timestep = 100_000_000
nchunks = 5
autocorr_threshold = 0.4

save_dir = (
    Path("~/git/circuitree-paper/data/oscillation/mcts/230908_3tf_localredis")
    .expanduser()
    .as_posix()
)


@app.task(queue="cleanup", ignore_result=True)
def run_ssa_nolimit(
    state: str,
    seed: int,
):
    """Run SSA without a time limit."""
    task_logger.info(f"Received {seed=}, {state=}")
    state_key = "state_" + state.strip("*")
    existing_entry = database.hget(state_key, str(seed))
    if existing_entry is not None:
        autocorr_min, sim_time = json.loads(existing_entry)
        if (autocorr_min <= 0.0) and (sim_time > 0.0):
            task_logger.info(f"Entry already exists. Skipping {seed=}, {state=}")
            return

    # Use the parameter set corresponding to the random seed
    prots0, params = json.loads(database.hget("parameter_table", str(seed)))
    kwargs = dict(
        seed=seed,
        prots0=prots0,
        params=params,
        state=state,
        dt=dt,
        nt=nt,
        max_iter_per_timestep=INT64_MAXVAL,
        autocorr_threshold=autocorr_threshold,
        save_dir=save_dir,
    )

    prots_t, autocorr_min, sim_time = _run_ssa(**kwargs)
    sim_time = float(sim_time)
    kwargs["sim_time"] = sim_time
    kwargs["prots_t"] = prots_t.tolist()

    autocorr_min = float(autocorr_min)
    task_logger.info(
        f"Finished in {sim_time:.4f} secs with autocorr. min: {autocorr_min:.4f}. "
        "Entering data in database."
    )
    database.hset(state_key, str(seed), json.dumps([autocorr_min, sim_time]))

    oscillated = -autocorr_min > autocorr_threshold
    if oscillated:
        task_logger.info("Oscillation detected, saving results.")
        save_results.delay(prefix="osc_", autocorr_min=autocorr_min, **kwargs)
    else:
        task_logger.info("No oscillations detected.")


@app.task(soft_time_limit=120, time_limit=300, queue="simulations")
def run_ssa(
    state: str,
    seed: int,
):
    task_logger.info(f"Received {seed=}, {state=}")
    # Use the parameter set corresponding to the visit number
    prots0, params = json.loads(database.hget("parameter_table", str(seed)))
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
    )

    try:
        prots_t, autocorr_min, sim_time = _run_ssa(**kwargs)
        sim_time = float(sim_time)
        kwargs["sim_time"] = sim_time
        kwargs["prots_t"] = prots_t.tolist()

        if np.isnan(autocorr_min):
            autocorr_min = 1.0
            task_logger.info(
                "Simulation returned NaNs - maxiter_per_timestep exceeded."
            )
            save_results.delay(prefix="nan_", autocorr_min=autocorr_min, **kwargs)
        else:
            autocorr_min = float(autocorr_min)
            task_logger.info(
                f"Finished in {sim_time:.4f} secs with autocorr. min: {autocorr_min:.4f}"
            )
            oscillated = -autocorr_min > autocorr_threshold
            if oscillated:
                task_logger.info("Oscillation detected, saving results.")
                save_results.delay(prefix="osc_", autocorr_min=autocorr_min, **kwargs)
            else:
                task_logger.info("No oscillations detected.")

        return autocorr_min, sim_time

    except Exception as e:
        if isinstance(e, (SystemError, SoftTimeLimitExceeded)):
            task_logger.info(f"Soft time limit exceeded, writing metadata to file.")
            save_results.delay(prefix="timeout_", **kwargs)
            return 1.0, -1.0
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
        nchunks=nchunks,  # simulation is split into 5 chunks - can be interrupted more easily
    )
    end = perf_counter()
    sim_time = end - start
    return prots_t, autocorr_min, sim_time


@app.task(queue="io", ignore_result=True)
def save_results(
    seed: int,
    state: str,
    dt: float,
    nt: int,
    max_iter_per_timestep: int,
    autocorr_threshold: float,
    save_dir: Path,
    prefix: str,
    autocorr_min: float = 1.0,
    sim_time: float = -1.0,
    exist_ok: bool = True,
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
    task_logger.info(f"Saving to {fpath}")
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


class OscillationTreeCeleryLocal(OscillationTree, ParallelTree):
    """ """

    def __init__(
        self,
        *args,
        time_limit: int = 120,
        callback_every: int = 10,
        counter: Optional[Counter] = None,
        n_param_sets: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.time_limit = time_limit
        self.visit_counter = Counter(counter)
        self.logger = task_logger

        self.iteration_counter = 0
        self.callback_every = callback_every

        if n_param_sets is None:
            self.n_param_sets: int = database.hlen("parameter_table")

        # Specify attributes that should not/cannot be serialized when dumping to file
        self._non_serializable_attrs.extend(
            ["visit_counter", "logger", "iteration_counter"]
        )

    def get_reward(self, state: str):
        visit0 = self.visit_counter[state]
        seeds = (visit0 + np.arange(self.batch_size)) % self.n_param_sets

        # Check if the state is in the transposition table and pull any rewards
        state_key = "state_" + state.strip("*")  # asterisk is interpreted as wildcard
        state_in_ttable = bool(
            database.sismember("transposition_table_keys", state_key)
        )
        autocorr_mins = -np.ones(self.batch_size, dtype=np.float64)
        which_to_simulate = np.ones(self.batch_size, dtype=bool)
        if state_in_ttable:
            past_results = database.hmget(state_key, seeds.tolist())
            for i, res in enumerate(past_results):
                in_table = res is not None
                which_to_simulate[i] = not in_table
                if in_table:
                    # autocorr_min should be nonpositive. A positive value means the
                    # simulation returned NaN (maxiter exceeded or timeout)
                    autocorr_min, sim_time = json.loads(res)
                    if autocorr_min > 0.0:
                        autocorr_mins[i] = np.nan

        # Dispatch simulation tasks
        n_to_simulate = which_to_simulate.sum()
        new_autocorr_mins = np.ones(n_to_simulate, dtype=np.float64)
        new_sim_times = -np.ones(n_to_simulate, dtype=np.float64)
        group_result: AsyncResult | GroupResult = group(
            run_ssa.s(state, int(seed)) for seed in seeds[which_to_simulate]
        ).apply_async()
        if n_to_simulate:
            try:
                new_results = group_result.get(timeout=self.time_limit)
                new_autocorr_mins[:], new_sim_times[:] = zip(*new_results)
            except Exception as e:
                if not isinstance(e, (TimeoutError, TimeLimitExceeded)):
                    raise

                # get() call timed out or task received hard timeout
                results: Iterable[AsyncResult] = group_result.results
                for i, result in enumerate(results):
                    if result.ready():
                        _autocorr, _sim_time = result.get()
                        if _autocorr <= 0:
                            new_autocorr_mins[i] = _autocorr
                            new_sim_times[i] = _sim_time
                        else:
                            # Simulation returned NaNs (maxiter_per_timestep exceeded)
                            new_autocorr_mins[i] = np.nan
                            new_sim_times[i] = np.nan
                    else:
                        new_autocorr_mins[i] = np.nan
                        new_sim_times[i] = np.nan

                # Cancel any running tasks - SIGINT signal is equivalent to
                # KeyboardInterrupt, a softer signal than the default (SIGTERM)
                revoked = 0
                for result in results:
                    if not result.ready():
                        revoked += 1
                        result.revoke(terminate=True, signal="SIGINT")
                if revoked:
                    self.logger.info(f"Time limit exceeded, canceled {revoked} tasks.")

            # Add simulation results
            autocorr_mins[which_to_simulate] = new_autocorr_mins

            # Store simulation results in transposition table
            if not state_in_ttable:
                # ignored if the key already exists
                database.sadd("transposition_table_keys", state_key)

            result_mapping = {}
            for seed, acf_min, sim_time in zip(
                seeds[which_to_simulate], new_autocorr_mins, new_sim_times
            ):
                if database.hexists(state_key, str(seed)):
                    continue
                if np.isnan(acf_min) or acf_min > 0.0:
                    acf_min = 1.0
                    sim_time = -1.0

                    # dispatch to cleanup queue - runs with no time limit
                    run_ssa_nolimit.delay(state, int(seed))
                result_mapping[str(seed)] = json.dumps(
                    [float(acf_min), float(sim_time)]
                )
            if result_mapping:
                self.logger.info(f"Storing {len(result_mapping)} results in database.")
                database.hset(state_key, mapping=result_mapping)

        # Increment visit counter
        self.visit_counter[state] += self.batch_size

        # Handle any NaN rewards
        nan_rewards = np.isnan(autocorr_mins)
        if nan_rewards.all():
            self.logger.warning(f"All rewards in batch are NaNs. Skipping this batch.")
            reward = self.get_reward(state)
        elif nan_rewards.any():
            self.logger.warning(f"Found NaN rewards in batch.")
            reward = np.mean(-autocorr_mins[~nan_rewards] > self.autocorr_threshold)
        else:
            reward = np.mean(-autocorr_mins > self.autocorr_threshold)

        self.iteration_counter += 1
        if self.iteration_counter % self.callback_every == 0:
            self._done_callback(self, state, seeds.tolist(), autocorr_mins.tolist())
        return reward

    @staticmethod
    def _done_callback(self_obj, state, visits, rewards):
        pass

    def add_done_callback(self, callback, *args, **kwargs):
        self._done_callback = partial(callback, *args, **kwargs)
