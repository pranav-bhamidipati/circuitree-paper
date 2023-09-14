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


@app.task(soft_time_limit=120, time_limit=300, queue="simulations")
def run_ssa(
    state: str,
    visit_num: int,
    # prots0: list[int],
    # params: list[float],
):
    # Use the parameter set corresponding to the visit number
    seed = visit_num
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
        task_logger.info(f"Running SSA with {seed=}, {state=}")
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.time_limit = time_limit
        self.visit_counter = Counter(counter)
        self.logger = task_logger

        self.iteration_counter = 0
        self.callback_every = callback_every

        # Specify attributes that should not/cannot be serialized when dumping to file
        self._non_serializable_attrs.extend(
            ["visit_counter", "logger", "iteration_counter"]
        )

    def get_reward(self, state: str):
        visit0 = self.visit_counter[state]
        visits = np.arange(visit0, visit0 + self.batch_size)

        # Check if the state is in the transposition table and pull any rewards
        state_key = "state_" + state.strip("*")  # asterisk is interpreted as wildcard
        state_in_ttable = bool(
            database.sismember("transposition_table_keys", state_key)
        )
        autocorr_mins = -np.ones(self.batch_size, dtype=np.float64)
        which_to_simulate = np.ones(self.batch_size, dtype=bool)
        if state_in_ttable:
            past_results = database.hmget(state_key, visits.tolist())
            for i, res in enumerate(past_results):
                in_table = res is not None
                which_to_simulate[i] = not in_table
                if in_table:
                    # autocorr_min should be nonpositive. A positive value means the
                    # simulation returned NaN (maxiter exceeded or timeout)
                    autocorr_min, sim_time = json.loads(res)
                    if autocorr_min > 0.0:
                        ...
                        autocorr_mins[i] = np.nan

        # Dispatch simulation tasks
        n_to_simulate = which_to_simulate.sum()
        new_autocorr_mins = np.ones(n_to_simulate, dtype=np.float64)
        new_sim_times = -np.ones(n_to_simulate, dtype=np.float64)
        group_result: AsyncResult | GroupResult = group(
            run_ssa.s(state, int(visit)) for visit in visits[which_to_simulate]
        ).apply_async()
        if n_to_simulate:
            try:
                new_results = group_result.get(timeout=self.time_limit)
                new_autocorr_mins[:], new_sim_times[:] = zip(*new_results)
            except Exception as e:
                # get() timed out or task received hard timeout
                if isinstance(e, (TimeoutError, TimeLimitExceeded)):
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
                        self.logger.info(
                            f"Time limit exceeded, canceled {revoked} tasks."
                        )
                else:
                    raise

            if np.isnan(new_autocorr_mins).any():
                ...

            # Add simulation results
            autocorr_mins[which_to_simulate] = new_autocorr_mins

            # Store simulation results in transposition table
            if not state_in_ttable:
                # ignored if the key already exists
                database.sadd("transposition_table_keys", state_key)

            result_mapping = {}
            for v, a, t in zip(
                visits[which_to_simulate], new_autocorr_mins, new_sim_times
            ):
                if database.hexists(state_key, str(v)):
                    continue
                if np.isnan(a) or a > 0.0:
                    a = 1.0
                    t = -1.0
                result_mapping[str(v)] = json.dumps([float(a), float(t)])
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
            self._done_callback(self, state, visits.tolist(), autocorr_mins.tolist())
        return reward

    @staticmethod
    def _done_callback(self_obj, state, visits, rewards):
        pass

    def add_done_callback(self, callback, *args, **kwargs):
        self._done_callback = partial(callback, *args, **kwargs)


# def save_ttable_every_n_iters(tree: OscillationTreeParallel, *args, **kwargs):
#     """Callback function to save the transposition table every n iterations."""
#     tree.iteration_counter += 1
#     tree.logger.info(f"Iteration {tree.iteration_counter} complete.")
#     i = tree.iteration_counter
#     if i % tree.save_ttable_every == 0:
#         ttable_fpath = Path(tree.save_dir) / f"iter{i}_trans_table.csv"
#         sim_table_fpath = Path(tree.save_dir) / f"iter{i}_simtime_table.csv"
#         tree.logger.info(
#             f"Saving transposition table and simulation times to {tree.save_dir}"
#         )
#         tree.ttable.to_csv(ttable_fpath, **kwargs)
#         tree.simtime_table.to_csv(sim_table_fpath, **kwargs)


# class OscillationTreeCeleryLocal(OscillationTreeParallel):
#     def __init__(
#         self,
#         save_ttable_every: int = 10,
#         sim_time_table: Optional[TranspositionTable] = None,
#         logger: Any = None,
#         time_limit: int = 120,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)

#         self._simulation_time_table = TranspositionTable(sim_time_table)
#         self.iteration_counter = 0
#         self.save_ttable_every = save_ttable_every
#         self.time_limit = time_limit

#         if logger is None:
#             self.logger = task_logger
#         elif isinstance(logger, str) or isinstance(logger, Path):
#             import logging

#             self.logger = logging.getLogger(__name__)
#             self.logger.handlers.clear()
#             self.logger.setLevel(logging.INFO)
#             log_fmt = logging.Formatter(
#                 "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#             )
#             logfile = Path(logger)
#             fh = logging.FileHandler(logfile, mode="w")
#             fh.setFormatter(log_fmt)
#             self.logger.addHandler(fh)
#         else:
#             self.logger = logger

#         self.add_done_callback(save_ttable_every_n_iters)

#         # Specify any attributes that should not be serialized when dumping to file
#         self._non_serializable_attrs.extend(["_simulation_time_table", "logger"])

#     @property
#     def run_task(self) -> Task:
#         return run_ssa

#     @property
#     def simtime_table(self):
#         return self._simulation_time_table

#     def save_results(self, data: dict[str, Any]) -> None:
#         """Results are saved in the worker processes, so this method is a no-op."""
#         return

#     def compute_tasks_and_handle_errors(
#         self, iter_args, n_retries: int = 5, **kwargs
#     ) -> tuple[list[float], list[float]]:
#         task_group = group(self.run_task.s(*args, **kwargs) for args in iter_args)
#         group_result: GroupResult | AsyncResult = task_group.apply_async(
#             retry=True, retry_policy={"max_retries": n_retries}
#         )
#         try:
#             results = group_result.get(timeout=self.time_limit)
#             rewards, sim_times = zip(*results)
#         except Exception as e:
#             # get() timed out or task received hard timeout
#             if isinstance(e, (TimeoutError, TimeLimitExceeded)):
#                 rewards = []
#                 sim_times = []
#                 revoked = 0
#                 results: Iterable[AsyncResult] = group_result.results
#                 for result in results:
#                     if result.ready():
#                         reward, sim_time = result.get()
#                         r = reward if reward >= 0 else np.nan
#                         s = sim_time if reward >= 0 else np.nan
#                     else:
#                         # Cancel any running tasks - SIGINT signal is equivalent to
#                         # KeyboardInterrupt and is a softer terminate signal than the
#                         # default (SIGTERM)
#                         revoked += 1
#                         result.revoke(terminate=True, signal="SIGINT")
#                         r = np.nan
#                         s = np.nan
#                     rewards.append(r)
#                     sim_times.append(s)
#                 self.logger.info(f"Time limit exceeded, canceled {revoked} tasks.")

#         return rewards, sim_times

#     @cached_property
#     def task_kwargs(self) -> dict[str, Any]:
#         return dict(
#             dt=float(self.dt),
#             nt=int(self.nt),
#             autocorr_threshold=float(self.autocorr_threshold),
#             max_iter_per_timestep=int(self.max_iter_per_timestep),
#             save_dir=str(self.save_dir),
#         )

#     def simulate_visits(self, state, visits) -> tuple[list[float], dict[str, Any]]:
#         # Make the input args JSON serializable
#         input_args = []
#         for v in visits:
#             seed, inits, params = self.param_table[v]
#             input_args.append(
#                 (int(seed), list(map(int, inits)), list(map(float, params)))
#             )

#         # Submit the tasks and wait for them to finish, handling timeouts and retries
#         rewards, sim_times = self.compute_tasks_and_handle_errors(
#             input_args, state=str(state), **self.task_kwargs
#         )
#         self.simtime_table[state].extend(list(sim_times))

#         return rewards, {}
