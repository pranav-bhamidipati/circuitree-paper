from gevent import monkey

monkey.patch_all()

from time import perf_counter
from gevent import getcurrent, sleep
from gevent.event import Event

# from gevent.lock import RLock
from gevent.queue import Queue
from itertools import count
from celery.result import AsyncResult
import numpy as np
from pathlib import Path
import redis
import sys
from typing import Hashable, Optional
import datetime
from uuid import uuid4

from circuitree.parallel import ParallelNetworkTree
from redis_backup import main as backup_database

from oscillation_app import database, app, task_logger, run_ssa_no_time_limit


### Define some helpful primitives for multithreading ###


class ExhaustionError(RuntimeError):
    """Raised when the entire search tree is exhaustively sampled."""


class BackupContextManager:
    def __init__(self, backup_not_in_progress: Event):
        self._backup_not_in_progress = backup_not_in_progress

    def __enter__(self):
        self._backup_not_in_progress.clear()

    def __exit__(self, *args):
        self._backup_not_in_progress.set()


class ManagedEvent(Event):
    """Event object that also provides a context manager."""

    def backup_context(self):
        """During a backup, the context manager will clear the event, blocking all
        threads until the backup is complete. Then the event is set again and
        threads are released."""
        return BackupContextManager(self)


class AtomicCounter:
    """Increments a counter atomically. Uses itertools.count(), which
    is implemented in C as an atomic operation.

    **REQUIRES CYPTHON**

    From StackOverflow user `PhilMarsh`:
        https://stackoverflow.com/a/71565358

    """

    def __init__(self):
        self._incs = count()
        self._accesses = count()

    def increment(self):
        next(self._incs)

    def value(self):
        return next(self._incs) - next(self._accesses)


### Define the multithreaded MCTS search tree and its batched counterpart ###


class MultithreadedOscillationTree(ParallelNetworkTree):
    """Run CircuiTree with a lock-free parallel implementation of the MCTS algorithm."""

    def __init__(
        self,
        *,
        save_dir: str | Path,
        success_threshold: float = 0.01,
        autocorr_threshold: float = 0.4,
        dt: float = 20.0,
        nt: int = 2000,
        nchunks: int = 5,
        database: redis.Redis = database,
        queue_size: Optional[int] = None,
        logger=None,
        tz_offset: int = -7,  # Pacific time
        **kwargs,
        # max_iter_per_timestep: int = 100_000_000,
    ):
        super().__init__(**kwargs)

        self.tree_id = str(uuid4())

        self.autocorr_threshold = autocorr_threshold
        self.dt = dt
        self.nt = nt
        self.nchunks = nchunks
        self.success_threshold = success_threshold

        self.backup_not_in_progress = ManagedEvent()
        self.backup_not_in_progress.set()
        self.last_backed_up_iteration: int = 0
        self.global_iteration = AtomicCounter()

        self.time_zone = datetime.timezone(datetime.timedelta(hours=tz_offset))
        self.next_backup_time = datetime.datetime.now(self.time_zone)

        self.queue_size = queue_size or 0
        self.result_history = Queue(maxsize=self.queue_size)

        if not save_dir:
            raise ValueError("Must specify a save directory.")

        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(exist_ok=True)
        self.save_dir = str(save_dir_path.resolve().absolute())

        self.database = database
        self.n_param_sets: int = self.database.hlen("parameter_table")

        self.logger = logger or task_logger

        # Specify attributes that should be skipped when dumping to file
        self._non_serializable_attrs.extend(
            [
                "save_dir",
                "logger",
                "database",
                "backup_not_in_progress",
                "last_backed_up_iteration",
                "global_iteration",
                "time_zone",
                "next_backup_time",
                "result_history",
            ]
        )

    @property
    def sim_kwargs(self):
        return dict(
            dt=self.dt,
            nt=self.nt,
            nchunks=self.nchunks,
            autocorr_threshold=self.autocorr_threshold,
            save_dir=self.save_dir,
        )

    def select_and_expand(
        self, rg: Optional[np.random.Generator] = None
    ) -> list[Hashable]:
        rg = self.rg if rg is None else rg

        # Start at root
        node = self.root
        selection_path = [node]
        actions = self.grammar.get_actions(node)

        # Select the child with the highest UCB score until you reach a terminal
        # state or an unexpanded edge
        while actions:
            max_ucb = -np.inf
            best_child = None
            rg.shuffle(actions)
            for action in actions:
                child = self._do_action(node, action)

                # Skip this child if it has been exhausted (i.e. all of its terminal
                # descendants have been expanded and simulated exhaustively)
                if self.graph.nodes[child].get("is_exhausted", False):
                    continue

                ucb = self.get_ucb_score(node, child)

                # An unexpanded edge has UCB score of infinity.
                # In this case, expand and select the child.
                if ucb == np.inf:
                    self.expand_edge(node, child)
                    selection_path.append(child)
                    return selection_path

                # Otherwise, track the child with the highest UCB score
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_child = child

            node = best_child
            selection_path.append(node)
            actions = self.grammar.get_actions(node)

        # If the loop breaks, we have reached a terminal state.

        # Check for exhaustion. A terminal state can become exhausted if it has been
        # simulated enough times to have sampled all parameter sets.
        if self.graph.nodes[node].get("visits", 0) >= self.n_param_sets - 1:
            self.mark_as_exhausted(node)

        return selection_path

    def mark_as_exhausted(self, node: str) -> None:
        """Mark a terminal node as exhausted. Its parent nodes will be modified to
        forget results from this node (i.e. the visits are decremented and the reward
        is subtracted from the total reward). If all nodes of the parent are exhausted,
        the parent is also marked as exhausted - this is done recursively up the tree.
        """
        self.logger.info(f"Marking node {node} as exhausted.")
        self.graph.nodes[node]["is_exhausted"] = True

        # If the whole tree is exhausted, we are done
        if node == self.root:
            self.logger.info(
                "Every node in the tree has been visited to exhaustion. Exiting."
            )
            raise ExhaustionError(
                "Every node in the tree has been visited to exhaustion."
            )

        # Forget results from this node
        for parent in self.graph.predecessors(node):
            edge_visits = self.graph.edges[parent, node]["visits"]
            edge_reward = self.graph.edges[parent, node]["reward"]
            self.graph.nodes[parent]["visits"] -= edge_visits
            self.graph.nodes[parent]["reward"] -= edge_reward

        # Recursively mark parent nodes as exhausted
        for parent in self.graph.predecessors(node):
            if all(
                self.graph.nodes[c].get("is_exhausted", False)
                for c in self.graph.successors(parent)
            ):
                self.mark_as_exhausted(parent)

    def get_param_set_index(self, state: str, visit: int) -> int:
        """Get the index of the parameter set to use for this state and visit number."""
        # Shuffle the param sets in a manner unique to each state
        param_set_indices = np.arange(self.n_param_sets)
        hash_val = hash(state) + sys.maxsize  # Make sure it's non-negative
        np.random.default_rng(hash_val).shuffle(param_set_indices)
        return param_set_indices[visit % self.n_param_sets]

    def get_reward(self, state: str, visit_number: int, **kwargs) -> float:
        """Get the simulation result for this state-seed pair"""
        param_index = self.get_param_set_index(state, visit_number)
        self.logger.info(f"Getting reward for {visit_number=}, {state=}")
        task: AsyncResult = run_ssa_no_time_limit.apply_async(
            args=(state, int(param_index)), kwargs=self.sim_kwargs, **kwargs
        )

        # Don't update iteration counter until a running backup is complete
        self.backup_not_in_progress.wait()
        self.global_iteration.increment()

        # Wait for the simulation to complete
        autocorr_min, (result_was_cached, simulation_time) = task.get()
        reward = float(-autocorr_min > self.autocorr_threshold)

        # Don't backpropagate until a running backup is complete. Most thread-time is
        # spent in the get() call above, so simulations will stlll be running in the
        # background during the backup.
        self.backup_not_in_progress.wait()
        self.logger.info(
            f"Autocorr. min. of {autocorr_min:.4f} for visit {visit_number} "
            f"to state {state}. Oscillating? {bool(reward)}."
        )
        self.result_history.put((state, reward))
        return reward


class BatchedOscillationTree(MultithreadedOscillationTree):
    """Run multiple simulations at each visit, backpropagating the incremental
    (average) reward."""

    def __init__(self, *, batch_size: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def traverse(self, thread_idx: int, **kwargs):
        """Select a node, simulate a trajectory, and backpropagate the result.
        Performs a single iteration of the MCTS algorithm, with multiple simulations
        in parallel per iteration."""
        # Select the next state to sample and the terminal state to be simulated.
        # Expands a child if possible.
        rg = self._random_generators[thread_idx]
        selection_path = self.select_and_expand(rg=rg)

        # Between backpropagation of visit and reward, we incur virtual loss
        self.backpropagate_visit(selection_path)

        # Simulate multiple trajectories from the selected state. The parameter
        # set for each trajectory is selected by shuffling the parameter set indices
        # in a manner unique to each state.
        selected_state = selection_path[-1]
        sim_nodes_and_param_indices = []
        for _ in range(self.batch_size):
            sim_node = self.get_random_terminal_descendant(selected_state, rg=rg)
            sample_number = self.sample_counter[sim_node]
            self.sample_counter[sim_node] += 1
            param_idx = self.get_param_set_index(sim_node, sample_number)
            sim_nodes_and_param_indices.append((sim_node, param_idx))

        # Submit all simulations to the Celery task queue
        self.logger.info(
            f"Getting reward for {self.batch_size} simulations of {selected_state=}"
        )
        tasks: list[AsyncResult] = []
        for sim_node, param_idx in sim_nodes_and_param_indices:
            task: AsyncResult = run_ssa_no_time_limit.apply_async(
                args=(sim_node, int(param_idx)), kwargs=self.sim_kwargs, **kwargs
            )
            tasks.append((sim_node, task))

        # As tasks complete, backpropagate the updated reward
        rewards = []
        while tasks:
            sleep(0.1)
            completed_tasks = [(s, t) for s, t in tasks if t.ready()]
            for simulated_state, task in completed_tasks:
                # Get task result
                autocorr_min, (result_was_cached, simulation_time) = task.get()

                # # If the result was cached, sleep for the duration of simulation time.
                # # This is to ensure that non-cached and cached results take the same amount
                # # of time to complete. Otherwise, MCTS will preferentially select cached
                # # results, which will bias the search.
                # if result_was_cached:
                #     self.logger.info(
                #         f"Result was cached. Sleeping for {simulation_time:.2f}s."
                #     )
                #     sleep(simulation_time)

                reward = float(-autocorr_min > self.autocorr_threshold)
                rewards.append(reward)

                # Don't backpropagate until a running backup has completed
                self.backup_not_in_progress.wait()
                self.logger.info(
                    f"Autocorr. min. of {autocorr_min:.4f} for {simulated_state=} "
                    f"Oscillating? {bool(reward)}."
                )

                # Backpropagate a fraction of the reward
                self.result_history.put((selected_state, reward))
                self.backpropagate_reward(selection_path, reward / self.batch_size)

                # Remove the task from the to-do list
                tasks.remove((simulated_state, task))

        # Keep track of the total number of iterations performed
        self.backup_not_in_progress.wait()
        self.global_iteration.increment()

        return selection_path, reward, sim_node


def progress_callback_in_thread(
    mtree: MultithreadedOscillationTree, iteration: int, *args, **kwargs
):
    """Callback function to report progress of MCTS search."""
    thread_id = getattr(getcurrent(), "minimal_ident", "__main__")
    _progress_msg = f"{iteration} iterations complete for thread {thread_id}"
    mtree.logger.info(_progress_msg)


def progress_and_backup_in_thread(
    mtree: MultithreadedOscillationTree,
    iteration: int,
    *args,
    progress_every: int = 10,
    backup_dir: str | Path,
    backup_every: int = 3600,
    n_tree_backups: int = 10,
    db_keyset: str = "transposition_table_keys",
    db_backup_prefix: str = "mcts_5tf_",
    backup_results: bool = True,
    force_backup: bool = False,
    dry_run: bool = False,
    # skip_sync: bool = False,
    **db_kwargs,
):
    """Callback function called at the end of every `callback_every` steps *per-thread*.
    When called:
        1) Reports progress of MCTS search
        2) If enough time has elapsed, backs up the tree to disk
        3) [Commented out] Blocks until all threads complete their callback
    """
    thread_id = getattr(getcurrent(), "minimal_ident", "__main__")
    if iteration > 0 and iteration % progress_every == 0:
        mtree.logger.info(
            f"Iterations complete: {mtree.global_iteration.value()} "
            f"({iteration} in thread {thread_id}))"
        )

    # Backup the tree if enough time has elapsed
    now = datetime.datetime.now(mtree.time_zone)
    do_backup = mtree.next_backup_time < now
    if do_backup or force_backup:
        if not mtree.backup_not_in_progress.is_set():
            # During a backup, non-backup threads block until the backup is complete.
            if iteration == 0:
                # No search iterations are performed before the initial backup.
                # After this, threads are blocked during the get_reward() call.
                mtree.backup_not_in_progress.wait()
            return

        # First thread to reach this point will perform the backup
        with mtree.backup_not_in_progress.backup_context():
            global_iter = mtree.global_iteration.value()
            mtree.logger.info(
                f"Backup triggered in thread {thread_id} at global iteration {global_iter}."
            )
            save_metadata = iteration == 0
            backup_results = backup_results and iteration != 0
            _backup_in_thread(
                mtree=mtree,
                global_iter=global_iter,
                backup_dir=backup_dir,
                backup_every=backup_every,
                n_tree_backups=n_tree_backups,
                db_keyset=db_keyset,
                db_backup_prefix=db_backup_prefix,
                db_kwargs=db_kwargs,
                backup_results=backup_results,
                save_metadata=save_metadata,
                dry_run=dry_run,
                **db_kwargs,
            )

        # if mtree.backup_not_in_progress.is_set():
        #     # Clear the event and perform the backup. Calls to backup_not_in_progress.wait() will
        #     # block until the backup is complete.
        #     with mtree.backup_not_in_progress.backup_context():
        #         global_iter = mtree.global_iteration.value()
        #         mtree.logger.info(
        #             f"Backup triggered in thread {thread_id} at global iteration {global_iter}."
        #         )
        #         save_metadata = iteration == 0
        #         backup_results = backup_results and iteration != 0
        #         _backup_in_thread(
        #             mtree=mtree,
        #             global_iter=global_iter,
        #             backup_dir=backup_dir,
        #             backup_every=backup_every,
        #             n_tree_backups=n_tree_backups,
        #             db_keyset=db_keyset,
        #             db_backup_prefix=db_backup_prefix,
        #             db_kwargs=db_kwargs,
        #             backup_results=backup_results,
        #             save_metadata=save_metadata,
        #             dry_run=dry_run,
        #             **db_kwargs,
        #         )

    # if skip_sync:
    #     return

    # # Regardless of backup, wait for all threads to reach this point before proceeding
    # with mtree.thread_lock:
    #     mtree.n_synced_threads += 1
    #     if mtree.n_synced_threads == 1:
    #         mtree.logger.info(
    #             f"Syncing threads at iteration {iteration} in thread {thread_id}"
    #         )
    #         mtree.threads_are_synced.clear()
    #     if mtree.n_synced_threads == mtree.threads:
    #         mtree.logger.info(
    #             f"All threads synced at iteration {iteration} in thread {thread_id}"
    #         )
    #         mtree.n_synced_threads = 0
    #         mtree.threads_are_synced.set()

    # mtree.threads_are_synced.wait()


def _backup_in_thread(
    mtree: MultithreadedOscillationTree,
    global_iter: int,
    backup_dir: str | Path,
    backup_every: int,
    n_tree_backups: int,
    db_keyset: str,
    db_backup_prefix: str,
    backup_results: bool,
    save_metadata: bool,
    dry_run: bool = False,
    **db_kwargs,
):
    date_time_fmt = "%Y-%m-%d_%H-%M-%S"
    now = datetime.datetime.now(mtree.time_zone)
    now_fmt = now.strftime(date_time_fmt)

    gml_file = Path(backup_dir) / f"tree-{mtree.tree_id}_{now_fmt}.gml"
    existing_gmls = sorted(gml_file.parent.glob(f"*{mtree.tree_id}*.gml*"))
    # Delete old backups
    for f in existing_gmls[:-n_tree_backups]:
        mtree.logger.info(f"Deleting old backup file: {f}")
        f.unlink()

    # Tree attributes (metadata) are only saved once
    if save_metadata:
        json_file = Path(backup_dir) / f"tree-{mtree.tree_id}_{now_fmt}.json"
        mtree.logger.info(f"Backing up tree metadata to file: {json_file}")
    else:
        json_file = None

    gml_compressed = gml_file.with_suffix(".gml.gz")
    mtree.logger.info(f"Backing up tree graph to file: {gml_compressed}")

    tree_start = perf_counter()
    mtree.to_file(gml_file, json_file, compress=True)
    tree_end = perf_counter()

    mtree.logger.info(f"Graph backup completed in {tree_end-tree_start:.4f} seconds.")

    mtree.logger.info(f"Backing up transposition table database...")
    database_info = mtree.database.connection_pool.connection_kwargs
    db_start = perf_counter()
    backup_database(
        keyset=db_keyset,
        host=database_info["host"],
        port=database_info["port"],
        db=database_info["db"],
        save_dir=backup_dir,
        prefix=db_backup_prefix,
        tz=mtree.time_zone,
        progress_bar=False,
        print_progress=True,
        logger=mtree.logger,
        dry_run=dry_run,
        **db_kwargs,
    )
    db_end = perf_counter()
    mtree.logger.info(f"Database backup completed in {db_end-db_start:.4f} seconds.")

    if backup_results:
        last_backed_up = mtree.last_backed_up_iteration
        results_file = Path(mtree.save_dir).joinpath(
            f"results_steps{last_backed_up+1}-{global_iter}"
            f"_{mtree.tree_id}_{now_fmt}.txt"
        )
        mtree.logger.info(
            f"Backing up states visited at steps "
            f"{last_backed_up+1}-{global_iter} to file: {results_file}"
        )

        # gevent Queue object can be called as an iterator, calling get() repeatedly.
        # This clears the queue. Note that StopIteration is required, otherwise the
        # iterator won't terminate and get() will block indefinitely.
        result_start = perf_counter()
        mtree.result_history.put(StopIteration)
        visit_results = [
            ",".join([state, str(reward)]) for state, reward in mtree.result_history
        ]
        results_file.write_text("\n".join(visit_results))
        result_end = perf_counter()
        mtree.logger.info(
            f"Results backup completed in {result_end-result_start:.4f} seconds."
        )

    # Release all threads
    mtree.logger.info("Done. Releasing all threads.")
    mtree.last_backed_up_iteration = global_iter
    mtree.next_backup_time = now + datetime.timedelta(seconds=backup_every)
