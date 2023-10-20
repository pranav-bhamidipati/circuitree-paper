from gevent import monkey

monkey.patch_all()

from time import perf_counter
from gevent import getcurrent
from gevent.event import Event
from gevent.queue import Queue
from itertools import count
from celery.result import AsyncResult
import numpy as np
from pathlib import Path
import redis
from typing import Optional
import datetime
from uuid import uuid4

from circuitree.parallel import ParallelNetworkTree
from redis_backup import main as backup_database

from oscillation_app import database, app, task_logger, run_ssa_no_time_limit


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


class MultithreadedOscillationTree(ParallelNetworkTree):
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
        **kwargs,
        # max_iter_per_timestep: int = 100_000_000,
    ):
        super().__init__(**kwargs)

        self.autocorr_threshold = autocorr_threshold
        self.dt = dt
        self.nt = nt
        self.nchunks = nchunks
        self.success_threshold = success_threshold

        self.tree_id = str(uuid4())
        self.backup_not_in_progress = ManagedEvent()
        self.backup_not_in_progress.set()
        self.last_backed_up_iteration: int = 0
        self.next_backup_time: datetime.datetime = datetime.datetime.now()
        self.global_iteration = AtomicCounter()

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
                "next_backup_time",
                "global_iteration",
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

    def get_param_set_index(self, state: str, visit: int) -> int:
        """Get the index of the parameter set to use for this state and visit number."""
        # Shuffle the param sets in a manner unique to each state
        param_set_indices = np.arange(self.n_param_sets)
        np.random.default_rng(hash(state)).shuffle(param_set_indices)
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
        autocorr_min = task.get()
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
    backup_dir: str | Path,
    backup_every: int = 3600,
    tz_offset: int = -7,
    n_tree_backups: int = 10,
    db_keyset: str = "transposition_table_keys",
    db_backup_prefix: str = "mcts_5tf_",
    db_kwargs: dict = None,
    backup_results: bool = True,
    force_backup: bool = False,
    **kwargs,
):
    """Callback function to report progress of MCTS search and save the tree to disk."""
    thread_id = getattr(getcurrent(), "minimal_ident", "__main__")
    if iteration > 0:
        mtree.logger.info(
            f"Iterations complete: {mtree.global_iteration.value()} "
            f"({iteration} in thread {thread_id}))"
        )

    # Backup the tree if enough time has elapsed
    now = datetime.datetime.now()
    if mtree.next_backup_time < now or force_backup:
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
                tz_offset=tz_offset,
                n_tree_backups=n_tree_backups,
                db_keyset=db_keyset,
                db_backup_prefix=db_backup_prefix,
                db_kwargs=db_kwargs,
                backup_results=backup_results,
                save_metadata=save_metadata,
            )


def _backup_in_thread(
    mtree: MultithreadedOscillationTree,
    global_iter: int,
    backup_dir: str | Path,
    backup_every: int,
    tz_offset: int,
    n_tree_backups: int,
    db_keyset: str,
    db_backup_prefix: str,
    db_kwargs: dict,
    backup_results: bool,
    save_metadata: bool,
):
    date_time_fmt = "%Y-%m-%d_%H-%M-%S"
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=tz_offset)))
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
        tz_offset=tz_offset,
        progress_bar=False,
        print_progress=True,
        logger=mtree.logger,
        **(db_kwargs or {}),
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
