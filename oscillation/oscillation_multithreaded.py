from gevent import monkey

monkey.patch_all()

from gevent import getcurrent
from gevent.queue import Queue

from celery.result import AsyncResult
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from time import perf_counter
from typing import Optional
from uuid import uuid4

from circuitree.utils import (
    DatabaseBackupManager,
    ThreadsafeCounter,
    ThreadsafeCountTable,
)

from redis_backup import main as backup_database
from oscillation import OscillationTree
from oscillation_app import database, app, task_logger, run_ssa_no_time_limit


class TimedBackupManager(DatabaseBackupManager):

    def backup_to_file(
        self,
        save_dir: str | Path,
        prefix: str,
        keyset: str = "transposition_table_keys",
        logger=None,
        dry_run: bool = False,
        timed: bool = True,
        **kwargs,
    ):
        """Backup the database to a file."""
        if timed:
            start = perf_counter()
        backup_database(
            keyset=keyset,
            host=self.database_info["host"],
            port=self.database_info["port"],
            db=self.database_info["db"],
            save_dir=save_dir,
            prefix=prefix,
            tz=self.time_zone,
            progress_bar=False,
            print_progress=True,
            logger=logger,
            dry_run=dry_run,
            **kwargs,
        )
        if timed:
            end = perf_counter()
            return end - start
        else:
            return


class MultithreadedOscillationTree(OscillationTree):
    """Run CircuiTree with a lock-free parallel implementation of the MCTS algorithm."""

    def __init__(
        self,
        *,
        save_dir: str | Path,
        param_sets_csv: str | Path,
        n_exhausted: Optional[int] = None,
        success_threshold: float = 0.01,
        autocorr_threshold: float = -0.4,
        dt: float = 20.0,
        nt: int = 2000,
        nchunks: int = 5,
        queue_size: Optional[int] = None,
        logger=None,
        tz_offset: int = -7,  # Pacific time
        next_backup_in_seconds: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tree_id = str(uuid4())
        self.logger = logger or task_logger

        self.backup = TimedBackupManager(
            database_info=database.connection_pool.connection_kwargs,
            tz_offset=tz_offset,
            next_backup_in_seconds=next_backup_in_seconds,
        )
        # self.global_iteration = AtomicCounter()
        self.global_iteration = ThreadsafeCounter()
        self.visit_counter = ThreadsafeCountTable()
        self.last_results_dump_iter = 0

        self.queue_size = queue_size or 0
        self.result_history = Queue(maxsize=self.queue_size)

        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(exist_ok=True)
        save_dir_path.joinpath("backups").mkdir(exist_ok=True)
        save_dir_path.joinpath("results").mkdir(exist_ok=True)
        save_dir_path.joinpath("data").mkdir(exist_ok=True)
        self.save_dir = str(save_dir_path.resolve().absolute())

        # The parameter sets to sample are stored in a CSV file
        self.param_sets_csv = str(param_sets_csv.resolve().absolute())
        self.n_param_sets = pd.read_csv(self.param_sets_csv).shape[0]

        self.autocorr_threshold = autocorr_threshold
        self.success_threshold = success_threshold
        self.sim_kwargs = dict(
            dt=dt,
            nt=nt,
            nchunks=nchunks,
            autocorr_threshold=self.autocorr_threshold,
            param_sets_csv=self.param_sets_csv,
            save_dir=self.save_dir,
            root_seed=self.seed,
        )

        # The number of times a terminal node is visited before it is exhausted
        # (i.e. all parameter sets have been sampled)
        self.n_exhausted = n_exhausted

        # Specify attributes that should be skipped when dumping to file
        self._non_serializable_attrs.extend(
            [
                "save_dir",
                "logger",
                "database",
                "backup",
                "global_iteration",
                "visit_counter",
                "result_history",
            ]
        )

    def get_param_set_index(self, state: str, visit: int) -> int:
        """Get the index of the parameter set to use for this state and visit number.
        Shuffles the order of the parameter sets in a manner unique to each state,
        so that different threads will sample parameter sets in a different order. This
        is because most nodes will only be sampled once, so the order of sampling is
        important for the diversity of the search.

        Under the hood, this function uses a random number generator seeded with the
        hash of the state string. This ensures that the order of parameter sets is
        deterministic for each state, but uncorrelated between states.
        """
        param_set_indices = np.arange(self.n_param_sets)
        hash_val = hash(state) + sys.maxsize  # Ensure positive hash value
        np.random.default_rng(hash_val).shuffle(param_set_indices)
        return param_set_indices[visit % self.n_param_sets]

    def get_reward(self, state: str, **kwargs) -> float:
        """Get the simulation result for this state-seed pair"""
        visit_number = self.visit_counter.get_val_and_increment(state)
        param_index = self.get_param_set_index(state, visit_number)
        self.logger.info(f"Getting reward for {visit_number=}, {state=}")
        task: AsyncResult = run_ssa_no_time_limit.apply_async(
            args=(state, int(param_index)), kwargs=self.sim_kwargs, **kwargs
        )

        # Don't update iteration counter until a running backup is complete
        self.backup.wait_until_finished()
        self.global_iteration.increment()

        # Wait for the simulation to complete
        autocorr_min, (result_was_cached, simulation_time) = task.get()
        reward = float(autocorr_min < self.autocorr_threshold)

        # Wait for any backups to complete before the next step (backpropagation)
        self.backup.wait_until_finished()
        self.logger.info(
            f"Autocorr. min. of {autocorr_min:.4f} for visit {visit_number} "
            f"to state {state}. Oscillating? {bool(reward)}."
        )
        self.result_history.put((state, reward))
        return reward

    def dump_results(self, timestamp: str = None, *args, **kwargs):
        """Dump the results of the MCTS search to a file."""

        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f"_{timestamp}.txt"

        last_dumped = self.last_results_dump_iter
        global_iter = self.global_iteration.value()
        results_file = Path(self.save_dir).joinpath(
            f"results_steps{last_dumped+1}-{global_iter}" f"_{self.tree_id}{suffix}.txt"
        )
        self.logger.info(
            f"Dumping results of iters {last_dumped+1}-{global_iter} to: {results_file}"
        )

        # gevent Queue object can be called as an iterator, calling get() repeatedly.
        # This clears the queue. Note that StopIteration is required, otherwise the
        # iterator won't terminate and get() will block indefinitely.
        self.result_history.put(StopIteration)
        visit_results = [
            ",".join([state, str(reward)]) for state, reward in self.result_history
        ]
        results_file.write_text("\n".join(visit_results))


def progress_and_backup_in_thread(
    mtree: MultithreadedOscillationTree,
    iteration: int,
    *args,
    progress_every: int = 10,
    backup_every: int = 10_800,
    n_tree_backups: int = 200,
    db_keyset: str = "transposition_table_keys",
    db_backup_prefix: str = "mcts_5tf_",
    dump_results: bool = True,
    force_backup: bool = False,
    dry_run: bool = False,
    # skip_sync: bool = False,
    **db_kwargs,
):
    """Callback function called at the end of every `callback_every` steps *per-thread*.
    When called:
        1) Reports progress of MCTS search
        2) If enough time has elapsed, backs up the search tree and database to disk
        3) If `dump_results` is True, dumps to disk the results of iterations completed
            since the last dump.
    """
    thread_id = getattr(getcurrent(), "minimal_ident", "__main__")
    if iteration > 0 and iteration % progress_every == 0:
        mtree.logger.info(
            f"Iterations complete: {mtree.global_iteration.value()} "
            f"({iteration} in thread {thread_id}))"
        )

    # Backup the tree if enough time has elapsed
    if force_backup or mtree.backup.is_due():
        if mtree.backup.is_running():
            # During a backup, non-backup threads block until the backup is complete.
            if iteration == 0:
                # No search iterations are performed before the initial backup.
                # After this, threads are blocked during the get_reward() call.
                mtree.backup.wait()
            return

        # First thread to reach this point will perform the backup
        with mtree.backup():
            global_iter = mtree.global_iteration.value()
            mtree.logger.info(
                f"Backup triggered in thread {thread_id} at global iteration {global_iter}."
            )
            save_metadata = iteration == 0
            dump_results = dump_results and iteration != 0

            backup_dir = Path(mtree.save_dir).joinpath("backups")
            backup_dir.mkdir(exist_ok=True)
            backup_in_thread(
                mtree=mtree,
                global_iter=global_iter,
                backup_dir=backup_dir,
                backup_every=backup_every,
                n_tree_backups=n_tree_backups,
                db_keyset=db_keyset,
                db_backup_prefix=db_backup_prefix,
                db_kwargs=db_kwargs,
                dump_results=dump_results,
                save_metadata=save_metadata,
                dry_run=dry_run,
                **db_kwargs,
            )


def backup_in_thread(
    mtree: MultithreadedOscillationTree,
    global_iter: int,
    backup_dir: str | Path,
    backup_every: int,
    n_tree_backups: int,
    db_keyset: str,
    db_backup_prefix: str,
    dump_results: bool,
    save_metadata: bool,
    dry_run: bool = False,
    **db_kwargs,
):
    date_time_fmt = "%Y-%m-%d_%H-%M-%S"
    now = datetime.datetime.now(mtree.backup.time_zone)
    now_fmt = now.strftime(date_time_fmt)

    gml_file = Path(backup_dir) / f"tree-{mtree.tree_id}_{now_fmt}.gml"
    existing_gmls = sorted(gml_file.parent.glob(f"*{mtree.tree_id}*.gml*"))
    # Delete old backups
    for f in existing_gmls[:-n_tree_backups]:
        mtree.logger.info(f"Deleting old backup file: {f}")
        f.unlink()

    # Tree attributes (metadata) only need to be saved once, at the beginning
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
    database_info = database.connection_pool.connection_kwargs
    backup_elapsed = mtree.backup.backup_to_file(
        save_dir=backup_dir,
        prefix=db_backup_prefix,
        database_info=database_info,
        keyset=db_keyset,
        dry_run=dry_run,
        **db_kwargs,
    )
    mtree.logger.info(f"Database backup completed in {backup_elapsed:.4f} seconds.")
    mtree.backup.schedule_next(backup_every)

    if dump_results:
        last_dumped = mtree.last_results_dump_iter
        result_start = perf_counter()
        mtree.dump_results(now_fmt)
        result_end = perf_counter()
        mtree.logger.info(
            f"Dumped {global_iter - last_dumped} results in "
            f"{result_end-result_start:.4f} secs."
        )

    # Release all threads
    mtree.logger.info("Done. Releasing all threads.")
    mtree.last_results_dump_iter = global_iter


def progress_callback_in_thread(
    mtree: MultithreadedOscillationTree, iteration: int, *args, **kwargs
):
    """Callback function to report progress of MCTS search."""
    thread_id = getattr(getcurrent(), "minimal_ident", "__main__")
    _progress_msg = f"{iteration} iterations complete for thread {thread_id}"
    mtree.logger.info(_progress_msg)
