from gevent import monkey

monkey.patch_all()

from time import perf_counter
from collections import Counter
from gevent import Greenlet, getcurrent
from gevent.event import Event
from gevent.queue import Queue
from greenlet import greenlet
from threading import current_thread
from celery.result import AsyncResult
from pathlib import Path
import redis
from typing import Optional
import datetime
from uuid import uuid4

from circuitree.parallel import MultithreadedCircuiTree, search_mcts_in_thread
from oscillation import OscillationGrammar
from redis_backup import main as backup_database

from oscillation_app import database, app, task_logger, run_ssa_no_time_limit


class MultithreadedOscillationTree(OscillationGrammar, MultithreadedCircuiTree):
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
        min_ssa_seed: Optional[int] = None,
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
        self.backup_not_in_progress = Event()
        self.backup_not_in_progress.set()
        self.last_backed_up_iteration: int = 0
        self.next_backup_time: datetime.datetime = datetime.datetime.now()
        self.current_iteration = Counter()

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
        self.min_ssa_seed = min_ssa_seed or 0

        # Specify attributes that should be skipped when dumping to file
        self._non_serializable_attrs.extend(
            [
                "save_dir",
                "logger",
                "database",
                "backup_not_in_progress",
                "last_backed_up_iteration",
                "next_backup_time",
                "current_iteration",
                "result_history",
            ]
        )

        # self.max_iter_per_timestep = max_iter_per_timestep-

    @property
    def sim_kwargs(self):
        return dict(
            dt=self.dt,
            nt=self.nt,
            nchunks=self.nchunks,
            autocorr_threshold=self.autocorr_threshold,
            save_dir=self.save_dir,
        )

    def get_random_seed(self, visit: int) -> int:
        return self.min_ssa_seed + visit % self.n_param_sets

    def get_reward(self, state: str, visit_number: int, **kwargs) -> float:
        """Get the simulation result for this state-seed pair"""
        seed = self.get_random_seed(visit_number)
        self.logger.info(f"Getting reward for {seed=}, {state=}")
        task: AsyncResult = run_ssa_no_time_limit.apply_async(
            args=(state, int(seed)), kwargs=self.sim_kwargs, **kwargs
        )

        # Don't update iteration count until a running backup is complete
        self.backup_not_in_progress.wait()
        curr_thread = getcurrent()
        thread_id = getattr(curr_thread, "minimal_ident", None)
        if thread_id is None:
            self.current_iteration[0] += 1
        else:
            self.current_iteration[thread_id] += 1

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
    db_backup_dir: str | Path,
    backup_every: int = 3600,
    gml_file: Optional[str | Path] = None,
    json_file: Optional[str | Path] = None,
    tz_offset: int = -7,
    keep_single_gml_backup: bool = True,
    db_keyset: str = "transposition_table_keys",
    db_backup_prefix: str = "mcts_5tf_",
    db_kwargs: dict = None,
    backup_results: bool = True,
    **kwargs,
):
    """Callback function to report progress of MCTS search and save the tree to disk."""
    thread_id = getattr(getcurrent(), "minimal_ident", "__main__")
    overall_iteration = sum(mtree.current_iteration)
    if iteration > 0:
        mtree.logger.info(
            f"Iterations complete: {overall_iteration} ({iteration} "
            f"in thread {thread_id}))"
        )

    # Backup the tree if enough time has elapsed
    now = datetime.datetime.now()
    if mtree.next_backup_time < now:
        if mtree.backup_not_in_progress.is_set():
            # First thread to reach this point will perform the backup
            mtree.backup_not_in_progress.clear()
        else:
            # Other threads remain blocked until the backup is complete.
            if iteration == 0:
                # Ensure no search iterations are performed until the initial backup is complete.
                # After this, threads are blocked during the get_reward() call.
                mtree.backup_not_in_progress.wait()
            return

        date_time_fmt = "%Y-%m-%d_%H-%M-%S"
        now_fmt = datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=tz_offset))
        ).strftime(date_time_fmt)

        mtree.logger.info(
            f"Backup triggered in thread {thread_id} at overall iteration {overall_iteration}."
        )

        # Tree attributes (metadata) are only saved once
        if iteration == 0:
            json_file = (
                json_file
                or Path(mtree.save_dir) / f"tree_{mtree.tree_id}_{now_fmt}.json"
            )
        else:
            json_file = None

        if gml_file is not None:
            # Wildcard catches any compression file extensions (like .gml.gz)
            if any(gml_file.parent.glob(f"{gml_file.name}*")):
                mtree.logger.info(f"Backup file already exists. Overwriting...")
        else:
            gml_file = Path(mtree.save_dir) / f"tree-{mtree.tree_id}_{now_fmt}.gml"
            existing_gmls = list(gml_file.parent.glob(f"*{mtree.tree_id}*.gml*"))
            if existing_gmls and keep_single_gml_backup:
                mtree.logger.info(
                    f"Some backup file(s) already exist and will be deleted:",
                    *existing_gmls,
                    sep="\n\t",
                )
                for f in existing_gmls:
                    f.unlink()

        mtree.logger.info(f"Backing up tree graph to file: {gml_file}")

        if json_file is not None:
            mtree.logger.info(f"Backing up tree metadata to file: {json_file}")

        tree_start = perf_counter()
        mtree.to_file(gml_file, json_file, compress=True)
        tree_end = perf_counter()
        mtree.logger.info(
            f"Tree backup completed in {tree_end-tree_start:.4f} seconds."
        )
        mtree.logger.info(f"Backing up transposition table database...")
        database_info = mtree.database.connection_pool.connection_kwargs
        db_start = perf_counter()
        backup_database(
            keyset=db_keyset,
            host=database_info["host"],
            port=database_info["port"],
            db=database_info["db"],
            save_dir=db_backup_dir,
            prefix=db_backup_prefix,
            tz_offset=tz_offset,
            progress_bar=False,
            print_progress=True,
            **(db_kwargs or {}),
        )
        db_end = perf_counter()
        mtree.logger.info(
            f"Database backup completed in {db_end-db_start:.4f} seconds."
        )

        if backup_results and iteration > 0:
            last_backed_up = mtree.last_backed_up_iteration
            results_file = Path(mtree.save_dir).joinpath(
                f"results_steps{last_backed_up+1}-{overall_iteration}"
                f"_{mtree.tree_id}_{now_fmt}.txt"
            )
            mtree.logger.info(
                f"Backing up states visited at steps "
                f"{last_backed_up+1}-{overall_iteration} to file: {results_file}"
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
        mtree.last_backed_up_iteration = overall_iteration
        mtree.next_backup_time = now + datetime.timedelta(seconds=backup_every)
        mtree.backup_not_in_progress.set()
