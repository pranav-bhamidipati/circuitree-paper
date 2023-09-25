from time import perf_counter
from gevent import monkey

monkey.patch_all()

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

        self.tree_id = uuid4()
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
                "last_backup_iteration",
                "next_backup_time",
                "current_iteration",
                "visited_states",
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

        # If there's a backup in progress, block until it's complete. Simulations are
        # performed in the meantime!
        self.backup_not_in_progress.wait()
        self.current_iteration[getcurrent().minimal_ident] += 1

        autocorr_min = task.get()
        reward = float(-autocorr_min > self.autocorr_threshold)
        self.logger.info(
            f"Autocorr. min. of {autocorr_min:.4f} for visit {visit_number} "
            f"to state {state}. Oscillating? {bool(reward)}."
        )
        self.backup_not_in_progress.wait()
        self.result_history.put((state, reward))
        return reward


def progress_callback_in_main(
    mtree: MultithreadedOscillationTree, iteration: int, *args, **kwargs
):
    """Callback function to report progress of MCTS search."""
    _progress_msg = f"{iteration} iterations completed."
    mtree.logger.info(_progress_msg)


def progress_callback_in_thread(
    mtree: MultithreadedOscillationTree, iteration: int, *args, **kwargs
):
    """Callback function to report progress of MCTS search."""
    thread_id = getcurrent().minimal_ident
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
    backup_visits: bool = True,
    **kwargs,
):
    """Callback function to report progress of MCTS search and save the tree to disk."""
    thread_id = getcurrent().minimal_ident
    overall_iteration = sum(mtree.current_iteration)
    if iteration > 0:
        mtree.logger.info(
            f"Iterations complete: {overall_iteration} ({iteration} in thread {thread_id}))"
        )

    # Backup the tree if enough time has elapsed
    now = datetime.datetime.now()
    if mtree.next_backup_time < now:
        if mtree.backup_not_in_progress.is_set():
            # First thread to reach this point will perform the backup
            mtree.backup_not_in_progress.clear()
            mtree.next_backup_time = now + datetime.timedelta(seconds=backup_every)
        else:
            # Other threads remain blocked until the backup is complete.
            if iteration == 0:
                # Ensure no search iterations are performed until the initial backup is complete.
                # After this, threads are blocked during the get_reward() call.
                mtree.backup_not_in_progress.wait()
            return

        mtree.logger.info(
            f"Backup triggered in thread {thread_id} at overall iteration {overall_iteration}."
        )

        # Tree attributes (metadata) are only saved once
        if iteration == 0:
            json_file = (
                json_file or Path(mtree.save_dir) / f"tree_{mtree.tree_id}_{now}.json"
            )
        else:
            json_file = None

        date_time_fmt = "%Y-%m-%d_%H-%M-%S"
        now = datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=tz_offset))
        ).strftime(date_time_fmt)

        if gml_file is not None:
            if gml_file.exists():
                mtree.logger.info(f"Backup file already exists. Overwriting...")
        else:
            gml_file = Path(mtree.save_dir) / f"tree-{mtree.tree_id}_{now}.gml"
            existing_gmls = list(gml_file.parent.glob(f"*{mtree.tree_id}*.gml"))
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

        start = perf_counter()
        mtree.to_file(gml_file, json_file, compress=True)
        end = perf_counter()
        mtree.logger.info(f"Tree backup completed in {end-start:.4f} seconds.")
        mtree.logger.info(f"Backing up transposition table database...")
        database_info = mtree.database.connection_pool.connection_kwargs
        backup_database(
            keyset=db_keyset,
            host=database_info["host"],
            port=database_info["port"],
            db=database_info["db"],
            save_dir=db_backup_dir,
            prefix=db_backup_prefix,
            tz_offset=tz_offset,
            **(db_kwargs or {}),
        )
        mtree.logger.info(f"Database backup complete.")

        if backup_visits:
            last_backed_up = mtree.last_backed_up_iteration
            mtree.logger.info(
                f"Backing up states visited at steps "
                f"{last_backed_up+1}-{overall_iteration}..."
            )
            visit_results_file = Path(mtree.save_dir).joinpath(
                f"results_steps{last_backed_up+1}-{overall_iteration}"
                f"_{mtree.tree_id}_{now}.txt"
            )

            # gevent Queue object can be called as an iterator, calling get() repeatedly.
            # This clears the queue. Note that StopIteration is required, otherwise the
            # iterator won't terminate and get() will block indefinitely.
            mtree.result_history.put(StopIteration)
            visit_results = [
                ",".join(state, str(reward)) for state, reward in mtree.result_history
            ]
            visit_results_file.write_text("\n".join(visit_results))

        # Release all threads
        mtree.last_backed_up_iteration = overall_iteration
        mtree.backup_not_in_progress.set()
