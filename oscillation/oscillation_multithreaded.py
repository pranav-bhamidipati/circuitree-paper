from gevent import monkey

monkey.patch_all()

from gevent import Greenlet, getcurrent
from greenlet import greenlet
from threading import current_thread
from celery.result import AsyncResult
from pathlib import Path
import redis
from typing import Optional

from circuitree.parallel import MultithreadedCircuiTree, search_mcts_in_thread
from oscillation import OscillationGrammar

# from circuitree_monkey_patched import (
#     MultithreadedCircuiTree,
#     search_mcts_in_thread,
#     OscillationGrammar,
# )

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

        if not save_dir:
            raise ValueError("Must specify a save directory.")

        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(exist_ok=True)
        self.save_dir = str(save_dir_path.resolve().absolute())

        self.database = database
        self.n_param_sets: int = self.database.hlen("parameter_table")

        self.logger = logger or task_logger
        self.min_ssa_seed = min_ssa_seed or 0

        # Specify attributes that should not/cannot be serialized when dumping to file
        self._non_serializable_attrs.extend(["save_dir", "logger", "database"])

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
        autocorr_min = task.get()
        reward = float(-autocorr_min > self.autocorr_threshold)
        self.logger.info(
            f"Autocorr. min. of {autocorr_min:.4f} for visit {visit_number} "
            f"to state {state}. Oscillating? {bool(reward)}."
        )
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


# def search_mcts(
#     thread_idx: int,
#     mtree: MultithreadedOscillationTree,
#     n_steps: int,
#     callback_every: int,
#     **kwargs,
# ) -> MultithreadedOscillationTree:
#     return search_mcts_in_thread(
#         mtree=mtree,
#         thread_idx=thread_idx,
#         n_steps=n_steps,
#         callback=progress_callback,
#         callback_every=callback_every,
#         return_metrics=False,
#         **kwargs,
#     )
