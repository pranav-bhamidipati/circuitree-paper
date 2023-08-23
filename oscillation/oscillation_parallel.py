from multiprocessing import Lock
from multiprocessing.pool import Pool
import numpy as np
from time import perf_counter
from typing import Any, Iterable, Optional

from circuitree.parallel import TranspositionTable, ParameterTable, ParallelTree
import pandas as pd
from tf_network import TFNetworkModel
from oscillation import OscillationTree

__all__ = ["OscillationTreeParallel"]


def _run_ssa(visit, state, dt, nt):
    start = perf_counter()
    model = TFNetworkModel(state)
    model.initialize_ssa(seed=visit, dt=dt, nt=nt)
    y_t, pop0, params, reward = model.run_batch_with_params(
        size=1,
        freqs=False,
        indices=False,
        abs=True,
    )
    end = perf_counter()
    sim_time = end - start
    return reward, y_t, pop0, params, sim_time

class OscillationTreeParallel(OscillationTree, ParallelTree):
    """Searches the space of TF networks for oscillatory topologies.
    Each step of the search takes the average of multiple draws.
    Uses a transposition table to access and store reward values. If desired
    results are not present in the table, they will be computed in parallel.
    Random seeds, parameter sets, and initial conditions are selected from the
    parameter table `self.param_table`.

    The `simulate_visits` and `save_results` methods must be implemented in a
    subclass.

    An invalid simulation result (e.g. a timeout) should be represented by a
    NaN reward value. If all rewards in a batch are NaNs, a new batch will be
    drawn from the transposition table. Otherwise, any NaN rewards will be
    ignored and the mean of the remaining rewards will be returned as the
    reward for the batch.
    """

    def __init__(
        self,
        *,
        pool: Optional[Pool] = None,
        nprocs: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pool = Pool(nprocs) if pool is None else pool
        self.n_procs = self.pool._processes

    @property
    def pool(self) -> Pool:
        return self._pool

    def simulate_visits(self, state, visits) -> tuple[list[float], dict[str, Any]]:
        """Should return a list of reward values and a dictionary of any data
        to be analyzed. Takes a state and a list of which visits to simulate.
        Random seeds, parameter sets, and initial conditions are selected from
        the parameter table `self.param_table`."""
        
        raise NotImplementedError

    def save_results(self, state, visits, rewards, data: dict[str, Any]) -> None:
        """Optionally save the results of simulated visits. May or may not update the
        transposition table. Parameter sets and initial conditions can be accessed from
        the parameter table `self.param_table`."""
        raise NotImplementedError


# states_and_seeds = zip(repeat(genotype), range(start_idx, end_idx))
# run_ssa = partial(_run_ssa, dt=self.dt, nt=self.nt)
# sim_results = self.pool.map(run_ssa, states_and_seeds)

# def _write_results(
#     self,
#     table: TranspositionTable,
#     state: str,
#     rewards: Iterable[float],
#     lock: Optional[Any] = None,
#     timeout: Optional[float] = None,
# ):
#     if len(rewards) == 0:
#         return
#     lock = lock or Lock()
#     with lock(timeout=timeout):
#         self.ttable[state].extend(list(rewards))
