from collections import Counter
from functools import partial
from itertools import repeat
from multiprocessing.pool import Pool
from threading import Lock
from typing import Any, Iterable, Optional

import numpy as np

from circuitree.parallel import MCTSResult, TranspositionTable
from oscillation import TFNetworkModel, OscillationTree

__all__ = ["OscillationTreeParallel"]


def _run_ssa(genotype_visit, dt, nt):
    genotype, visit = genotype_visit
    model = TFNetworkModel(genotype)
    model.initialize_ssa(seed=visit, dt=dt, nt=nt)
    y_t, pop0, params, reward = model.run_ssa_and_get_acf_minima(
        size=1,
        freqs=False,
        indices=False,
        abs=True,
    )
    return reward, genotype, visit, y_t, pop0, params


class OscillationTreeParallel(OscillationTree):
    """Searches the space of TF networks for oscillatory topologies.
    Each step of the search takes the average of multiple draws.
    Uses a transposition table to store and access results. If desired
    results are not present in the table, they will be computed in parallel
    and added to the table."""

    # TODO: Put all the multiprocessing elements in a mixin class??

    def __init__(
        self,
        pool: Optional[Pool] = None,
        n_threads: int = 1,
        transposition_table: Optional[TranspositionTable] = None,
        columns: Optional[Iterable[str]] = None,
        counter: Counter = None,
        read_only: bool = False,
        table_lock: Optional[Any] = None,
        bootstrap: bool = False,
        **kwargs,
    ):
        if pool is None:
            if bootstrap:
                self._pool = None
            else:
                self._pool = Pool(n_threads)
        else:
            self._pool = pool

        if self._pool is not None:
            kwargs.update({"batch_size": pool._processes})

        super().__init__(**kwargs)

        self.columns = columns
        if transposition_table is None:
            self._transposition_table = TranspositionTable(columns=self.columns)
        else:
            self._transposition_table = transposition_table

        self.visit_counter = Counter(counter)
        self.read_only = read_only

        if table_lock is None:
            table_lock = Lock()
        self._table_lock = table_lock

        # Specify attributes that should not be serialized when dumping to file
        self._non_serializable_attrs.extend(
            ["_pool", "_transposition_table", "visit_counter", "_table_lock"]
        )

        self.bootstrap = bootstrap
        if self.bootstrap and not self.read_only:
            raise ValueError(
                "Can only use bootstrap sampling with a read-only transposition table."
            )

    @property
    def transposition_table(self):
        return self._transposition_table

    @property
    def pool(self):
        return self._pool

    @property
    def table_lock(self):
        return self._table_lock

    def get_reward(self, genotype, write_timeout: Optional[float] = None):
        visit_num = self.visit_counter[genotype]

        if self.bootstrap:
            rewards = self.transposition_table.draw_bootstrap_reward(
                state=genotype, size=self.batch_size, rg=self.rg
            )
        else:
            n_recorded_visits = self.transposition_table.n_visits(genotype)
            start_idx = max(visit_num, n_recorded_visits)
            end_idx = max(visit_num + self.batch_size, n_recorded_visits)
            n_to_simulate = end_idx - start_idx

            if n_to_simulate == 0:
                sim_results = []
            else:
                states_and_seeds = zip(repeat(genotype), range(start_idx, end_idx))
                run_ssa = partial(_run_ssa, dt=self.dt, nt=self.nt)
                sim_results = self.pool.map(run_ssa, states_and_seeds)

            rewards = []
            for i in range(self.batch_size - n_to_simulate):
                visit = visit_num + i
                r = self.transposition_table.get_reward(genotype, visit)
                rewards.append(r)
            rewards = rewards + [r[0] for r in sim_results]
            rewards = np.array(rewards)

            if (not self.read_only) and (n_to_simulate > 0):
                self._write_results(
                    self.transposition_table,
                    genotype,
                    rewards[-n_to_simulate:],
                    lock=self.table_lock,
                    timeout=write_timeout,
                )

        raise NotImplementedError

        ...

        # Need to insert logic here to deal with nans in rewards!!
        # Should take a parameter to decide what a nan means
        #   - remove nan from batch and take mean of rest
        #       - catch case of all nans and return 0 ?
        #   - nan means re-run entire batch
        #   - nan means re-run just the nan

        # *** Save nan runs' details for later inspection ***

        self.visit_counter[genotype] += self.batch_size

        return (np.array(rewards) > self.autocorr_threshold).mean()

    def _write_results(
        self,
        table: TranspositionTable,
        state: str,
        rewards: np.ndarray | list[float],
        lock: Optional[Any] = None,
        timeout: Optional[float] = None,
    ):
        if len(rewards) == 0:
            return

        if lock is None:
            if self.table_lock is None:
                lock = Lock()
            else:
                lock = self.table_lock
        with lock(timeout=timeout):
            table[state].extend(list(rewards))
