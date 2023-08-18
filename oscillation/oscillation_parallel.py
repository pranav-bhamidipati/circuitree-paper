from collections import Counter
from functools import partial
from itertools import repeat
from multiprocessing.pool import Pool
import numpy as np
from threading import Lock
from typing import Any, Iterable, Optional
import warnings

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
        warn_if_nan: bool = False,
        handle_nan: str = "omit",
        record_nans: bool = True,
        nan_default: float = 0.0,
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

        self.warn_if_nan = warn_if_nan
        self.handle_nan = handle_nan
        self.nan_default = nan_default
        self.record_nans = record_nans

        self._nan_data = []

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
            bootstrap_indices, rewards = self.transposition_table.draw_bootstrap_reward(
                state=genotype, size=self.batch_size, rg=self.rg
            )

        else:
            n_recorded_visits = self.transposition_table.n_visits(genotype)
            start_idx = max(visit_num, n_recorded_visits)
            end_idx = max(visit_num + self.batch_size, n_recorded_visits)
            n_to_simulate = end_idx - start_idx
            n_to_read = self.batch_size - n_to_simulate

            if n_to_simulate == 0:
                sim_results = []
            else:
                states_and_seeds = zip(repeat(genotype), range(start_idx, end_idx))
                run_ssa = partial(_run_ssa, dt=self.dt, nt=self.nt)
                sim_results = self.pool.map(run_ssa, states_and_seeds)

            visit_indices = np.arange(visit_num, visit_num + n_to_read)
            rewards = [
                self.transposition_table.get_reward(genotype, v) for v in visit_indices
            ]
            rewards.extend([r[0] for r in sim_results])
            rewards = np.array(rewards)

            if (not self.read_only) and (n_to_simulate > 0):
                self._write_results(
                    self.transposition_table,
                    genotype,
                    rewards[-n_to_simulate:],
                    lock=self.table_lock,
                    timeout=write_timeout,
                )

        rewards = np.array(rewards)
        nan_rewards = np.isnan(rewards)
        if nan_rewards.all():
            if self.warn_if_nan:
                warnings.warn(
                    f"All rewards for {genotype} are nan. Re-running batch..."
                )
                return self.get_reward(genotype, write_timeout=write_timeout)
        elif nan_rewards.any():
            if self.warn_if_nan:
                warnings.warn(f"Some rewards for {genotype} are nan.")
            if self.handle_nan == "omit":
                if self.warn_if_nan:
                    warnings.warn(f"Omitting nans for reward calculation.")
                reward = (rewards[~nan_rewards] > self.autocorr_threshold).mean()
            elif self.handle_nan == "rerun":
                if self.warn_if_nan:
                    warnings.warn(f"Re-running batch...")
                return self.get_reward(genotype, write_timeout=write_timeout)
            elif self.handle_nan == "default":
                if self.warn_if_nan:
                    warnings.warn(
                        f"Replacing nans with default value: {self.nan_default}."
                    )
                rewards[nan_rewards] = self.nan_default
                reward = rewards.mean()
            else:
                raise ValueError(
                    f"Unrecognized value for handle_nan: {self.handle_nan}"
                )
        else:
            reward = rewards.mean()

        if self.record_nans and nan_rewards.any():
            n_nans = nan_rewards.sum()
            if self.bootstrap:
                nan_data = {
                    "genotype": [genotype] * n_nans,
                    "indices": bootstrap_indices[nan_rewards],
                }
            else:
                genotypes = [genotype] * n_nans
                indices = []
                seeds = []
                pop0s = []
                param_sets = []
                for i, isnan in enumerate(nan_rewards):
                    if not isnan:
                        continue
                    if i < n_to_read:
                        indices.append(bootstrap_indices[i])
                    else:
                        sim_idx = i - n_to_read
                        indices.append(-1)
                        seeds.append(sim_results[sim_idx][2])
                        pop0s.append(sim_results[sim_idx][4])
                        param_sets.append(sim_results[sim_idx][5])

                nan_data = {
                    "genotype": genotypes,
                    "indices": np.array(indices),
                    "seeds": np.array(seeds),
                    "pop0s": np.array(pop0s),
                    "param_sets": np.array(param_sets),
                }

            self._register_nan_data(nan_data)

        self.visit_counter[genotype] += self.batch_size

        return reward

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

    def _register_nan_data(self, nan_data: dict[str, list]):
        self._nan_data.append(nan_data)
