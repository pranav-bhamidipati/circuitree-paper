from functools import partial
import logging
from multiprocessing import Pool, active_children, current_process
from pathlib import Path
from numba import njit
import numpy as np
import pandas as pd
from psutil import cpu_count
from typing import Optional


@njit
def _run_n_steps_inplace(
    iteration: int,
    n: int,
    samples: np.ndarray,
    reward: np.ndarray,
    rg: np.random.Generator,
    order: np.ndarray,
    Qs: np.ndarray,
):
    n_states = len(order)
    for _ in range(n):
        which_state = order[iteration % n_states]
        samples[which_state] += 1
        if rg.uniform() < Qs[which_state]:
            reward[which_state] += 1
        iteration += 1


class SequentialSampler:
    def __init__(
        self,
        Q_table: pd.DataFrame,
        rg: np.random.Generator,
        save_dir: Path,
        Q_threshold: float = 0.01,
    ):
        self.rg = rg

        self.states = Q_table["state"].values
        self.Q = Q_table["p_oscillation"].values
        self.n_states = len(self.states)

        self.samples = np.zeros(self.n_states, dtype=np.int64)
        self.reward = np.zeros(self.n_states, dtype=np.int64)
        self.iteration = 0

        order = np.arange(len(self.states))
        self.rg.shuffle(order)
        self.order = order

        self.Q_threshold = Q_threshold
        self.save_dir = Path(save_dir)

    def run_n_steps_inplace(self, n_steps: int):
        _run_n_steps_inplace(
            self.iteration,
            n_steps,
            self.samples,
            self.reward,
            self.rg,
            self.order,
            self.Q,
        )
        self.iteration += n_steps

    def run_search(self, n_steps: int, save_every: int, print_every: int = np.inf):
        n_chunks, remainder = divmod(n_steps, save_every)
        n_steps_per_chunk = [save_every] * n_chunks
        if remainder:
            n_steps_per_chunk.append(remainder)

        print(f"Running {n_steps} search iterations in {len(n_steps_per_chunk)} chunks")

        next_print = print_every
        for i, n_steps in enumerate(n_steps_per_chunk):
            self.run_n_steps_inplace(n_steps)
            self.save_oscillation_data()
            if self.iteration >= next_print:
                print(f"Finished {self.iteration} iterations")
                next_print += print_every

    def oscillators_to_dataframe(self) -> pd.DataFrame:
        Q_search = np.divide(
            self.reward,
            self.samples,
            out=np.zeros_like(self.Q),
            where=self.samples != 0,
        )

        where_osc = np.where(Q_search >= self.Q_threshold)[0]
        df = pd.DataFrame(
            {
                "state": self.states[where_osc],
                "Q": Q_search[where_osc],
                "true_Q": self.Q[where_osc],
            }
        )
        df = df.sort_values("Q", ascending=False)
        return df

    def save_oscillation_data(self):
        oscillator_data = self.oscillators_to_dataframe()
        iteration = self.iteration
        target_csv = self.save_dir.joinpath(
            f"oscillation_sequential_bootstrap_{iteration}.csv"
        )
        oscillator_data.to_csv(target_csv)


def do_one_search(
    seed_and_save_dir: tuple[int, Path],
    n_steps: int,
    save_every: int,
    oscillator_table_csv: Path,
    print_every: int,
    Q_threshold: float = 0.01,
):
    seed, save_dir = seed_and_save_dir
    save_dir.mkdir(exist_ok=True)

    rg = np.random.default_rng(seed)
    oscillator_table = pd.read_csv(oscillator_table_csv)
    sampler = SequentialSampler(
        Q_table=oscillator_table, rg=rg, save_dir=save_dir, Q_threshold=Q_threshold
    )
    print("Starting search in process:", current_process().pid)
    sampler.run_search(n_steps, save_every, print_every=print_every)


def main(
    oscillator_table_csv: Path,
    save_dir: Path,
    n_replicates: int,
    parent_seed: int,
    n_steps: int = 34_110_000,
    save_every: int = 10_000,
    n_workers: Optional[int] = None,
    print_every: int = 10_000,
    Q_threshold: float = 0.01,
):
    n_workers = min(n_replicates, n_workers or cpu_count())
    print(f"Running {n_replicates} searches in parallel with {n_workers} workers")

    kw = dict(
        n_steps=n_steps,
        save_every=save_every,
        oscillator_table_csv=oscillator_table_csv,
        Q_threshold=Q_threshold,
        print_every=print_every,
    )

    parent_ss = np.random.SeedSequence(parent_seed)
    child_seeds = [np.random.default_rng(s) for s in parent_ss.spawn(n_replicates)]
    save_dirs = (save_dir.joinpath(f"{i}") for i in range(n_replicates))
    search_args = zip(child_seeds, save_dirs)

    if n_workers == 1:
        for i, args in enumerate(search_args):
            print(f"Running search {i+1} of {n_replicates}")
            do_one_search(args, **kw)
    else:
        print("Starting pool...")
        with Pool(n_workers) as pool:
            do_job_in_process = partial(do_one_search, **kw)
            pool.map(do_job_in_process, search_args)
            print("Done! Closing pool")


if __name__ == "__main__":
    from datetime import datetime

    now = datetime.now().strftime("%y%m%d_%H%M%S")

    oscillation_table_csv = Path("data/oscillation/230717_motifs.csv")

    save_dir = Path(f"data/oscillation/mcts/sequential_long_{now}")
    # save_dir = Path(f"data/oscillation/mcts/sequential_long_{now}")
    save_dir.mkdir(exist_ok=True)

    main(
        save_dir=save_dir,
        oscillator_table_csv=oscillation_table_csv,
        n_workers=1,
        n_replicates=12,
        parent_seed=2023,
        # n_steps=100_000,
        # print_every=10_000,
        # save_every=10_000,
        n_steps=34_110_000,
        save_every=100_000,
        print_every=34_110_000,
    )
