from functools import partial
from multiprocessing import Pool, current_process
from pathlib import Path
from numba import njit
import numpy as np
import pandas as pd
from psutil import cpu_count
from typing import Optional


@njit
def _run_n_steps_inplace(
    n: int,
    iteration: int,
    sample_indices: np.ndarray,
    samples: np.ndarray,
    reward: np.ndarray,
    rg: np.random.Generator,
    order: np.ndarray,
    Qs: np.ndarray,
):
    n_states = len(order)
    for i in range(n):
        which_state = order[iteration % n_states]
        sample_indices[i] = which_state
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
    ):
        self.rg = rg

        self.states = Q_table["state"].values
        self.Q = Q_table["p_oscillation"].values
        self.n_states = len(self.states)

        self.visits = np.zeros(self.n_states, dtype=np.int64)
        self.reward = np.zeros(self.n_states, dtype=np.int64)
        self.iteration = 0

        order = np.arange(len(self.states))
        self.rg.shuffle(order)
        self.order = order

        self.save_dir = Path(save_dir)

        self.sample_indices: np.ndarray[int] | None = None

    def run_n_steps_inplace(self, n_steps: int):
        self.sample_indices.fill(-1)
        _run_n_steps_inplace(
            n_steps,
            self.iteration,
            self.sample_indices,
            self.visits,
            self.reward,
            self.rg,
            self.order,
            self.Q,
        )
        self.iteration += n_steps

    def run_search(self, n_steps: int, save_every: int, print_every: int = np.inf):
        self.sample_indices = -np.ones(save_every, dtype=np.int64)
        print(self.sample_indices.size)
        n_chunks, remainder = divmod(n_steps, save_every)
        n_steps_per_chunk = [save_every] * n_chunks
        if remainder:
            n_steps_per_chunk.append(remainder)

        print(f"Running {n_steps} search iterations in {len(n_steps_per_chunk)} chunks")

        next_print = print_every
        for i, n_steps_in_chunk in enumerate(n_steps_per_chunk):
            self.run_n_steps_inplace(n_steps_in_chunk)
            self.save_data()
            if self.iteration >= next_print:
                print(f"Progress: {self.iteration/n_steps:.2%}")
                next_print += print_every
        print("Done!")

    def samples_to_dataframe(self) -> pd.DataFrame:
        Q_search = np.divide(
            self.reward,
            self.visits,
            out=np.zeros_like(self.Q),
            where=self.visits != 0,
        )
        results_data = pd.DataFrame(
            {
                "state": self.states,
                "reward": self.reward,
                "visits": self.visits,
                "Q": Q_search,
            }
        )
        results_data = results_data.sort_values("visits", ascending=False)
        return results_data

    def save_data(self):
        results_data = self.samples_to_dataframe()
        iteration = self.iteration
        results_csv = self.save_dir.joinpath(
            f"sequential_bootstrap_{iteration}_oscillators.csv"
        )
        results_data.to_csv(results_csv)
        samples_csv = self.save_dir.joinpath(
            f"sequential_bootstrap_{iteration}_samples.csv"
        )
        sampled_states = self.states[self.sample_indices[self.sample_indices != -1]]
        samples_csv.write_text("\n".join(sampled_states))


def do_one_search(
    seed_and_save_dir: tuple[int, Path],
    n_steps: int,
    save_every: int,
    oscillator_table_csv: Path,
    print_every: int,
):
    seed, save_dir = seed_and_save_dir
    save_dir.mkdir(exist_ok=True)

    rg = np.random.default_rng(seed)
    oscillator_table = pd.read_csv(oscillator_table_csv)
    sampler = SequentialSampler(
        Q_table=oscillator_table,
        rg=rg,
        save_dir=save_dir,
    )
    pid = current_process().pid
    print("Starting search in process:", pid)
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
):
    n_workers = min(n_replicates, n_workers or cpu_count())
    print(f"Running {n_replicates} searches in parallel with {n_workers} workers")
    print(f"Saving results to {save_dir}")

    kw = dict(
        n_steps=n_steps,
        save_every=save_every,
        oscillator_table_csv=oscillator_table_csv,
        print_every=print_every,
    )

    parent_ss = np.random.SeedSequence(parent_seed)
    child_seeds = [np.random.default_rng(s) for s in parent_ss.spawn(n_replicates)]
    save_dirs = (save_dir.joinpath(f"{i}") for i in range(n_replicates))
    search_args = zip(child_seeds, save_dirs)

    if n_workers == 1:
        print(
            f"Running {n_replicates} searches in main process and "
            f"saving results to {save_dir}"
        )
        for i, args in enumerate(search_args):
            print(f"Running search {i+1} of {n_replicates}")
            do_one_search(args, **kw)
        print("Done with all searches!")

    else:
        print("Starting pool...")
        with Pool(n_workers) as pool:
            do_job_in_process = partial(do_one_search, **kw)
            pool.map(do_job_in_process, search_args)
            print(f"Finished running searches and saving results to {save_dir}")
            print("Done! Closing pool")


if __name__ == "__main__":
    from datetime import datetime

    now = datetime.now().strftime("%y%m%d_%H%M%S")

    oscillation_table_csv = Path("data/oscillation/230717_motifs.csv")

    save_dir = Path(f"data/oscillation/mcts/sequential_bootstrap_short_{now}")
    # save_dir = Path(f"data/oscillation/mcts/sequential_bootstrap_long_{now}")
    save_dir.mkdir(exist_ok=True)

    main(
        save_dir=save_dir,
        oscillator_table_csv=oscillation_table_csv,
        n_workers=1,
        # n_workers=12,
        n_replicates=50,
        parent_seed=2023,
        n_steps=100_000,
        print_every=20_000,
        save_every=1_000,
        # n_steps=34_110_000,
        # save_every=100_000,
        # print_every=34_110_000,
    )
