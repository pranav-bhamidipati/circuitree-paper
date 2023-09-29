from functools import partial
import logging
from multiprocessing import Pool, active_children, current_process
from pathlib import Path
import numpy as np
import pandas as pd
from psutil import cpu_count
from typing import Optional

from oscillation import OscillationTree


class BootstrapOscillationTree(OscillationTree):
    def __init__(
        self,
        table: pd.DataFrame,
        state_col: str = "state",
        Q_col: str = "p_oscillation",
        Q_threshold: float = 0.01,
        save_dir: Path = None,
        save_every: int = 10_000,
        logger: logging.Logger = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if state_col not in table.columns:
            raise ValueError(f"State column not found: {state_col}")
        if Q_col not in table.columns:
            raise ValueError(f"Q column not found: {Q_col}")

        self.table = table.reset_index()
        self.state_col = state_col
        self.Q_col = Q_col

        self._Q = dict(zip(self.table[self.state_col], self.table[self.Q_col]))
        self.Q_threshold = Q_threshold

        self.iter_count = 0

        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.next_save = save_every

        self.logger = logger or logging.getLogger(__name__)

        self._non_serializable_attrs.extend(
            [
                "table",
                "state_col",
                "Q_col",
                "_Q",
                "logger",
                "save_dir",
            ]
        )

    @property
    def rewards_and_visits(self):
        return (
            (n, self.graph.nodes[n]["reward"], self.graph.nodes[n]["visits"])
            for n in self.terminal_states
        )

    @property
    def Q(self) -> dict[str, float]:
        return self._Q

    def get_reward(self, state: str) -> float:
        self.iter_count += 1
        if self.iter_count >= self.next_save:
            self.next_save += self.save_every
            self.save_oscillation_data()
        reward = float(self.rg.uniform(0, 1) < self.Q[state])
        return reward

    def get_oscillation_data(self) -> tuple[list[str], list[float]]:
        osc_data = []
        qs = []
        for n, reward, visits in self.rewards_and_visits:
            q = reward / max(visits, 1)
            osc_data.append(n)
            qs.append(q)
        return osc_data, qs

    def oscillators_to_dataframe(self) -> pd.DataFrame:
        states, Qs = self.get_oscillation_data()
        true_Qs = [self.Q[s] for s in states]
        df = pd.DataFrame({"state": states, "Q": Qs, "true_Q": true_Qs})
        df = df.sort_values("Q", ascending=False)
        return df

    def save_oscillation_data(self):
        oscillator_data = self.oscillators_to_dataframe()
        iteration = self.iter_count
        save_stem = f"oscillation_mcts_bootstrap_{iteration}"

        self.logger.info(f"Iteration {iteration}")

        self.logger.info(f"Saving tree data to {save_stem}*")
        gml_target = self.save_dir.joinpath(f"{save_stem}_tree.gml")
        attrs_target = self.save_dir.joinpath(f"{save_stem}_tree.json")
        self.to_file(gml_target, attrs_target)

        df_target = self.save_dir.joinpath(f"{save_stem}_oscillators.csv")
        self.logger.info(f"Saving oscillator data to {df_target}")
        oscillator_data.to_csv(df_target)


def run_search(
    seed_and_save_dir: tuple[int, Path],
    log_dir: Path,
    n_steps: int,
    save_every: int,
    oscillator_table_csv: Path,
    state_col: str = "state",
    Q_col: str = "p_oscillation",
    Q_threshold: float = 0.01,
    progress_bar: bool = True,
    replicate_ids: Optional[dict[int, int]] = None,
    **kwargs,
):
    tree_seed, save_dir = seed_and_save_dir

    pid = current_process().pid
    replicate_id = (replicate_ids or {}).get(pid, 0)

    logfile = Path(log_dir) / f"process-{replicate_id}-PID{pid}.log"
    logging.basicConfig(
        filename=str(logfile),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.info(
        f"Started replicate {replicate_id} with seed {tree_seed} in process {pid}"
    )

    logger.info(
        f"Loading table of 3-TF oscillator search results from {oscillator_table_csv}"
    )
    oscillator_table = pd.read_csv(oscillator_table_csv, index_col=0)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    ot = BootstrapOscillationTree(
        table=oscillator_table,
        seed=tree_seed,
        state_col=state_col,
        Q_col=Q_col,
        Q_threshold=Q_threshold,
        save_dir=save_dir,
        save_every=save_every,
        logger=logger,
        **kwargs,
    )

    logger.info(f"Running MCTS for {n_steps} steps")
    ot.search_mcts(
        n_steps=n_steps, progress_bar=progress_bar, pbar_position=replicate_id
    )
    ot.save_oscillation_data()
    logger.info(f"Done")


def main(
    oscillator_table_csv: Path,
    save_dir: Path,
    log_dir: Path,
    n_replicates: int,
    master_seed: int,
    n_steps: int = 34_110_000,
    save_every: int = 10_000,
    progress_bar: bool = True,
    n_workers: Optional[int] = None,
    state_col: str = "state",
    Q_col: str = "p_oscillation",
    Q_threshold: float = 0.01,
):
    n_workers = min(n_replicates, n_workers or cpu_count())
    print(f"Running {n_replicates} searches in parallel with {n_workers} workers")

    kw = dict(
        root="ABC::",
        components=["A", "B", "C"],
        interactions=["activates", "inhibits"],
        n_steps=n_steps,
        save_every=save_every,
        log_dir=log_dir,
        oscillator_table_csv=oscillator_table_csv,
        state_col=state_col,
        Q_col=Q_col,
        Q_threshold=Q_threshold,
        progress_bar=progress_bar,
    )

    seed_seq = np.random.SeedSequence(master_seed)
    prng_seeds = map(int, seed_seq.generate_state(n_replicates))
    save_dirs = (save_dir.joinpath(f"{i}") for i in range(n_replicates))

    search_args = zip(prng_seeds, save_dirs)

    with Pool(n_workers) as pool:
        # Get the index of each process in the pool
        kw["replicate_ids"] = {c.pid: i for i, c in enumerate(active_children())}
        run_search_in_process = partial(run_search, **kw)

        # Run the searches in parallel
        print("Running searches in parallel")
        pool.map(run_search_in_process, search_args)

        print("Done! Closing pool")


if __name__ == "__main__":
    from datetime import datetime

    now = datetime.now().strftime("%y%m%d_%H%M%S")

    oscillation_table_csv = Path("data/oscillation/230717_motifs.csv")

    save_dir = Path(f"data/oscillation/mcts/bootstrap_short_{now}")
    save_dir.mkdir(exist_ok=True)
    log_dir = Path(f"logs/oscillation/mcts/bootstrap_{now}")
    log_dir.mkdir(exist_ok=True)

    main(
        save_dir=save_dir,
        log_dir=log_dir,
        oscillator_table_csv=oscillation_table_csv,
        # n_replicates=1,
        n_replicates=12,
        master_seed=2023,
        # n_steps=100_000,
        # save_every=1_000,
        n_steps=5_000_000,
        save_every=100_000,
    )
