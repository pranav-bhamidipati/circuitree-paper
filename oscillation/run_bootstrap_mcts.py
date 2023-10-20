from functools import partial
import logging
from multiprocessing import Pool, active_children, current_process
from pathlib import Path
import numpy as np
import pandas as pd
from psutil import cpu_count
from typing import Any, Optional

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

        self.selection_history: list[str] = []
        self.simulation_history: list[str] = []

        self._non_serializable_attrs.extend(
            [
                "table",
                "state_col",
                "Q_col",
                "_Q",
                "logger",
                "save_dir",
                "selection_history",
                "simulation_history",
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

    def get_random_terminal_descendant(self, start: Any) -> Any:
        """**Identical to the CircuiTree class method, except it keeps a record of the
        selected node**

        Uses the random generator for the given thread to select a state to simulate,
        starting from the given starting state. If the given starting state is terminal,
        returns it. Otherwise, selects a random child recursively until a terminal state
        is reached."""
        self.selection_history.append(start)
        return super().get_random_terminal_descendant(start)

    def get_reward(self, state: str) -> float:
        self.simulation_history.append(state)
        self.iter_count += 1
        if self.iter_count >= self.next_save:
            self.next_save += self.save_every
            self.save_sample_data()
        reward = float(self.rg.uniform(0, 1) < self.Q[state])
        return reward

    def get_sample_data(self):
        # Get the selected and simulated nodes and clear the history
        selections = self.selection_history.copy()
        simulations = self.simulation_history.copy()
        self.selection_history.clear()
        self.simulation_history.clear()

        states, rewards, visits = zip(*self.rewards_and_visits)
        return selections, simulations, states, rewards, visits

    def get_oscillation_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        selections, simulations, states, rewards, visits = self.get_sample_data()
        Qs = np.divide(rewards, visits, out=np.zeros_like(rewards), where=visits != 0)
        true_Qs = [self.Q[s] for s in states]
        qdf = pd.DataFrame(
            {
                "state": states,
                "reward": rewards,
                "visits": visits,
                "Q": Qs,
                "true_Q": true_Qs,
            }
        )
        qdf = qdf.sort_values("Q", ascending=False)

        sample_data = pd.DataFrame(
            {"selected_node": selections, "simulated_node": simulations}
        )

        return qdf, sample_data

    def save_sample_data(self):
        iteration = self.iter_count
        save_stem = f"oscillation_mcts_bootstrap_{iteration}"
        self.logger.info(f"Iteration {iteration}")

        self.logger.info(f"Saving tree data to {save_stem}*")
        gml_target = self.save_dir.joinpath(f"{save_stem}_tree.gml")
        attrs_target = self.save_dir.joinpath(f"{save_stem}_tree.json")
        self.to_file(gml_target, attrs_target)

        oscillation_df, sample_data = self.get_oscillation_data()

        df_target = self.save_dir.joinpath(f"{save_stem}_oscillators.csv")
        samples_target = self.save_dir.joinpath(f"{save_stem}_samples.csv")
        self.logger.info(f"Saving oscillation data to {df_target}")
        oscillation_df.to_csv(df_target)
        self.logger.info(f"Saving samples to {samples_target}")
        sample_data.to_csv(samples_target)


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
    ot.search_mcts(n_steps=n_steps, progress_bar=progress_bar)
    ot.save_sample_data()
    logger.info(f"Done")


def main(
    oscillator_table_csv: Path,
    save_dir: Path,
    log_dir: Path,
    n_steps: int,
    save_every: int,
    n_replicates: int,
    master_seed: int,
    progress_bar: bool = True,
    n_workers: Optional[int] = None,
    state_col: str = "state",
    Q_col: str = "p_oscillation",
    Q_threshold: float = 0.01,
):
    n_workers = min(n_replicates, n_workers or cpu_count())
    print(f"Running {n_replicates} searches in parallel with {n_workers} workers.")
    print(f"Saving results to {save_dir.resolve().absolute()}")

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

    if n_workers == 1:
        print("Running searches in serial")
        for args in search_args:
            run_search(args, **kw)

    else:
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

    save_dir = Path(f"data/oscillation/mcts/mcts_bootstrap_short_{now}")
    log_dir = Path(f"logs/oscillation/mcts/mcts_bootstrap_short_{now}")

    # save_dir = Path(f"data/oscillation/mcts/mcts_bootstrap_long_{now}")
    # log_dir = Path(f"logs/oscillation/mcts/mcts_bootstrap_long_{now}")

    save_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    main(
        save_dir=save_dir,
        log_dir=log_dir,
        oscillator_table_csv=oscillation_table_csv,
        # n_workers=1,
        n_workers=13,
        n_replicates=50,
        master_seed=2023,
        n_steps=100_000,
        save_every=1_000,
        # n_steps=5_000_000,
        # save_every=100_000,
    )
