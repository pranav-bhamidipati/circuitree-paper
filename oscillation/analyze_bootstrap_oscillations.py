from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, Optional
import pandas as pd
from tqdm import tqdm


def _get_best_row(
    df: pd.DataFrame,
    metric: str | Callable,
):
    if hasattr(metric, "__call__"):
        values = metric(df)
        best_idx = np.argmax(values)
        return df.iloc[best_idx]
    elif isinstance(metric, str):
        return df.sort_values(metric, ascending=False).iloc[0]


def main(
    data_dir: Path,
    steps_to_sample: Iterable[int],
    save: bool = False,
    save_dir: Optional[Path] = None,
    suffix: str = "",
    metric: str | Callable = "reward",
    progress: bool = False,
    has_true_Q: bool = True,
):
    replicates = sorted(int(d.name) for d in data_dir.iterdir() if d.is_dir())

    best_oscillators = []
    visits = []
    rewards = []
    best_Qs = []
    cum_rewards = []

    if has_true_Q:
        true_Qs = []

    replicate_column, step_column = zip(*product(replicates, steps_to_sample))

    iterator = list(zip(replicate_column, step_column))
    if progress:
        iterator = tqdm(iterator)

    for replicate, step in iterator:
        file = next(data_dir.joinpath(str(replicate)).glob(f"*_{step}_oscillators.csv"))
        df = pd.read_csv(file)
        cum_rewards.append(df["reward"].sum())
        best_row = _get_best_row(df, metric=metric)
        best_oscillators.append(best_row["state"])
        visits.append(best_row["visits"])
        rewards.append(best_row["reward"])
        best_Qs.append(best_row["Q"])
        if has_true_Q:
            true_Qs.append(best_row["true_Q"])

    df = pd.DataFrame(
        {
            "replicate": replicate_column,
            "step": step_column,
            "cum_reward": cum_rewards,
            "best_oscillator": best_oscillators,
            "visits": visits,
            "reward": rewards,
            "best_Q": best_Qs,
            **({"true_Q": true_Qs} if has_true_Q else {}),
        }
    )

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = save_dir.joinpath(f"{today}_oscillation_data{suffix}.csv")
        print(f"Writing to: {fpath.resolve().absolute()}")
        df.to_csv(fpath, index=False)


if __name__ == "__main__":
    import numpy as np

    suffix = "_3tf_short"
    steps_to_sample = np.linspace(0, 100_000, 101, dtype=int)[1:].tolist()

    data_dir = Path("data/oscillation/mcts/mcts_bootstrap_short_231002_221756")
    has_true_Q = True

    # data_dir = Path("data/oscillation/mcts/sequential_bootstrap_short_231003_105603")
    # has_true_Q = False

    # steps_to_sample = np.linspace(0, 5_000_000, 51, dtype=int)[1:].tolist()
    # suffix = "_3tf_long"
    # data_dir = Path("data/oscillation/mcts/bootstrap_long_230928_000832/")
    # data_dir = Path("data/oscillation/mcts/sequential_long_230929_170434/")

    save_dir = data_dir

    main(
        data_dir=data_dir,
        save=True,
        save_dir=save_dir,
        suffix=suffix,
        steps_to_sample=steps_to_sample,
        progress=True,
        has_true_Q=has_true_Q,
    )
