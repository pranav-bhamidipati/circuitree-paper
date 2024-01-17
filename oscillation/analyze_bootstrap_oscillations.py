from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, Optional
import pandas as pd
from tqdm import tqdm


def _get_best_row(df: pd.DataFrame):
    # Get row(s) with the highest total reward
    best_rows = df.loc[df["reward"] == df["reward"].max()]
    if len(best_rows) == 1:
        return best_rows.iloc[0]
    # Break ties by Q_hat
    best_rows = best_rows.loc[best_rows["Q"] == best_rows["Q"].max()]
    if len(best_rows) == 1:
        return best_rows.iloc[0]
    # Break ties by using the shortest genotype code
    gen_len = best_rows["state"].str.len()
    best_rows = best_rows.loc[gen_len == gen_len.min()]
    if len(best_rows) == 1:
        return best_rows.iloc[0]
    # Break ties by alphabetically first genotype
    return best_rows.sort_values("state").iloc[0]


def main(
    data_dir: Path,
    steps_to_sample: Iterable[int],
    save: bool = False,
    save_dir: Optional[Path] = None,
    suffix: str = "",
    progress: bool = False,
    has_true_Q: bool = True,
):
    replicates = sorted(int(d.name) for d in data_dir.iterdir() if d.is_dir())

    best_oscillators = []
    visits = []
    highest_rewards = []
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
        step_df = pd.read_csv(file)
        cum_rewards.append(step_df["reward"].sum())
        best_row = _get_best_row(step_df)
        best_oscillators.append(best_row["state"])
        visits.append(best_row["visits"])
        highest_rewards.append(best_row["reward"])
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
            "highest_reward": highest_rewards,
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

    # suffix = "_3tf_100k_iters"
    # steps_to_sample = np.linspace(0, 100_000, 101, dtype=int)[1:].tolist()
    # data_dir = Path(
    #     "data/oscillation/mcts/mcts_bootstrap_short_exploration2.00_231103_140501"
    #     "data/oscillation/mcts/sequential_bootstrap_short_231003_105603"
    # )

    suffix = "_3tf_1mil_iters"
    data_dir = Path(
        "data/oscillation/mcts/mcts_bootstrap_1mil_iters_exploration2.00_231204_142301"
    )
    steps_to_sample = np.linspace(0, 1_000_000, 101, dtype=int)[1:].tolist()

    save_dir = data_dir

    main(
        data_dir=data_dir,
        steps_to_sample=steps_to_sample,
        save=True,
        save_dir=save_dir,
        suffix=suffix,
        progress=True,
        # has_true_Q=False,
    )
