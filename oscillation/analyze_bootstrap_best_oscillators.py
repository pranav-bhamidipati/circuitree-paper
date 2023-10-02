from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd


def main(
    data_dir: Path,
    steps_to_sample: Iterable[int],
    save: bool = False,
    save_dir: Optional[Path] = None,
    suffix: str = "",
):
    replicates = sorted(int(d.name) for d in data_dir.iterdir() if d.is_dir())

    best_oscillators = []
    best_Qs = []
    true_Qs = []

    replicate_column, step_column = zip(*product(replicates, steps_to_sample))

    for replicate, step in zip(replicate_column, step_column):
        file = next(data_dir.joinpath(str(replicate)).glob(f"*_{step}_*.csv"))
        best_row = pd.read_csv(file).sort_values("Q", ascending=False).iloc[0]
        best_oscillators.append(best_row["state"])
        best_Qs.append(best_row["Q"])
        true_Qs.append(best_row["true_Q"])

    df = pd.DataFrame(
        {
            "replicate": replicate_column,
            "step": step_column,
            "best_oscillator": best_oscillators,
            "best_Q": best_Qs,
            "true_Q": true_Qs,
        }
    )

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = save_dir.joinpath(f"{today}_best_oscillators{suffix}.csv")
        print(f"Writing to: {fpath.resolve().absolute()}")
        df.to_csv(fpath, index=False)


if __name__ == "__main__":
    import numpy as np

    suffix = "_3tf_short"
    steps_to_sample = np.linspace(0, 100_000, 101, dtype=int)[1:].tolist()
    
    # data_dir = Path("data/oscillation/mcts/bootstrap_short_230928_000000/")
    data_dir = Path("data/oscillation/mcts/bootstrap_short_231001_145157/")
    # data_dir = Path("data/oscillation/mcts/sequential_short_231001_143758/")
    # data_dir = Path("data/oscillation/mcts/sequential_short_231001_144834")



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
    )
