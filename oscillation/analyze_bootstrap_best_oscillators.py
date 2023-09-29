from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd


def main(
    data_dir: Path,
    steps_to_sample: Iterable[int] = [30_000, 40_000, 50_000, 60_000],
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
    data_dir = Path("data/oscillation/mcts/bootstrap_short_230928_000000/")
    # data_dir = Path("data/oscillation/mcts/bootstrap_long_230928_000832/")

    save_dir = data_dir
    
    main(
        data_dir=data_dir,
        save=True,
        save_dir=save_dir,
        suffix="_3tf_short",
        # suffix="_3tf_long",
    )
