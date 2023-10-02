from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns


def main(
    mcts_csv: Path,
    sequential_csv: Path,
    figsize: tuple = (3, 3),
    max_step: int = 100_000,
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
    **kwargs,
):
    df_mcts = pd.read_csv(mcts_csv)
    df_mcts = df_mcts.loc[df_mcts["step"] <= max_step]
    df_mcts["method"] = "MCTS"
    df_sequential = pd.read_csv(sequential_csv)
    df_sequential = df_sequential.loc[df_sequential["step"] <= max_step]
    df_sequential["method"] = "Sequential"
    df = pd.concat([df_mcts, df_sequential])
    fig = plt.figure(figsize=figsize)

    lines = sns.lineplot(
        data=df,
        x="step",
        # y="best_Q",
        y="true_Q",
        hue="method",
        errorbar=("ci", 95),
        lw=1,
    )

    plt.xlabel("Step")
    plt.ylabel(r"$\max \hat{Q}$")
    # plt.ylim(0.5, None)

    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_best_oscillators.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)
    plt.close()


if __name__ == "__main__":
    mcts_csv = Path(
        # "data/oscillation/mcts/bootstrap_long_230928_000832/230929_best_oscillators_3tf_long.csv"
        # "data/oscillation/mcts/bootstrap_short_230928_000000/231001_best_oscillators_3tf_short.csv"
        "data/oscillation/mcts/bootstrap_short_231001_145157/231001_best_oscillators_3tf_short.csv"
    )
    sequential_csv = Path(
        # "data/oscillation/mcts/sequential_long_230929_170434/230929_best_oscillators_3tf_long.csv"
        # "data/oscillation/mcts/sequential_short_231001_143758/231001_best_oscillators_3tf_short.csv"
        "data/oscillation/mcts/sequential_short_231001_144834/231001_best_oscillators_3tf_short.csv"
    )
    main(
        mcts_csv=mcts_csv,
        sequential_csv=sequential_csv,
        save=True,
        save_dir=Path("figures/oscillation"),
        fmt="png",
    )
