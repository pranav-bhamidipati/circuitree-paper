from datetime import datetime
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import warnings


def main(
    mcts_csv: Path,
    long_mcts_csv: Optional[Path] = None,
    exh_results_csv: Optional[Path] = None,
    best_mean_reward: Optional[float] = None,
    figsize: tuple = (2.7, 2.7),
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
    errorbar: Any = ("ci", 99),
    **kwargs,
):
    if exh_results_csv is not None:
        Q_values = pd.read_csv(exh_results_csv)["p_oscillation"]
        best_mean_reward = Q_values.max()
        mean_overall_reward = Q_values.mean()
    elif best_mean_reward is None:
        raise ValueError("Must provide either exh_results_csv or best_mean_reward")

    df = pd.read_csv(mcts_csv)
    max_step = df["step"].max()
    df = df.loc[df["step"] > 0]
    df["regret"] = best_mean_reward * df["step"] - df["cum_reward"]

    df = df.replace([-np.inf], 1e-4).reset_index()

    # Filter warnings from Seaborn
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    # Log-log plot of regret over sampling time
    fig = plt.figure(figsize=figsize)
    lines = sns.lineplot(
        data=df,
        x="step",
        y="regret",
        hue="replicate",
        palette="muted",
        errorbar=errorbar,
        lw=0.5,
        legend=False,
    )

    # Plot a dashed line indicating the naive regret (based on overall mean reward)
    naive_slope = best_mean_reward - mean_overall_reward
    plt.plot(
        [0, max_step],
        [0, (best_mean_reward - mean_overall_reward) * max_step],
        linestyle="--",
        color="black",
        lw=1,
    )

    plt.xlabel("Samples")
    plt.xlim(0, None)
    plt.ylabel(r"Regret")
    plt.ylim(0, None)

    # Scientific notation for axes
    plt.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    # Make plot axes equal height and width
    ax = plt.gca()
    ax.set_aspect(1 / naive_slope)

    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_mcts_regret.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)
    plt.close()

    # If long_mcts_csv is provided, plot the regret for the longer run on log-log axes
    if long_mcts_csv is not None:
        long_df = pd.read_csv(long_mcts_csv)
        long_max_step = long_df["step"].max()
        long_df = long_df.loc[long_df["step"] > 0]
        long_df["regret"] = best_mean_reward * long_df["step"] - long_df["cum_reward"]
        long_df = long_df.replace([-np.inf], 1e-4).reset_index()

        fig = plt.figure(figsize=figsize)
        lines = sns.lineplot(
            data=long_df,
            x="step",
            y="regret",
            hue="replicate",
            palette="muted",
            errorbar=errorbar,
            lw=0.5,
            legend=False,
        )

        # Plot a dashed line indicating the naive regret (based on overall mean reward)
        plt.plot(
            [10_000, long_max_step],
            [10_000 * naive_slope, long_max_step * naive_slope],
            linestyle="--",
            color="black",
            lw=1,
        )

        plt.xlabel("Samples")
        plt.xlim(10_000, None)
        plt.ylabel(r"Regret")
        plt.ylim(naive_slope * 10_000, None)

        plt.xscale("log")
        plt.yscale("log")

        # Make plot axes equal height and width
        ax = plt.gca()
        ax.set_aspect(1)

        plt.tight_layout()

        if save:
            today = datetime.today().strftime("%y%m%d")
            fpath = Path(save_dir).joinpath(f"{today}_mcts_regret_loglog.{fmt}")
            print(f"Writing to: {fpath.resolve().absolute()}")
            plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    mcts_csv = Path(
        "data/oscillation/mcts/mcts_bootstrap_short_exploration2.00_231103_140501"
        "/231204_oscillation_data_3tf_100k_iters.csv"
    )
    long_mcts_csv = Path(
        "data/oscillation/mcts/mcts_bootstrap_1mil_iters_exploration2.00_231204_142301"
        "/231206_oscillation_data_3tf_1mil_iters.csv"
    )

    exh_results_csv = Path("data/oscillation/231102_exhaustive_results.csv")

    save_dir = Path("figures/oscillation")

    main(
        mcts_csv=mcts_csv,
        long_mcts_csv=long_mcts_csv,
        exh_results_csv=exh_results_csv,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
