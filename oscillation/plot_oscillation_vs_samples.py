from circuitree.viz import plot_network
from datetime import datetime
from typing import Any, Iterable, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import warnings

from oscillation import OscillationGrammar

_network_kwargs = dict(
    padding=0.5,
    lw=1,
    node_shrink=0.7,
    offset=0.8,
    auto_shrink=0.9,
    width=0.005,
    plot_labels=False,
)


def main(
    mcts_csv: Path,
    sequential_csv: Path,
    figsize: tuple = (3, 3),
    max_step: int = 100_000,
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
    errorbar: Any = ("ci", 99),
    plot_network_at_steps: Iterable[int] = None,
    network_kwargs: Optional[dict] = None,
    network_replicate: int = 2,
    **kwargs,
):
    df_mcts = pd.read_csv(mcts_csv)
    df_mcts = df_mcts.loc[df_mcts["step"] <= max_step]
    df_mcts["method"] = "MCTS"
    df_sequential = pd.read_csv(sequential_csv)
    df_sequential = df_sequential.loc[df_sequential["step"] <= max_step]
    df_sequential["method"] = "Sequential"
    df = pd.concat([df_mcts, df_sequential])
    df["log_reward"] = np.log10(df["reward"])
    df["log_cum_reward"] = np.log10(df["cum_reward"])

    df = df.replace([-np.inf], 1e-4).reset_index()

    # Filter warnings from Seaborn
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    # Plot the highest reward among all states over time
    fig = plt.figure(figsize=figsize)
    lines = sns.lineplot(
        data=df,
        x="step",
        # y="best_Q",
        # y="true_Q",
        y="log_reward",
        hue="method",
        errorbar=errorbar,
        lw=1,
    )

    plt.xlabel("Samples")
    plt.xlim(500, None)
    plt.xscale("log")
    # plt.ylabel(r"$\max_i \; \hat{r}_i$")
    # plt.ylim(0.1, None)
    # plt.yscale("log")
    # plt.ylabel(r"Log$_{10}\left(\max_i \;\; \mathrm{r}_i\right)$")
    plt.ylabel(r"Best cumul. reward")
    plt.yticks(
        ticks=[0, 1, 2, 3, 4],
        labels=[r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"],
    )

    sns.despine()
    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_best_reward.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)
    plt.close()

    # Plot the total cumulative reward over time
    fig = plt.figure(figsize=figsize)
    lines = sns.lineplot(
        data=df,
        x="step",
        y="log_cum_reward",
        hue="method",
        errorbar=errorbar,
        lw=1,
    )

    plt.xlabel("Samples")
    plt.xlim(500, None)
    plt.xscale("log")
    # plt.ylabel("Cumulative reward")
    # plt.ylim(0.1, None)
    # plt.yscale("log")
    plt.ylabel(r"Log$_{10}$ Cumul. reward")

    sns.despine()
    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_cumulative_reward.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)
    plt.close()

    grammar = OscillationGrammar(
        components=["A", "B", "C"], interactions=["activates", "inhibits"]
    )
    network_kwargs = _network_kwargs | (network_kwargs or dict())
    plot_network_at_steps = plot_network_at_steps or []
    qhat_equals = r"$\hat{Q}=$"
    for step in plot_network_at_steps:
        best_row = df.loc[(df["step"] == step) & (df["replicate"] == network_replicate)]
        state = best_row["best_oscillator"].values[0]
        Qhat = best_row["reward"].values[0] / best_row["visits"].values[0]
        fig = plt.figure(figsize=figsize)
        plt.title(f"{step} samples, {qhat_equals}{Qhat:.3f}", size=10)
        plot_network(*grammar.parse_genotype(state), **network_kwargs)
        plt.xlim(-1.7, 1.7)
        plt.ylim(-1.2, 2.0)

        if save:
            today = datetime.today().strftime("%y%m%d")
            fpath = Path(save_dir).joinpath(f"{today}_best_network_step{step}.{fmt}")
            print(f"Writing to: {fpath.resolve().absolute()}")
            plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    mcts_csv = Path(
        "data/oscillation/mcts/mcts_bootstrap_short_231002_221756"
        "/231003_oscillation_data__3tf_short.csv"
    )
    sequential_csv = Path(
        "data/oscillation/mcts/sequential_bootstrap_short_231003_105603"
        "/231003_oscillation_data__3tf_short.csv"
    )

    main(
        mcts_csv=mcts_csv,
        sequential_csv=sequential_csv,
        save=True,
        save_dir=Path("figures/oscillation"),
        # fmt="eps",
        fmt="pdf",
        plot_network_at_steps=[1_000, 10_000, 50_000, 100_000],
        network_kwargs=dict(),
        network_replicate=2,
    )
