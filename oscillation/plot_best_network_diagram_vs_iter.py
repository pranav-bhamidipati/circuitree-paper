from circuitree.viz import plot_network
from datetime import datetime
from typing import Iterable, Optional
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from circuitree.models import SimpleNetworkGrammar

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
    bootstrap_results_csv: Path,
    plot_network_at_steps: Iterable[int],
    which_replicate: int = 2,
    network_kwargs: Optional[dict] = None,
    figsize: tuple = (3, 3),
    xlim: tuple[float] = (-1.7, 1.7),
    ylim: tuple[float] = (-1.2, 2.0),
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
    **kwargs,
):
    df = pd.read_csv(bootstrap_results_csv)
    df = df.loc[
        df["step"].isin(plot_network_at_steps) & (df["replicate"] == which_replicate)
    ]
    reward_col = "highest_reward" if "highest_reward" in df.columns else "reward"

    # Plot the best network at each step
    network_kwargs = _network_kwargs | (network_kwargs or dict())
    qhat_equals = r"$\hat{Q}=$"
    for step in plot_network_at_steps:
        step_row = df.loc[df["step"] == step]
        state = step_row["best_oscillator"].values[0]
        reward = int(step_row[reward_col].values[0])
        visits = int(step_row["visits"].values[0])
        qhat = step_row["best_Q"].values[0]
        fig = plt.figure(figsize=figsize)
        plt.title(f"{qhat_equals}{qhat:.2f} ({reward}/{visits})", size=10)
        plot_network(*SimpleNetworkGrammar.parse_genotype(state), **network_kwargs)
        plt.xlim(xlim)
        plt.ylim(ylim)

        if save:
            today = datetime.today().strftime("%y%m%d")
            fpath = Path(save_dir).joinpath(f"{today}_best_network_step{step}.{fmt}")
            print(f"Writing to: {fpath.resolve().absolute()}")
            plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    bootstrap_results_csv = Path(
        "data/oscillation/mcts/mcts_bootstrap_short_exploration2.00_231103_140501"
        "/231204_oscillation_data_3tf_100k_iters.csv"
    )

    today = datetime.today().strftime("%y%m%d")
    save_dir = Path(f"figures/oscillation/{today}_best_oscillator_vs_iter")
    save_dir.mkdir(exist_ok=True)

    main(
        bootstrap_results_csv=bootstrap_results_csv,
        plot_network_at_steps=[1_000, 10_000, 50_000, 100_000],
        save=True,
        save_dir=save_dir,
        fmt="pdf",
        which_replicate=0,
    )
