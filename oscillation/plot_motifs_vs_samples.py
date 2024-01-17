from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import warnings


def main(
    data_csv: Path,
    bins=100,
    log_scale: bool = False,
    xlim: tuple = None,
    palette: str = "muted",
    legend_title: str = "Motif combination",
    figsize: tuple = (5, 3),
    save: bool = False,
    save_dir: Path = None,
    suffix: str = "",
    fmt: str = "png",
    dpi: int = 300,
):
    data = pd.read_csv(data_csv)
    data["step"] = data["step"].astype(int)

    kw = {}
    if "motifs" not in data.columns:
        data = pd.melt(
            data,
            id_vars=["step", "replicate"],
            var_name="motifs",
            value_name="n_samples",
        )
        kw["weights"] = "n_samples"

    if log_scale:
        data = data.loc[data["step"] > 0]

    # Filter warnings from Seaborn
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    fig = plt.figure(figsize=figsize)
    p = sns.histplot(
        data=data,
        x="step",
        hue="motifs",
        bins=bins,
        palette=palette,
        multiple="fill",
        log_scale=log_scale,
        element="step",
        ec="none",
        # ec="gray",
        # lw=0.25,
        **kw,
    )
    plt.xlim(xlim)

    sns.move_legend(p, bbox_to_anchor=(1.05, 1), loc="upper left")
    p.legend_.set_title(legend_title)

    plt.xlabel("Samples")
    plt.ylabel("Proportion")

    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_sampling_proportions{suffix}.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    data_csv = Path(
        "data/oscillation/mcts/mcts_bootstrap_short_exploration2.00_231103_140501"
        "/231103_motif_counts.csv"
    )
    suffix = "_100k_iters"

    # data_csv = Path(
    #     "data/oscillation/mcts/mcts_bootstrap_1mil_iters_exploration2.00_231204_142301"
    #     "/231207_motif_counts.csv"
    # )
    # suffix = "_1mil_iters"

    save_dir = Path("figures/oscillation/")

    main(
        data_csv=data_csv,
        # log_scale=True,
        palette="deep",
        legend_title="Motifs",
        xlim=(10, None),
        suffix=suffix,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
