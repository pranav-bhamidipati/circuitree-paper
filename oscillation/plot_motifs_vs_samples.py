from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import warnings


def main(
    data_csv: Path,
    save: bool = False,
    save_dir: Path = None,
    figsize: tuple = (5, 3),
    fmt: str = "png",
    dpi: int = 300,
    palette: str = "muted",
    multiple: str = "fill",
    legend_title: str = "Motif combination",
    bins=100,
):
    data = pd.read_csv(data_csv)
    data["step"] = data["step"].astype(int)
    data = pd.melt(
        data, id_vars=["step", "replicate"], var_name="motifs", value_name="n_samples"
    )

    # Filter warnings from Seaborn
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    fig = plt.figure(figsize=figsize)
    p = sns.histplot(
        data=data,
        x="step",
        hue="motifs",
        weights="n_samples",
        bins=bins,
        palette=palette,
        multiple=multiple,
        ec=None,
    )

    sns.move_legend(p, bbox_to_anchor=(1.05, 1), loc="upper left")
    p.legend_.set_title(legend_title)

    plt.xlabel("Samples")
    plt.ylabel("Proportion")

    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_sampling_proportions.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    bins = 100
    # data_csv = Path(
    #     "data/oscillation/mcts/mcts_bootstrap_short_231002_221756"
    #     "/231003_motif_counts.csv"
    # )
    # data_csv = Path(
    #     "data/oscillation/mcts/mcts_bootstrap_short_231020_175449"
    #     "/231102_motif_counts.csv"
    # )
    # data_csv = Path(
    #     "data/oscillation/mcts/mcts_bootstrap_short_231102_111706"
    #     "/231102_motif_counts.csv"
    # )
    data_csv = Path(
        "data/oscillation/mcts/mcts_bootstrap_short_exploration2.00_231103_140501"
        "/231103_motif_counts.csv"
    )

    # bins = 50
    # data_csv = Path(
    #     "data/oscillation/mcts/mcts_bootstrap_long_231022_173227"
    #     "/231102_motif_counts.csv"
    # )

    # names = ["(none)", "AI", "Rep", "AI+Rep"]

    save_dir = Path("figures/oscillation/")

    main(
        data_csv=data_csv,
        save=True,
        save_dir=save_dir,
        # fmt="pdf",
        bins=bins,
        palette="deep",
        legend_title="Motifs",
    )
