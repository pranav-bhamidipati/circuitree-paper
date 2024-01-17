from circuitree import SimpleNetworkGrammar
from functools import partial
from pathlib import Path
import seaborn as sns
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from functools import cache

from gillespie import (
    PARAM_NAMES,
    SAMPLED_VAR_NAMES,
    SAMPLING_RANGES,
    SAMPLED_VAR_MATHTEXT,
    convert_params_to_sampled_quantities,
)

convert_to_sampled_params = partial(
    convert_params_to_sampled_quantities, param_ranges=SAMPLING_RANGES
)


def main(
    transposition_table_parquet: Path,
    vars: list[str] = SAMPLED_VAR_NAMES,
    figsize: tuple = (7.5, 7.5),
    ACF_cutoff: float = -0.4,
    Q_thresh: float = 0.01,
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
    bins: int = 40,
    # multiple: str = "fill",
):
    # Filter deprecation warnings from Seaborn and Pandas
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    sns.set_context("paper", font_scale=2.5)

    # Load the transposition table
    print("Loading transposition table...")
    data = pd.read_parquet(transposition_table_parquet)

    # Convert parameters to the sampled quantities used to derive them
    sampled_quantities = convert_to_sampled_params(data[PARAM_NAMES].values.T)
    for i, name in enumerate(SAMPLED_VAR_NAMES):
        data[name] = sampled_quantities[i]

    # Get which samples oscillated
    data["oscillated"] = data["reward"] < ACF_cutoff

    grammar = SimpleNetworkGrammar(
        components=["A", "B", "C"], interactions=["activates", "inhibits"]
    )

    @cache
    def has_pattern(state, pattern):
        return grammar.has_pattern(state, pattern)

    data["AI"] = [has_pattern(state, "ABa_BAi") for state in data["state"]]
    data["III"] = [has_pattern(state, "ABi_BCi_CAi") for state in data["state"]]
    data["toggle"] = [has_pattern(state, "ABi_BAi") for state in data["state"]]
    data["toggle_only"] = data["toggle"] & ~data["AI"] & ~data["III"]

    # Select only data where state has P(oscillated) > Q_thresh
    Qs = data.groupby("state")["oscillated"].mean()
    data = data.loc[data["state"].isin(Qs[Qs > Q_thresh].index)]
    data = data.loc[data["oscillated"]]

    # Select samples of toggle-only topologies
    data = data.loc[data["toggle_only"]]
    print(data)

    # Plot corner plot
    var_label_and_range = dict(
        zip(SAMPLED_VAR_NAMES, zip(SAMPLED_VAR_MATHTEXT, SAMPLING_RANGES))
    )
    print("Plotting corner plot of sampled parameters...")
    # Set axis label size
    fig = plt.figure(figsize=figsize)
    g = sns.pairplot(
        data,
        vars=vars,
        corner=True,
        kind="hist",
        diag_kind="hist",
        diag_kws=dict(bins=bins, edgecolor="none", linewidth=0.0),
        plot_kws=dict(bins=bins),
    )

    # Set axis-level options
    for ax in g.axes.flatten():
        if ax is None:
            continue
        xlab = ax.get_xlabel()
        if len(xlab) == 0:
            ax.set_xlabel(None)
        else:
            xlab_formatted, xlim = var_label_and_range[xlab]
            ax.set_xlabel(xlab_formatted)
            ax.set_xlim(xlim)

        ylab = ax.get_ylabel()
        if len(ylab) == 0:
            ax.set_xlabel(None)
        else:
            ylab_formatted, ylim = var_label_and_range[ylab]
            ax.set_ylabel(ylab_formatted)
            ax.set_ylim(ylim)

    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(
            f"{today}_oscillation_toggle_only_corner_plot.{fmt}"
        )
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    ttable_parquet = Path("data/oscillation/230717_transposition_table_hpc.parquet")
    save_dir = Path("figures/oscillation/toggle_corner_plots")
    save_dir.mkdir(exist_ok=True)

    main(
        transposition_table_parquet=ttable_parquet,
        vars=["log10_kd_1", "nlog10_km_rep_unbound_ratio"],
        bins=30,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
