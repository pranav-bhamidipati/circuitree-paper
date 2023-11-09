from functools import partial
from pathlib import Path
import seaborn as sns
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
    figsize: tuple = (7.5, 7.5),
    ACF_cutoff: float = -0.4,
    Q_thresh: float = 0.01,
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
    bins: int = 40,
    palette: str = "YlOrBr",
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

    # Plot initial conditions corner plot
    init_conds = ["A_0", "B_0", "C_0"]
    init_cond_labels = dict(zip(init_conds, [r"$A_0$", r"$B_0$", r"$C_0$"]))
    for c in init_conds:
        data[c] = data[c].astype(int)

    cmap = sns.color_palette(palette, as_cmap=True)
    cmap.set_under("white")

    print("Plotting corner plot of initial conditions...")
    fig = plt.figure(figsize=(figsize[0] / 2, figsize[1] / 2))
    g = sns.pairplot(
        data,
        vars=init_conds,
        hue="oscillated",
        corner=True,
        kind="hist",
        diag_kind="hist",
        diag_kws=dict(
            discrete=True,
            element="step",
            stat="density",
            fill=False,
            bins=30,
            common_norm=False,
        ),
        plot_kws=dict(
            discrete=True, stat="count", cmap=cmap, vmin=1, bins=100, common_norm=False
        ),
    )

    # Set axis-level options
    for ax in g.axes.flatten():
        if ax is None:
            continue
        xlab = ax.get_xlabel()
        if len(xlab) == 0:
            ax.set_xlabel(None)
        else:
            xlab_formatted = init_cond_labels[xlab]
            ax.set_xlabel(xlab_formatted)

        ylab = ax.get_ylabel()
        if len(ylab) == 0:
            ax.set_xlabel(None)
        else:
            ylab_formatted = init_cond_labels[ylab]
            ax.set_ylabel(ylab_formatted)

    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(
            f"{today}_oscillation_init_conditions_corner_plot.{fmt}"
        )
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)

    # Select only data where state has P(oscillated) > Q_thresh
    Qs = data.groupby("state")["oscillated"].mean()
    data = data.loc[data["state"].isin(Qs[Qs > Q_thresh].index)]
    data = data.loc[data["oscillated"]]

    # Plot corner plot
    var_label_and_range = dict(
        zip(SAMPLED_VAR_NAMES, zip(SAMPLED_VAR_MATHTEXT, SAMPLING_RANGES))
    )
    print("Plotting corner plot of sampled parameters...")
    # Set axis label size
    fig = plt.figure(figsize=figsize)
    g = sns.pairplot(
        data,
        vars=SAMPLED_VAR_NAMES,
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
        fpath = Path(save_dir).joinpath(f"{today}_oscillation_corner_plot.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    ttable_parquet = Path("data/oscillation/230717_transposition_table_hpc.parquet")
    save_dir = Path("figures/oscillation/corner_plots")
    save_dir.mkdir(exist_ok=True)
    main(
        transposition_table_parquet=ttable_parquet,
        bins=30,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
