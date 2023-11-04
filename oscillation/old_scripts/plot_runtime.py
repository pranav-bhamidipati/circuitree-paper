import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns


def plot_marginals(
    df,
    quantity_names,
    figsize=(8, 6),
    gridshape=(3, 3),
    save=False,
    save_dir=None,
    fmt="png",
    dpi=300,
):
    fig = plt.figure(figsize=figsize)
    sns.despine(fig, left=True, bottom=True)
    for i, name in enumerate(quantity_names):
        ax = fig.add_subplot(*gridshape, i + 1)
        g = sns.histplot(
            data=df,
            x=name,
            y="runtime_seconds",
            stat="count",
            # kde=True,
            ax=ax,
            log_scale=(False, True),
        )
        ax.set_ylabel("Runtime (seconds)")

    plt.suptitle("Marginal distributions of runtime vs. random parameters")
    plt.tight_layout()

    if save:
        fpath = save_dir.joinpath(f"2023-06-13_runtime_marginals.{fmt}")
        fpath = fpath.resolve().absolute()
        print(f"Saving figure to: {fpath}")
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")


def plot_scatterplot_cmap(
    df,
    x,
    y,
    figsize=(4, 3),
    save=False,
    save_dir=None,
    fmt="png",
    dpi=300,
):
    fig, ax = plt.subplots(figsize=figsize)
    plt.style.use("dark_background")
    plt.rcParams.update({"grid.linewidth": 0.5, "grid.alpha": 0.5})
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="log10_runtime_seconds",
        alpha=1.0,
        s=8,
        palette="magma",
    )
    # sns.despine()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    if save:
        fpath = save_dir.joinpath(f"2023-06-13_runtime_heatmap_{x}_vs_{y}.{fmt}")
        fpath = fpath.resolve().absolute()
        print(f"Saving figure to: {fpath}")
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")


def main(
    data_path: Path,
    save=False,
    save_dir=None,
    fmt="png",
    dpi=300,
    plot_pairs=(),
    quantity_names=(),
):
    df = pd.read_csv(data_path).reset_index(drop=True)
    kw = dict(save=save, save_dir=save_dir, fmt=fmt, dpi=dpi)

    plot_marginals(df, quantity_names=quantity_names, **kw)

    df["log10_runtime_seconds"] = np.log10(df["runtime_seconds"])
    df_mean = (
        df.groupby(quantity_names)[["runtime_seconds", "log10_runtime_seconds"]]
        .agg(np.mean)
        .reset_index()
    )

    if plot_pairs:
        x, y = plot_pairs[0]
        plot_scatterplot_cmap(df_mean, x=x, y=y, save=False)

    for x, y in plot_pairs:
        plot_scatterplot_cmap(df_mean, x=x, y=y, **kw)


if __name__ == "__main__":
    # from gillespie_newranges import (
    from gillespie import (
        SAMPLED_VAR_NAMES,
    )

    data_path = Path(
        "data/oscillation/gillespie_runtime/2023-06-15_random_sample_runtimes_oldranges.csv"
        # "data/oscillation/gillespie_runtime/2023-06-15_random_sample_runtimes_newranges.csv"
        # "data/oscillation/gillespie_runtime/2023-06-13_random_sample_runtimes_AIcircuit_newranges.csv"
    )
    save_dir = Path(
        "figures/oscillation/runtime_oldranges"
        # "figures/oscillation/runtime_newranges"
    )
    save_dir.mkdir(exist_ok=True)

    pairs = [
        ("log10_k_off_1", "log10_k_on"),
        ("log10_k_on", "nlog10_gamma_m"),
        ("log10_k_off_1", "nlog10_gamma_m"),
    ]

    # pairs = [
    #     ("log10_kd_1", "kd_2_1_ratio"),
    #     ("log10_kd_1", "gamma_m"),
    # ]

    main(
        data_path=data_path,
        plot_pairs=pairs,
        quantity_names=SAMPLED_VAR_NAMES,
        save_dir=save_dir,
        save=True,
    )
