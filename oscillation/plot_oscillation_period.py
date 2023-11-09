from datetime import datetime
from functools import cache, partial
from pathlib import Path
from typing import Optional
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import warnings

from circuitree.models import SimpleNetworkGrammar

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


def _get_motif(
    state: str,
    motif: str,
    grammar: SimpleNetworkGrammar,
):
    return grammar.has_pattern(state, motif)


def main(
    frequencies_csv: Path,
    transposition_table_parquet: Path,
    motifs={
        "AI": "ABa_BAi",
        "III": "ABi_BCi_CAi",
        "Toggle": "ABi_BAi",
        # "AAI": "ABa_BCa_CAi",
    },
    figsize: tuple[float, float] = (8, 4),
    save: bool = False,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
    dpi: int = 300,
):
    grammar = SimpleNetworkGrammar(
        components=["A", "B", "C"], interactions=["activates", "inhibits"]
    )
    freqs_df = pd.read_csv(frequencies_csv)
    df = pd.read_parquet(transposition_table_parquet).reset_index(drop=True)
    df = df.iloc[freqs_df["index_in_transposition_table"].values, :]
    df["frequency_per_min"] = freqs_df["frequency_per_min"].values
    df["period_mins"] = 1 / df["frequency_per_min"]

    for motif, motif_code in motifs.items():
        get_motif = partial(_get_motif, motif=motif_code, grammar=grammar)
        df[motif] = df["state"].apply(get_motif)

    @cache
    def concat_motifs(*has_motifs):
        motif_strs = [m for m, has_motif in zip(motifs, has_motifs) if has_motif]
        if motif_strs:
            return "+".join([m for m in motif_strs])
        else:
            return "(none)"

    motif_combinations = df[list(motifs.keys())].values
    df["pattern_combination"] = pd.Categorical(
        [concat_motifs(*mc) for mc in motif_combinations]
    )

    # Convert parameters to the sampled quantities used to derive them
    sampled_quantities = convert_to_sampled_params(df[PARAM_NAMES].values.T)
    for i, name in enumerate(SAMPLED_VAR_NAMES):
        df[name] = sampled_quantities[i]

    warnings.filterwarnings(action="ignore", category=FutureWarning)

    ### Plotting

    # Plot the period of oscillation vs. the motifs present in the circuit
    df["log_period_mins"] = np.log10(df["period_mins"])
    order = (
        df.groupby("pattern_combination")
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(figsize[0] * 0.9, figsize[1] * 1.1),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    sns.violinplot(
        data=df,
        x="log_period_mins",
        y="pattern_combination",
        scale="width",
        order=order,
        ax=ax1,
    )
    plt.xlabel(r"Period (mins)")
    plt.ylabel("Pattern combination")
    sns.despine()

    # Plot a barplot to the right of the violin plot with the number of samples in each group
    pattern_sizes = df.groupby("pattern_combination").size()
    sns.barplot(x=pattern_sizes, y=pattern_sizes.index, order=order, ax=ax2)
    plt.xlabel("# Oscillating samples")
    # for i, group in enumerate(order):
    #     group_size = df[df["pattern_combination"] == group].shape[0]
    #     ax.text(
    #         1.05,
    #         i,
    #         f"{group_size:,}",
    #         ha="left",
    #         va="center",
    #         transform=ax.get_yaxis_transform(),
    #     )

    plt.tight_layout()
    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_log_period_by_motif.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")

    # Plot the frequency of oscillation vs. each parameter
    fig = plt.figure(figsize=figsize)

    # Set a viridis cmap that makes the lowest value (0) white
    cmap = sns.color_palette("viridis", as_cmap=True)
    for i, (param, label) in enumerate(zip(SAMPLED_VAR_NAMES, SAMPLED_VAR_MATHTEXT)):
        ax = fig.add_subplot(2, 4, i + 1)
        sns.histplot(
            x=df["period_mins"],
            y=df[param],
            log_scale=(True, False),
            bins=35,
            cmap=cmap,
            ax=ax,
        )
        plt.xlabel(r"Period (mins)")
        plt.xscale("log")
        xsuffix = "log_period"

        # plt.scatter(np.sqrt(df["frequency_per_min"]), df[param], s=1, alpha=0.5)
        # plt.xlabel(r"$\sqrt{\omega}$ (min$^{-1/2}$)")
        # xsuffix = "sqrt_frequency"

        plt.ylabel(label)

    plt.tight_layout()
    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_params_vs_{xsuffix}.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")

    # Plot the frequency of oscillation vs. gamma_p / gamma_m
    fig, ax = plt.subplots(figsize=(figsize[0] / 2.5, figsize[0] / 3))

    # Set a viridis cmap that makes the lowest value (0) white
    cmap = sns.color_palette("viridis", as_cmap=True)
    df["degradation_ratio"] = df["gamma_p"] / df["gamma_m"]
    sns.histplot(
        data=df,
        x="period_mins",
        y="degradation_ratio",
        log_scale=(True, True),
        bins=35,
        cmap=cmap,
    )
    plt.xlabel(r"Period (mins)")
    plt.ylabel(r"$\frac{\gamma_p}{\gamma_m}$", rotation=0, labelpad=12, fontsize=16)
    # Plot colorbar
    cbar = plt.colorbar(ScalarMappable(cmap=cmap), ax=ax)

    plt.tight_layout()
    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_degradation_ratio_vs_period.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    frequencies_csv = Path(
        "data/oscillation/bfs_230710_hpc/231011_oscillator_frequencies.csv"
    )
    ttable_parquet = Path("data/oscillation/230717_transposition_table_hpc.parquet")
    save_dir = Path("figures/oscillation")

    main(
        frequencies_csv=frequencies_csv,
        transposition_table_parquet=ttable_parquet,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
