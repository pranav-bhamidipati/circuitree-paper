from datetime import datetime
from functools import cache, partial
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

from oscillation import OscillationGrammar

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


def _get_n_interactions(
    state: str,
    grammar: OscillationGrammar,
):
    components, activations, inhibitions = grammar.parse_genotype(state)
    return len(activations) + len(inhibitions)


def _get_motif(
    state: str,
    motif: str,
    grammar: OscillationGrammar,
):
    return grammar.has_motif(state, motif)


def main(
    frequencies_csv: Path,
    transposition_table_parquet: Path,
    motifs={"AI": "ABa_BAi", "AAI": "ABa_BCa_CAi", "III": "ABi_BCi_CAi"},
    figsize: tuple[float, float] = (8, 4),
    save: bool = False,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
    dpi: int = 300,
):
    grammar = OscillationGrammar(
        components=["A", "B", "C"], interactions=["activates", "inhibits"]
    )
    get_n_interactions = partial(_get_n_interactions, grammar=grammar)
    freqs_df = pd.read_csv(frequencies_csv)
    df = pd.read_parquet(transposition_table_parquet).reset_index(drop=True)
    del df["reward"]
    df = df.iloc[freqs_df["index_in_transposition_table"].values, :]
    df["frequency_per_min"] = freqs_df["frequency_per_min"].values
    df["ACF_min"] = freqs_df["ACF_min"].values
    df["period_mins"] = 1 / df["frequency_per_min"]

    df["n_interactions"] = pd.Categorical(
        df["state"].apply(get_n_interactions), ordered=True, categories=range(0, 10)
    )

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
    df["NFR_combination"] = pd.Categorical(
        [concat_motifs(*mc) for mc in motif_combinations]
    )

    # Convert parameters to the sampled quantities used to derive them
    sampled_quantities = convert_to_sampled_params(df[PARAM_NAMES].values.T)
    for i, name in enumerate(SAMPLED_VAR_NAMES):
        df[name] = sampled_quantities[i]

    warnings.filterwarnings(action="ignore", category=FutureWarning)

    # Plot the frequency of oscillation vs. number of interactions
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    sns.histplot(
        data=df,
        x="period_mins",
        y="NFR_combination",
        stat="proportion",
        # y=df["n_interactions"],
        bins=35,
        log_scale=(True, False),
        # multiple="stack",
        ax=ax,
    )

    plt.tight_layout()
    if save:
        today = datetime.today().strftime("%Y-%m-%d")
        fpath = Path(save_dir).joinpath(f"{today}_log_period_by_NFR.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")

    # Plot the frequency of oscillation vs. each parameter
    fig = plt.figure(figsize=figsize)

    # Set a viridis cmap that makes the lowest value (0) white
    cmap = sns.color_palette("viridis", as_cmap=True)
    for i, (param, label) in enumerate(zip(SAMPLED_VAR_NAMES, SAMPLED_VAR_MATHTEXT)):
        ax = fig.add_subplot(2, 4, i + 1)

        # plt.scatter(df["period_mins"], df[param], s=1, alpha=0.5)
        # plt.hist2d(df["period_mins"], df[param])
        # sns.histplot(
        #     x=df["period_mins"],
        #     y=df[param],
        #     bins=20,
        #     cmap=cmap,
        #     cbar=True,
        #     vmin=1,
        #     cbar_kws={"label": "Number of samples"},
        #     ax=ax,
        # )
        # plt.xlabel("Period (min)")
        # xsuffix = "period"

        # plt.scatter(df["frequency_per_min"], df[param], s=1, alpha=0.5)
        # plt.hist2d(df["frequency_per_min"], df[param], cmap=cmap, bins=50, cmin=1)
        # sns.kdeplot(
        #     x=df["period_mins"],
        #     y=df[param],
        #     cmap=cmap,
        #     shade=True,
        #     ax=ax,
        # )
        sns.histplot(
            x=df["period_mins"],
            y=df[param],
            # size=1,
            # alpha=0.5,
            log_scale=(True, False),
            bins=35,
            cmap=cmap,
            # cbar=True,
            # vmin=-0.0,
            # cbar_kws={"label": "Number of samples"},
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
        today = datetime.today().strftime("%Y-%m-%d")
        fpath = Path(save_dir).joinpath(f"{today}_params_vs_{xsuffix}.{fmt}")
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
