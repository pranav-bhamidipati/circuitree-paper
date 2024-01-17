from datetime import datetime
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import warnings

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

from graph_utils import simplenetwork_n_interactions


def main(
    data_dir: Path,
    params_csv: Path,
    Q_threshold: float = 0.01,
    min_visits: int = 100,
    highlight_states: dict[str, str] = None,
    highlight_bias: list[Literal["left", "right"]] = None,
    figsize: tuple[float, float] = (3.5, 2.6),
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
):
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Read in transposition table (outcomes of all simulations)
    print("Reading in transposition table...")
    ttable_parquet = sorted(data_dir.glob("*.parquet"))[-1]
    params_df = pd.read_csv(params_csv)
    ttable_df = pd.read_parquet(ttable_parquet)

    if "visit" in ttable_df.columns:
        ttable_df = ttable_df.rename(columns={"visit": "param_index"})

    # Merge information about whether or not there was a mutated component
    ttable_df = pd.merge(
        ttable_df,
        params_df.loc[:, ["param_index", "component_mutation"]],
        on="param_index",
    )
    ttable_df["has_mutation"] = ttable_df["component_mutation"] != "(none)"

    # Find which samples oscillated and whether or not they had a mutation
    ttable_df["oscillated"] = ttable_df["autocorr_min"] < -0.4
    ttable_df["oscillated_with_mutation"] = (
        ttable_df["oscillated"] & ttable_df["has_mutation"]
    )

    # Aggregatee results for each state
    agg_df = (
        ttable_df.groupby("state")
        .agg(
            visits=pd.NamedAgg("state", "count"),
            n_osc=pd.NamedAgg("oscillated", "sum"),
            n_mut=pd.NamedAgg("has_mutation", "sum"),
            n_osc_with_mut=pd.NamedAgg("oscillated_with_mutation", "sum"),
        )
        .reset_index()
    )

    # Compute complexity of each state
    agg_df["complexity"] = agg_df["state"].map(simplenetwork_n_interactions)

    # Compute robustness
    agg_df["Q_hat"] = agg_df["n_osc"] / agg_df["visits"]
    agg_df["Q_hat_std_err"] = np.sqrt(
        agg_df["Q_hat"] * (1 - agg_df["Q_hat"]) / agg_df["visits"]
    )

    # Isolate putative oscillators with at least a certain number of visits
    agg_df = agg_df.loc[
        (agg_df["Q_hat"] > Q_threshold) & (agg_df["visits"] > min_visits)
    ].reset_index(drop=True)
    agg_df["state"] = agg_df["state"].cat.remove_unused_categories()
    print(f"Plotting data for {len(agg_df)} oscillators with >{min_visits} visits")

    # Compute the robustness for the subset of samples with/without mutations
    agg_df["Q_with_mutation"] = agg_df["n_osc_with_mut"] / agg_df["n_mut"]
    agg_df["Q_with_mut_std_err"] = np.sqrt(
        agg_df["Q_with_mutation"] * (1 - agg_df["Q_with_mutation"]) / agg_df["n_mut"]
    )
    agg_df["Q_wout_mutation"] = (agg_df["n_osc"] - agg_df["n_osc_with_mut"]) / (
        agg_df["visits"] - agg_df["n_mut"]
    )
    agg_df["Q_wout_mut_std_err"] = np.sqrt(
        agg_df["Q_wout_mutation"]
        * (1 - agg_df["Q_wout_mutation"])
        / (agg_df["visits"] - agg_df["n_mut"])
    )

    # Plot
    sns.set_context("paper")

    fig, ax = plt.subplots(figsize=figsize)

    # Scatterplot
    sns.scatterplot(
        data=agg_df,
        x="Q_wout_mutation",
        y="Q_with_mutation",
        hue="Q_hat",
        size="visits",
        size_norm=(min_visits, agg_df["visits"].max()),
        sizes=(1, 100),
        # palette="viridis",
        # palette=sns.color_palette("rocket", as_cmap=True),
        # New palette, viridis and rocket are too light at the high end
        palette=sns.color_palette("flare", as_cmap=True),
        hue_norm=(0, agg_df["Q_hat"].max()),
        # s=5,
        edgecolor="lightgray",
        # edgecolor="none",
        alpha=0.6,
        linewidth=0.25,
        ax=ax,
    )
    ax.set_aspect("equal")
    sns.despine()

    err_kw = dict(
        x=agg_df["Q_wout_mutation"],
        y=agg_df["Q_with_mutation"],
        fmt="none",
        ecolor="lightgray",
        elinewidth=0.25,
        capsize=1,
        capthick=0.25,
        zorder=-1,
    )

    plt.xlabel(r"$\hat{Q}$, no mutation")
    ax.errorbar(xerr=agg_df["Q_wout_mut_std_err"], **err_kw)

    plt.ylabel(r"$\hat{Q}$, with mutation")
    ax.errorbar(yerr=agg_df["Q_with_mut_std_err"], **err_kw)

    # Plot lines for an oscillator robust to X/5 possible deletions
    for n_ft in (2, 3, 4):
        ax.plot(
            [0.1, 1.0],
            [n_ft * 0.02, n_ft * 0.2],
            color="black",
            linewidth=0.5,
            zorder=-2,
            linestyle="--",
        )

    # Plot colormap in  the top-left of the plot axis
    norm = Normalize(vmin=0.0, vmax=agg_df["Q_hat"].max())
    sm = ScalarMappable(norm=norm, cmap="flare")
    sm.set_array([])

    # Remove the default legend and create a new one
    ax.get_legend().remove()
    cax = fig.add_axes([0.2, 0.75, 0.2, 0.03])
    cbar = fig.colorbar(
        sm, pad=0.05, orientation="horizontal", cax=cax, ticks=[0.0, 0.2, 0.4, 0.6]
    )
    # Place label above colorbar
    cax.xaxis.set_label_position("top")
    cax.set_xlabel(r"$\hat{Q}$")

    if highlight_states:
        for (label, state), bias in zip(highlight_states.items(), highlight_bias):
            x = agg_df.loc[agg_df["state"] == state, "Q_wout_mutation"].values[0]
            y = agg_df.loc[agg_df["state"] == state, "Q_with_mutation"].values[0]
            print(f"Highlighting {label} state: {state} ({x}, {y})")

            if bias == "left":
                xytext = (x - 0.15, y + 0.15)
            elif bias == "right":
                xytext = (x + 0.15, y - 0.15)
            else:
                raise ValueError(f"Invalid value for entry of hightlight_bias: {bias}")

            # Draw a thin black arrow pointing to the state and label it
            ax.annotate(
                label,
                xy=(x, y),
                xytext=xytext,
                zorder=3,
                # thin black arrow
                color="black",
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    shrinkA=0,
                    shrinkB=0,
                    linewidth=0.5,
                ),
            )

    # # Plot markers for a range of Q_tilde values
    # plt.legend()
    # marker_visits = np.array([min_visits, agg_df["visits"].max()])
    # marker_sizes = (1, 100)
    # x_markers = x_min + dx * (4 + np.arange(marker_qtildes.size))
    # y_marker = level_yvals[0] + dy / 2
    # plt.text(
    #     np.median(x_markers),
    #     y_marker + dy / 2,
    #     r"$Q_\mathrm{motif}$",
    #     ha="center",
    #     va="center",
    # )
    # marker_pos = {i: (x, y_marker) for i, x in enumerate(x_markers)}
    # G_marker = nx.Graph()
    # G_marker.add_nodes_from(marker_pos.keys())

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir) / f"{today}_FTO_robustness_wwo_deletion.{fmt}"
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")

    # Plot a box plot of Q_hat grouped by complexity
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 0.75))
    sns.boxplot(
        data=agg_df,
        x="complexity",
        y="Q_hat",
        orient="h",
        fliersize=2,
        ax=ax,
    )
    sns.despine()

    plt.xlabel("Complexity")
    plt.ylabel(r"$\hat{Q}$")

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir) / f"{today}_robustness_v_interactions_boxplot.{fmt}"
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    data_dir = Path(
        "data/aws_exhaustion_exploration2.00/231118-02-23-28_5tf"
        "_exhaustion100_mutationrate0.5_batchsize1"
        "_max_interactions15_exploration2.000"
        "/backups"
    )
    params_csv = Path("data/oscillation/231104_param_sets_10000_5tf_pmutation0.5.csv")

    save_dir = Path("figures/oscillation/231121_oscillator_quality")
    save_dir.mkdir(exist_ok=True)

    highlight_states = {
        "i": "*ABCDE::AAa_ABa_ACi_BAi_BBa_CBi_CCa",
        "ii": "*ABCDE::AAa_ABa_ACi_BAi_BBa_CBi_CCa_CDi_CEi_DAi_DDa_EAi",
        "iii": "*ABCDE::AAa_ABa_ACi_BAi_BBa_BDa_CBi_CCa_CDi_CEi_DAi_DBi_DCa_EAi_EEa",
    }
    highlight_bias = ["right", "left", "left"]

    main(
        data_dir=data_dir,
        params_csv=params_csv,
        highlight_states=highlight_states,
        highlight_bias=highlight_bias,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
