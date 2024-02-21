from datetime import datetime
from math import floor
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from tqdm import tqdm


def get_complexity(state: str) -> int:
    _, interactions_joined = state.split("::")
    if interactions_joined:
        return interactions_joined.count("_") + 1
    else:
        return 0


def get_Qval(state: str, graph: nx.DiGraph) -> float:
    if state in graph.nodes:
        return graph.nodes[state]["reward"] / max(1, graph.nodes[state]["visits"])
    else:
        return 0.0


def main(
    results_csv: Path,
    data_dir: Path,
    Qthresh: float = 0.01,
    step_max: Optional[int] = None,
    figsize: tuple[float, float] = (5, 4),
    save: bool = False,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
    dpi: int = 300,
):
    # Get which states are oscillators
    results_df = pd.read_csv(results_csv, index_col=0)
    results_df["complexity"] = results_df["state"].map(get_complexity)
    results_df = results_df.sort_values(
        ["complexity", "p_oscillation"], ascending=(True, False)
    )
    results_df["rank"] = results_df["p_oscillation"].rank(ascending=False).astype(int)
    states = results_df["state"].copy()
    results_df = results_df.reset_index(drop=True)
    # results_df = results_df.loc[results_df["p_oscillation"] >= Qthresh]
    oscillators = results_df.loc[results_df["p_oscillation"] >= Qthresh]["state"]

    # Load sampling data
    replicate_dirs = [d for d in data_dir.glob("*") if d.is_dir()]
    n_replicates = len(replicate_dirs)

    unique_steps = set()
    for csv in Path(data_dir).glob("*/*_oscillators.csv"):
        step = int(csv.name.split("_")[-2])
        unique_steps.add(step)
    steps = np.array(sorted(unique_steps))
    if step_max is None:
        step_max = steps[-1]
    where_step_max = np.searchsorted(steps, step_max, side="right")
    n_steps = len(steps)

    # Count the replicates that classified each state as an oscillator at each step
    state_found = np.zeros((n_replicates, n_steps, len(states)), dtype=bool)
    for i, replicate_dir in enumerate(tqdm(replicate_dirs, desc="Loading replicates")):
        for j, step in enumerate(steps):
            csv = next(replicate_dir.glob(f"*_{step}_oscillators.csv"))
            sample_df = pd.read_csv(csv)
            discovered_idx = states.isin(
                sample_df["state"].loc[sample_df["Q"] >= Qthresh]
            )
            state_found[i, j] = discovered_idx

    # Make a dataframe of the discovery rate for each true oscillator at each step
    osc_discovery_vs_t = state_found[:, :, oscillators.index].mean(axis=0).T
    osc_discovery_df = pd.DataFrame(
        osc_discovery_vs_t, index=oscillators, columns=steps
    )
    osc_discovery_df.index.name = "oscillator"
    osc_discovery_df.columns.name = "sampling_time"

    # Compute precision and recall for each replicate over time
    osc_mask = states.isin(oscillators).values
    precision_vs_t = state_found[:, :, osc_mask].sum(axis=-1) / state_found.sum(axis=-1)
    recall_vs_t = state_found[:, :, osc_mask].sum(axis=-1) / len(oscillators)

    top_pct = 10
    top_N = floor(len(oscillators) * top_pct / 100)
    top_oscillators = results_df.loc[results_df["rank"] <= top_N]["state"]
    top_mask = states.isin(top_oscillators).values
    top_recall_vs_t = state_found[:, :, top_mask].sum(axis=-1) / top_N
    top_precision_vs_t = state_found[:, :, top_mask].sum(axis=-1) / state_found.sum(
        axis=-1
    )

    # For each replicate, calculate the sensitivity and false positive rate
    osc_mask = states.isin(oscillators).values
    discoveries = state_found[:, -1]
    n_true_positives = discoveries[:, osc_mask].sum(axis=1)
    n_false_positives = discoveries[:, ~osc_mask].sum(axis=1)
    n_false_negatives = (~discoveries)[:, osc_mask].sum(axis=1)
    n_true_negatives = (~discoveries)[:, ~osc_mask].sum(axis=1)

    precision = n_true_positives / (n_true_positives + n_false_positives)
    sensitivities = n_true_positives / len(oscillators)
    false_positives = n_false_positives / (len(states) - len(oscillators))

    print("\nPrecision: \n", pd.Series(precision).describe())
    print("\nSensitivity/Recall: \n", pd.Series(sensitivities).describe())
    print("\nFalse positive rate: \n", pd.Series(false_positives).describe())

    discovery_rate = discoveries.mean(axis=0)
    discovery_df = pd.DataFrame(
        dict(
            state=results_df["state"],
            discovery_rate=discovery_rate,
            complexity=results_df["complexity"],
            Q=results_df["p_oscillation"],
        )
    )

    ...

    warnings.filterwarnings("ignore", category=FutureWarning)

    ### Plot % of replicates that classified each state as an oscillator vs. Q
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(r"P(Labeled oscillator)")
    palette = sns.color_palette(
        "turbo_r", n_colors=discovery_df["complexity"].nunique(), desat=0.8
    )
    sns.scatterplot(
        data=discovery_df,
        x="Q",
        y="discovery_rate",
        edgecolor="none",
        palette=palette,
        hue="complexity",
        legend="full",
        # alpha=0.5,
        s=10,
    )
    # Move the legend outside the plot, adding a title "Complexity"
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, title="Complexity")

    # Plot a dashed line showing the threshold for Q
    plt.axvline(Qthresh, color="gray", ls="--", lw=0.5)
    plt.text(
        Qthresh * 0.85,
        0.8,
        r"$Q_\mathrm{thresh}$",
        ha="right",
        va="center",
        color="gray",
        fontsize="small",
    )

    sns.despine()

    plt.xlabel(r"$Q$")
    plt.xscale("log")

    plt.ylabel(r"$P(\hat{Q} > 0.01)$")
    # plt.yscale("log")

    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_logP_discovery_vs_Q.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)

    ## Plot the precision and recall of oscillators discovered over time
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(steps, precision_vs_t.mean(axis=0), label="Precision", color="C0")
    ax.plot(steps, recall_vs_t.mean(axis=0), label="Recall (all)", color="C1")
    ax.plot(
        steps,
        top_recall_vs_t.mean(axis=0),
        label=f"Recall (Top {top_pct}%)",
        color="C3",
    )
    ax.fill_between(
        steps, *np.percentile(precision_vs_t, [5, 95], axis=0), alpha=0.2, color="C0"
    )
    ax.fill_between(
        steps, *np.percentile(recall_vs_t, [5, 95], axis=0), alpha=0.2, color="C1"
    )
    ax.fill_between(
        steps, *np.percentile(top_recall_vs_t, [5, 95], axis=0), alpha=0.2, color="C3"
    )

    ax.set_xlabel(r"Sampling time ($10^3$ iterations)")
    xticks = np.linspace(0, steps.max(), 11).astype(int)
    xticklabels = [f"{xt // 1000}" for xt in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Rate")

    ax.legend()
    sns.despine()
    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_precision_recall_vs_t.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)

    # Plot a kymograph of oscillators discovered over time
    #   - x-axis is sampling time, y-axis is oscillator
    #   - cmap is the discovery rate
    results_df = results_df.loc[results_df["p_oscillation"] >= Qthresh]

    fig = plt.figure(figsize=figsize)
    plt.title("Oscillators discovered during sampling")

    # palette = sns.color_palette(
    hmap = sns.heatmap(
        data=osc_discovery_vs_t[:, :where_step_max],
        vmin=0,
        vmax=1,
        cmap="rocket_r",
        cbar_kws={"label": f"P(Discovery), n={n_replicates}"},
    )

    # Plot a horizontal line between each level of complexity
    results_df["last_of_level"] = results_df["complexity"].diff().fillna(0) != 0
    where_complexity_changes = np.where(results_df["last_of_level"])[0]
    complexities = []
    y_complexities = []
    for idx in where_complexity_changes:
        hmap.axhline(idx, color="black", lw=0.5, ls="--")
        cpx = results_df.iloc[idx]["complexity"]
        y_cpx = np.where(results_df["complexity"] == cpx)[0].mean()
        complexities.append(cpx)
        y_complexities.append(y_cpx)

    # Use fewer ticks on the x-axis
    # plt.xlabel(r"Sampling time ($10^3$ iterations)")
    plt.xlabel(r"$N$ ($10^3$ iterations)")
    xtick_idx = np.linspace(0, where_step_max, 11).astype(int)[1:] - 1
    xticklabels = [f"{steps[idx] // 1000}" for idx in xtick_idx]
    hmap.set_xticks(xtick_idx + 0.5)
    hmap.set_xticklabels(xticklabels)

    # Remove y ticks and add labels for each complexity at the mean y position
    plt.ylabel("Complexity (# interactions)")
    hmap.set_yticks(y_complexities)
    hmap.set_yticklabels(complexities)

    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_oscillator_kymograph.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    # results_csv = Path("data/oscillation/230717_motifs.csv")
    results_csv = Path("data/oscillation/231102_exhaustive_results.csv")

    # data_dir = Path("data/oscillation/mcts/mcts_bootstrap_short_231020_175449")
    # step_max = 50_000

    data_dir = Path(
        "data/oscillation/mcts/mcts_bootstrap_short_exploration2.00_231103_140501"
    )
    step_max = 100_000

    save_dir = Path("figures/oscillation")

    main(
        results_csv=results_csv,
        data_dir=data_dir,
        step_max=step_max,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
