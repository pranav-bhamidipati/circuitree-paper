from collections import Counter
from datetime import datetime
from itertools import combinations
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import spatial
import scipy.cluster.hierarchy as sch
from scipy.stats import mannwhitneyu
import warnings

from graph_utils import simplenetwork_n_interactions


def main(
    pdist_hdf: Path,
    n_clusts: int = 2,
    figsize: tuple = (6, 6),
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
    suffix: str = "",
    boxplot_figsize: tuple = (2, 3),
):
    # Filter warnings from Seaborn
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    print("Loading distance matrix and metadata...")
    with h5py.File(pdist_hdf, "r") as f:
        dist_matrix = f["distance_matrix"][:]

    agg_data = pd.read_hdf(pdist_hdf, key="metadata")

    print(f"Distance matrix shape: {dist_matrix.shape}")

    print("Making linkage matrix...")
    linkage = sch.linkage(spatial.distance.squareform(dist_matrix), method="average")

    print("Plotting heatmap of pairwise circuit distance...")
    sns.clustermap(
        dist_matrix,
        row_linkage=linkage,
        col_linkage=linkage,
        xticklabels=False,
        yticklabels=False,
        figsize=figsize,
    )

    if save:
        today = datetime.today().strftime("%y%m%d")
        fname = Path(save_dir).joinpath(f"{today}_pairwise_dist{suffix}.png")
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(fname, dpi=dpi, bbox_inches="tight", transparent=True)

    if "complexity" not in agg_data.columns:
        agg_data["complexity"] = agg_data["state"].map(simplenetwork_n_interactions)

    # Get complexity vs cluster identity
    clusters = sch.fcluster(linkage, n_clusts, criterion="maxclust")
    agg_data["cluster"] = clusters

    ## Compute Mann-Whitney U test for complexity vs cluster.
    ## This also determines the y-value for reporting the test results for each cluster pair.
    bottom_yval = agg_data["complexity"].max() + 2
    dy = 2.0
    cluster_pairs = list(combinations(range(1, n_clusts + 1), 2))
    n_tests = len(cluster_pairs)
    levels_above_cluster = Counter()
    clusters_yvals_and_pvals = {}
    next_pair = 0
    while cluster_pairs:
        i, j = cluster_pairs.pop(next_pair)

        # Perform test
        mw = mannwhitneyu(
            agg_data.loc[agg_data["cluster"] == i, "complexity"].values,
            agg_data.loc[agg_data["cluster"] == j, "complexity"].values,
        )
        pval = mw.pvalue
        stat = mw.statistic

        # Determine y-value for reporting test results
        y_level = max([levels_above_cluster[i], levels_above_cluster[j]])
        yval = bottom_yval + dy * y_level
        clusters_yvals_and_pvals[(i, j)] = yval, pval, stat

        # Pick the next pair to be the one with the lowest y-value
        for k in range(i, j + 1):
            levels_above_cluster[k] += 1
        next_pair = min(
            range(len(cluster_pairs)),
            key=lambda i: max(levels_above_cluster[c] for c in cluster_pairs[i]),
            default=-1,
        )

    # Plot boxplot of complexity vs cluster
    fig = plt.figure(figsize=boxplot_figsize)
    sns.boxplot(data=agg_data, x="cluster", y="complexity")
    plt.xlabel("Cluster")
    plt.ylabel("Complexity")

    # Print the Mann-Whitney U test results and print the corrected p-values
    print()
    print(
        "Pairwise Mann-Whitney U tests were performed to compare the distribution of"
        "\ncomplexity between clusters. p-values below have been corrected for multiple"
        "\ntesting using the Bonferroni method. "
    )
    print(f"Number of tests: {n_tests}")
    print(f"Significance marked as:")
    print("\tns  : p >= 0.05")
    print("\t*   : p < 0.05")
    print("\t**  : p < 0.01")
    print("\t*** : p < 0.001")
    print()

    # Print header for table with columns: cluster 1 vs cluster 2, significance, corrected p-value (original p-value)
    print(
        "Group 1 | Group 2 | Significance | Corrected p-value (original) | U-statistic"
    )
    print(
        "------- | ------- | ------------ | ---------------------------- | -----------"
    )

    height = dy / 3
    for i, j in sorted(clusters_yvals_and_pvals):
        y, pval, stat = clusters_yvals_and_pvals[(i, j)]
        corrected_pval = pval * n_tests
        if corrected_pval >= 0.05:
            significance = "ns"
        elif corrected_pval >= 0.01:
            significance = "*"
        elif corrected_pval >= 0.001:
            significance = "**"
        else:
            significance = "***"

        # Print results in a row of the table
        pvals = f"{corrected_pval:.4e}  ({pval:.4e})"
        print(f"{i:^7} | {j:^7} | {significance:^12} | {pvals:<28} | {stat:^11.3}")

        x1 = i - 1
        x2 = j - 1
        plt.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=1.5, c="k")
        plt.text(
            (x1 + x2) * 0.5,
            y + height,
            significance,
            ha="center",
            va="bottom",
            color="k",
        )
    print()

    sns.despine()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fname = Path(save_dir).joinpath(
            f"{today}_complexity_vs_cluster_boxplot{suffix}.{fmt}"
        )
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(fname, dpi=dpi, bbox_inches="tight")

    # Plot Q-hat vs complexity as a box plot
    fig = plt.figure(figsize=(boxplot_figsize[0] * 3, boxplot_figsize[1]))
    sns.boxplot(data=agg_data, x="complexity", y="Q_hat")
    plt.xlabel("Complexity")
    plt.ylabel(r"$\hat{Q}$")
    sns.despine()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fname = Path(save_dir).joinpath(
            f"{today}_Qhat_vs_complexity_boxplot{suffix}.{fmt}"
        )
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(fname, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    pdist_hdf = Path(
        "data/oscillation/FT_oscillator_pairwise_dist"
        "/231208_pairwise_dist_5TF_FT_oscs_min_visits100.hdf5"
    )
    save_dir = Path("figures/oscillation")

    main(
        pdist_hdf=pdist_hdf,
        n_clusts=2,
        save=True,
        save_dir=save_dir,
        suffix="_5TF_FT_oscillators",
        dpi=600,
        fmt="pdf",
    )
