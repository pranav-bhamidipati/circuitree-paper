from collections import Counter
from datetime import datetime
from functools import partial
from itertools import chain
import json
from math import ceil, floor
from typing import Literal, Optional
import warnings
from matplotlib import colors
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from circuitree import SimpleNetworkGrammar
import networkx as nx
from tqdm import tqdm
from graph_utils import compute_Q_tilde, simplenetwork_complexity_layout

from oscillation import OscillationTree


def count_successful_circuits_in_replicate(config_json: Path, gml: Path):
    successes = Counter()
    ctree = OscillationTree.from_file(
        gml, config_json, grammar_cls=SimpleNetworkGrammar
    )
    for n, d in ctree.graph.nodes(data=True):
        if (
            ctree.grammar.is_terminal(n)
            and d["visits"] > 0
            and d["reward"] / d["visits"] > ctree.Q_threshold
        ):
            successes[n] += 1
    return successes


def get_complexity(state: str) -> int:
    _, interactions_joined = state.split("::")
    if interactions_joined:
        return interactions_joined.count("_") + 1
    else:
        return 0


def main(
    stats_csv: Path,
    data_dir: Path,
    exhaustive_results_csv: Optional[Path] = None,
    exhaustive_stats_csv: Optional[Path] = None,
    n_exhaustive_samples: Optional[int] = None,
    gml_glob_str: str = "*/*_{}_tree.gml",
    json_glob_str: str = "*/*_{}_tree.json",
    step: int = 5_000_000,
    p_discovery_threshold=0.9,
    nprocs: int = 1,
    figsize: tuple[float | int] = (9.0, 5.0),
    node_size_scale: float = 120.0,
    guide_position: Literal["left", "right"] = "left",
    invert_xaxis: bool = False,
    save: bool = False,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
    dpi: int = 300,
    suffix: str = "",
):
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Load the config file and the list of search graphs for each replicate
    config_json = next(Path(data_dir).glob(json_glob_str.format(step)), None)
    if config_json is None:
        raise FileNotFoundError(
            f"No matches for the expression '{json_glob_str.format(step)}' "
            f"in directory: {data_dir}"
        )
    with open(config_json, "r") as f:
        components = "".join(json.load(f)["grammar"]["components"])

    search_graph_gmls = list(Path(data_dir).glob(gml_glob_str.format(step)))
    n_replicates = len(search_graph_gmls)

    # Load the dataframe of circuit patterns, p-values odds ratios, etc.
    stats_df = pd.read_csv(Path(stats_csv))
    stats_df["is_motif"] = (
        stats_df["significant"] & stats_df["overrepresented"] & stats_df["sufficient"]
    )

    agg_stats_df = stats_df.groupby("pattern").agg(
        complexity=("complexity", "first"),
        n_reps_with_motif=("is_motif", "sum"),
        odds_ratio_mean=("odds_ratio", "mean"),
        odds_ratio_std=("odds_ratio", "std"),
    )
    agg_stats_df = agg_stats_df.sort_values(
        ["complexity", "odds_ratio_mean"], ascending=(True, False)
    )
    agg_stats_df["pct_reps_with_motif"] = (
        agg_stats_df["n_reps_with_motif"] / n_replicates
    )
    agg_stats_df["nonterminal"] = [
        f"{components}::{pat}" for pat in agg_stats_df.index.values
    ]
    agg_stats_df["terminal"] = ["*" + nt for nt in agg_stats_df["nonterminal"]]
    agg_stats_df = agg_stats_df.set_index("nonterminal")

    if exhaustive_stats_csv is not None:
        # Contains Q-values (probability of oscillation) for each terminal circuit
        results_df = pd.read_csv(Path(exhaustive_results_csv))
        results_df = results_df.set_index("state")

        # Contains the results of testing each pattern in the exhaustive search
        # for overrepresentation
        exh_df = pd.read_csv(Path(exhaustive_stats_csv))
        exh_df["is_motif"] = (
            exh_df["significant"] & exh_df["overrepresented"] & exh_df["sufficient"]
        )
        n_motifs = exh_df["is_motif"].sum()
        print(f"Exhaustive search found {n_motifs} motifs")

    n_reps_found_successful_circuit = Counter()
    if nprocs == 1:
        for gml in tqdm(search_graph_gmls, desc="Loading data from replicates"):
            successes = count_successful_circuits_in_replicate(config_json, gml)
            n_reps_found_successful_circuit.update(successes)
    else:
        from multiprocessing import Pool

        count_successes = partial(count_successful_circuits_in_replicate, config_json)

        print(f"Using {nprocs} processes to load data from replicates")
        with Pool(nprocs) as pool:
            pbar = tqdm(
                total=len(search_graph_gmls), desc="Loading data from replicates"
            )
            for successes in pool.imap_unordered(
                count_successes, search_graph_gmls, chunksize=1
            ):
                pbar.update(1)
                n_reps_found_successful_circuit.update(successes)
            pbar.close()

    ### Plot the complexity graph
    print(f"Getting the complexity graph and its layout...")

    # Build the graph of the whole design space (tractable for small design spaces)
    ctree = OscillationTree.from_file(
        None, config_json, grammar_cls=SimpleNetworkGrammar
    )
    ctree.grow_tree()

    # If a dataframe of pattern test results from an exhaustive search is supplied, make
    # the complexity graph from successes in that dataframe. Otherwise, use successful
    # circuits found from the MCTS search.
    print("Using successes from the exhaustive search to make the complexity graph")

    exh_df["nonterminal"] = ["::".join([components, pat]) for pat in exh_df["pattern"]]
    exh_df["terminal"] = ["*" + nt for nt in exh_df["nonterminal"]]
    exh_df = exh_df.set_index("nonterminal")

    # Insert the visits and reward data into the search graph
    for n in ctree.terminal_states:
        Qval = results_df.loc[n, "p_oscillation"]
        ctree.graph.nodes[n]["visits"] = n_exhaustive_samples
        ctree.graph.nodes[n]["reward"] = Qval * n_exhaustive_samples

    # Compute the Q_tilde values for each node
    compute_Q_tilde(ctree.graph, ctree.root, inplace=True)

    complexity_graph = ctree.to_complexity_graph(
        successes=set(exh_df["terminal"].values)
    )

    # Get node positions
    pos: dict[str, tuple[float, float]] = simplenetwork_complexity_layout(
        complexity_graph, ctree.grammar
    )

    for n, (x, y) in pos.items():
        if n.count("_") == 2:
            print(f"{n=}\n\t{x=:0.2f}")

    # Add Q_tilde values to the results dataframe
    Q_tildes = dict(ctree.graph.nodes("Q_tilde"))
    results_df["nonterminal"] = [n[1:] for n in results_df.index.values]
    results_df["Q_tilde"] = results_df["nonterminal"].map(Q_tildes).values
    P_discoveries = (
        results_df["nonterminal"].map(agg_stats_df["pct_reps_with_motif"]).values
    )
    P_discoveries[np.isnan(P_discoveries)] = 0.0
    results_df["P_discovery"] = P_discoveries
    results_df["complexity"] = results_df["nonterminal"].map(get_complexity)

    # Print a report of all sources (motifs with no predecessors in the graph)
    sources = [n for n in complexity_graph.nodes if complexity_graph.in_degree(n) == 0]
    sources = sorted(sources, key=lambda n: -pos[n][1])
    print(f"Found {len(sources)} sources in the largest connected component:")
    last_complexity = -1
    for source in sources:
        interactions_joined = source.split("::")[1]
        if interactions_joined:
            complexity = interactions_joined.count("_") + 1
        else:
            complexity = 0
        if complexity != last_complexity:
            print()
            print(f" Complexity {complexity}")
            print(f"====================")
            last_complexity = complexity
        print(f"  {source}")
    print()

    print(f"Plotting complexity graph...")

    # Node color is continuous, based on P(motif discovery) for all motifs in exhaustive search
    exh_motifs = set(exh_df.loc[exh_df["is_motif"]].index.values)
    cmap = plt.get_cmap("coolwarm")
    searched_patterns = set(agg_stats_df.index.values)
    not_motifs = set()
    nodelist = []
    node_p_discoveries = []
    for i, nt in enumerate(complexity_graph.nodes):
        if nt in exh_motifs:
            nodelist.append(nt)
            if nt in searched_patterns:
                p_discovery = agg_stats_df.loc[nt, "pct_reps_with_motif"]
            else:
                p_discovery = 0.0
            node_p_discoveries.append(p_discovery)
        else:
            not_motifs.add(nt)

    node_colors = cmap(node_p_discoveries)

    # Node size depends on Q_tilde
    node_qtildes = np.array([ctree.graph.nodes[n]["Q_tilde"] for n in nodelist])
    node_sizes = 1 + node_size_scale * node_qtildes

    # # Node size is based on P(motif discovery) for all motifs in exhaustive search
    # searched_patterns = set(agg_stats_df.index.values)
    # node_sizes = []
    # for i, nt in enumerate(complexity_graph.nodes):
    #     if nt in exh_motifs:
    #         nodelist.append(nt)
    #         if nt in searched_patterns:
    #             p_discovery = agg_stats_df.loc[nt, "pct_reps_with_motif"]
    #         else:
    #             p_discovery = 0.0
    #         nsize = 2 + node_size_scale * p_discovery
    #         node_sizes.append(nsize)
    #     else:
    #         not_motifs.add(nt)

    # # Node color depends on Q_tilde
    # cmap = plt.get_cmap("coolwarm")
    # node_q_tildes = np.array([ctree.graph.nodes[n]["Q_tilde"] for n in nodelist])
    # node_colors = cmap(node_q_tildes / node_q_tildes.max())

    faint_nodes = []
    faint_node_p_discoveries = []
    for i, nt in enumerate(complexity_graph.nodes):
        faint_nodes.append(nt)
        if nt in searched_patterns:
            p_discovery = agg_stats_df.loc[nt, "pct_reps_with_motif"]
        else:
            p_discovery = 0.0
        faint_node_p_discoveries.append(p_discovery)

    faint_node_colors = cmap(faint_node_p_discoveries)

    # faint_nodes = list(not_motifs)
    faint_node_sizes = np.array([ctree.graph.nodes[n]["Q_tilde"] for n in faint_nodes])
    faint_node_sizes = 1 + node_size_scale * faint_node_sizes

    # # Nodes that don't meet the overrepresentation requirement to be a motif are
    # # drawn faintly
    # faint_edges = set(
    #     (u, v)
    #     for u, v in complexity_graph.edges
    #     if u in not_motifs or v in not_motifs
    # )
    # solid_edges = set(complexity_graph.edges) - faint_edges

    fig, ax = plt.subplots(figsize=figsize)

    # Draw any faint edges behind the solid edges, since alpha is not supported
    # if faint_edges:
    #     nx.draw_networkx_edges(
    #         complexity_graph,
    #         pos,
    #         # edgelist=faint_edges,
    #         width=0.5,
    #         edge_color="silver",
    #         arrows=False,
    #         ax=ax,
    #     )

    nx.draw_networkx_edges(
        complexity_graph,
        pos,
        width=0.5,
        # edge_color="silver",
        edge_color="gainsboro",
        arrows=False,
        ax=ax,
    )

    # Highlight edges between motifs with high discovery rate
    highlight_edges = []
    for u, v in complexity_graph.edges:
        # if u in exh_motifs and v in exh_motifs:
        u_p_discovery = results_df.loc[
            results_df["nonterminal"] == u, "P_discovery"
        ].item()
        v_p_discovery = results_df.loc[
            results_df["nonterminal"] == v, "P_discovery"
        ].item()
        if (
            u_p_discovery > p_discovery_threshold
            and v_p_discovery > p_discovery_threshold
        ):
            highlight_edges.append((u, v))

    highlighted = set(chain.from_iterable(highlight_edges))
    high_P_discovery = set(
        results_df.loc[
            results_df["P_discovery"] > p_discovery_threshold, "nonterminal"
        ].values.flatten()
    )
    n_not_highlighted = len(high_P_discovery - highlighted)
    print(f"How many high-P_discovery-motifs are not highlighted? {n_not_highlighted}")

    nx.draw_networkx_edges(
        complexity_graph,
        pos,
        edgelist=highlight_edges,
        width=0.5,
        edge_color="black",
        arrows=False,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        complexity_graph,
        pos,
        nodelist=faint_nodes,
        edgecolors="none",
        node_size=faint_node_sizes,
        # node_color="silver",
        node_color=faint_node_colors,
        linewidths=0.5,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        complexity_graph,
        pos,
        nodelist=nodelist,
        edgecolors="black",
        linewidths=1.0,
        node_size=node_sizes,
        node_color=node_colors,
        ax=ax,
    )

    plt.axis("off")

    # Plot a guide for the number of interactions in each level in the tree
    x_min, x_max = plt.xlim()
    if guide_position == "right":
        x_guide = x_max + 0.05 * (x_max - x_min)
    elif guide_position == "left":
        x_guide = x_min - 0.05 * (x_max - x_min)
    else:
        raise ValueError(f"Invalid position for guide labels: {guide_position}")
    dx = 0.05 * (x_max - x_min)

    level_yvals = sorted(set(y for x, y in pos.values()), reverse=True)
    min_depth = min(c.count("_") for c in complexity_graph.nodes) + 1
    dy = np.diff(-np.array(level_yvals)).min()
    plt.text(x_guide, level_yvals[0] + dy, "Complexity", ha="center", va="center")
    for i, y in enumerate(level_yvals):
        plt.text(x_guide, y, i + min_depth, ha="center", va="center")

    # Plot markers for a range of Q_tilde values
    marker_qtildes = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    marker_sizes = 1 + node_size_scale * marker_qtildes
    x_markers = x_min + dx * (4 + np.arange(marker_qtildes.size))
    y_marker = level_yvals[0] + dy / 2
    plt.text(
        np.median(x_markers),
        y_marker + dy / 2,
        r"$Q_\mathrm{motif}$",
        ha="center",
        va="center",
    )
    marker_pos = {i: (x, y_marker) for i, x in enumerate(x_markers)}
    G_marker = nx.Graph()
    G_marker.add_nodes_from(marker_pos.keys())

    nx.draw_networkx_nodes(
        G_marker,
        marker_pos,
        node_size=marker_sizes,
        node_color="dimgray",
        edgecolors="k",
        linewidths=1.0,
    )

    for i, qtilde in enumerate(marker_qtildes):
        # plt.scatter(x_markers[i], y_marker, s=marker_sizes[i], color="dimgray", edgecolors="none")
        plt.text(
            x_markers[i],
            y_marker - dy / 3,
            f"{qtilde:.1f}",
            ha="center",
            va="center",
            fontsize="small",
        )

    # Plot markers to show that a black outline indicates that the node is a true motif
    x_motif_example = x_max - 6 * dx
    y_motif_example = level_yvals[0] + dy / 2
    pos_motif_example = {
        0: (x_motif_example - dx / 2, y_motif_example),
        1: (x_motif_example + dx / 2, y_motif_example),
    }
    G_motif_example = nx.Graph()
    G_motif_example.add_nodes_from(pos_motif_example.keys())
    nx.draw_networkx_nodes(
        G_motif_example,
        pos_motif_example,
        node_size=marker_sizes[marker_sizes.size // 2],
        node_color="dimgray",
        edgecolors=("none", "k"),
        linewidths=1.0,
    )
    plt.text(
        x_motif_example,
        y_marker + dy / 2,
        "True motif",
        ha="center",
        va="center",
    )
    plt.text(
        x_motif_example - dx / 2,
        y_motif_example - dy / 3,
        "No",
        ha="center",
        va="center",
        fontsize="small",
    )
    plt.text(
        x_motif_example + dx / 2,
        y_motif_example - dy / 3,
        "Yes",
        ha="center",
        va="center",
        fontsize="small",
    )

    # Plot colorbar for P(motif discovery) = coolwarm matplotlib colormap
    # Horizontal at the top-right of the plot
    cmap = plt.get_cmap("coolwarm")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Plot the colorbar at the top-right of the plot, inside the plot area
    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        location="top",
        shrink=0.1,
        aspect=4,
        ticks=[0.0, 0.5, 1.0],
        panchor=(0.6, 0.75),
    )
    # Put the label below the colorbar
    cbar.set_label("P(Assembly motif)")

    # Optionally flip the x-axis direction for aesthetic reasons
    if invert_xaxis:
        plt.gca().invert_xaxis()

    if save:
        today = datetime.now().strftime("%y%m%d")
        fname = save_dir / f"{today}_complexity_graph_from_replicates{suffix}.{fmt}"
        print(f"Saved figure to {fname}")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")

    ...

    # import seaborn as sns

    # # Plot P(motif discovery) vs log(Q_tilde)
    # fig = plt.figure()

    # sns.scatterplot(
    #     data=results_df.loc[results_df["p_oscillation"] >= ctree.Q_threshold],
    #     x="Q_tilde",
    #     y="P_discovery",
    #     edgecolor="none",
    #     palette="husl",
    #     hue="complexity",
    #     legend="full",
    #     alpha=0.5,
    #     s=10,
    # )

    # # Move the legend outside the plot, adding a title "Complexity"
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, title="Complexity")

    # plt.xlabel(r"$\tilde{Q}$ in oscillators")
    # plt.xscale("log")
    # plt.ylabel("P(motif discovery)")

    # sns.despine()

    # if save:
    #     today = datetime.now().strftime("%y%m%d")
    #     fname = save_dir / f"{today}_motif_discovery_vs_Q_tilde{suffix}.{fmt}"
    #     print(f"Saved figure to {fname}")
    #     fig.savefig(fname, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    exhaustive_results_csv = Path("data/oscillation/230717_motifs.csv")
    exhaustive_stats_csv = Path(
        "data/oscillation/231119_circuit_pattern_tests_exhaustive_search.csv"
        # "data/oscillation/231101_circuit_pattern_tests_exhaustive_search_depth9.csv"
        # "data/oscillation/231030_circuit_pattern_tests_exhaustive_search_depth4.csv"
    )

    # data_dir = Path("data/oscillation/mcts/mcts_bootstrap_long_231022_173227")
    # stats_csv = data_dir / "231030_circuit_pattern_tests_reps12_depth4.csv"
    # suffix="_3tf_reps12_depth4_steps5mil",
    # step = 5_000_000

    # data_dir = Path("data/oscillation/mcts/mcts_bootstrap_short_231020_175449")
    # stats_csv = data_dir / "231031_circuit_pattern_tests_reps50_depth4.csv"
    # suffix = "_3tf_reps50_depth4_steps100k"
    # stats_csv = data_dir / "231031_circuit_pattern_tests_reps50_depth9.csv"
    # suffix = "_3tf_reps50_depth9_steps100k"
    # step = 100_000

    data_dir = Path(
        "data/oscillation/mcts/mcts_bootstrap_short_exploration2.00_231103_140501"
    )
    stats_csv = data_dir / "231103_circuit_pattern_tests_reps50_depth9.csv"
    suffix = "_3tf_reps50_depth9_steps100k_exploration2.00"
    step = 100_000

    save_dir = Path("figures/oscillation")

    main(
        stats_csv=stats_csv,
        data_dir=data_dir,
        exhaustive_results_csv=exhaustive_results_csv,
        exhaustive_stats_csv=exhaustive_stats_csv,
        n_exhaustive_samples=10_000,
        p_discovery_threshold=0.8,
        node_size_scale=150,
        # invert_xaxis=True,
        step=step,
        nprocs=12,
        suffix=suffix,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
