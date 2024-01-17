from datetime import datetime
from math import ceil
from typing import Literal, Optional
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps
from circuitree import SimpleNetworkGrammar
from circuitree.viz import plot_network
import networkx as nx
from graph_utils import compute_Q_tilde, simplenetwork_complexity_layout

from oscillation import OscillationTree


_network_kwargs = dict(
    fontsize=6,
    padding=0.5,
    lw=1,
    node_shrink=0.7,
    offset=0.8,
    auto_shrink=0.9,
    width=0.005,
    plot_labels=False,
)


def main(
    stats_csv: Path,
    tree_gml: Path,
    config_json: Path,
    n_visits_exhaustive: int = np.inf,
    node_size_scale: float = 50.0,
    min_samples: int = 0,
    figsize: tuple[float | int] = (4.0, 4.0),
    guide_position: Literal["left", "right"] = "left",
    save: bool = False,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
    dpi: int = 300,
    suffix: str = "",
):
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Load the config file and the search graph
    print(f"Loading CiruiTree object from file...")
    ctree = OscillationTree.from_file(
        tree_gml, config_json, grammar_cls=SimpleNetworkGrammar
    )
    components = "".join(ctree.grammar.components)

    # Load the dataframe of circuit patterns, p-values odds ratios, etc.
    print(f"Loading data of statistical tests on circuit patterns...")
    stats_df = pd.read_csv(Path(stats_csv))

    # Any samples with no observations in either category are not overrepresented
    stats_df["overrepresented"].loc[
        (stats_df["pattern_in_null"] == 0) & (stats_df["pattern_in_succ"] == 0)
    ] = False

    # For now, we will consider patterns that are not observed in the null distribution
    # even though their significance cannot be quantified
    stats_df["conditionally_significant"] = (stats_df["pattern_in_null"] == 0) & (
        stats_df["pattern_in_succ"] > 0
    )

    stats_df = stats_df.sort_values(
        ["complexity", "odds_ratio"], ascending=(True, False)
    )
    stats_df["is_motif"] = (
        stats_df["overrepresented"]
        & stats_df["sufficient"]
        & (stats_df["significant"] | stats_df["conditionally_significant"])
    )
    stats_df["nonterminal"] = [
        f"{components}::{pat}" for pat in stats_df["pattern"].values
    ]
    stats_df["terminal"] = ["*" + nt for nt in stats_df["nonterminal"]]
    stats_df = stats_df.set_index("nonterminal")
    motif_nonterminals = stats_df.loc[stats_df["is_motif"]].index.values.tolist()
    motif_terminals = stats_df.loc[stats_df["is_motif"]]["terminal"].values.tolist()
    print(f"Found {len(motif_nonterminals)} motifs:")
    print(stats_df.loc[stats_df["is_motif"], ["complexity", "odds_ratio", "pvalue"]])
    print()

    # Keep nodes that were sufficiently sampled
    motif_terminal_samples = np.array(
        [ctree.graph.nodes[n]["visits"] for n in motif_terminals]
    )
    where_enough_samples = np.where(motif_terminal_samples >= min_samples)[0]
    motif_terminal_samples = motif_terminal_samples[where_enough_samples]
    motif_terminals = [
        n for i, n in enumerate(motif_terminals) if i in where_enough_samples
    ]
    motif_nonterminals = [n[1:] for n in motif_terminals]
    print(f"Keeping {len(motif_terminals)} motifs with >= {min_samples} samples")

    # # Rank the motifs by Q-hat
    # motif_qhat = {
    #     nt: ctree.graph.nodes[t]["reward"] / ctree.graph.nodes[t]["visits"]
    #     for nt, t in zip(motif_nonterminals, motif_terminals)
    # }
    # motif_qhat = dict(sorted(motif_qhat.items(), key=lambda item: -item[1]))

    # Compute Q-tilde for each node in the tree
    compute_Q_tilde(ctree.graph, ctree.root, inplace=True)

    # Identify motifs that were sampled to exhaustion
    exhausted_motifs = set(
        nt
        for nt, t in zip(motif_nonterminals, motif_terminals)
        if ctree.graph.nodes[t]["visits"] >= n_visits_exhaustive
    )

    ### Plot the complexity graph
    print(f"Getting the complexity graph of motifs...")

    # The subgraph of successful nodes
    complexity_graph = ctree.to_complexity_graph(successes=motif_terminals)
    print(
        f"Complexity graph has {len(complexity_graph.nodes)} nodes "
        f"and {len(complexity_graph.edges)} edges"
    )

    # # Print a report of all sources (motifs with no predecessors in the graph)
    # sources = [n for n in complexity_graph.nodes if complexity_graph.in_degree(n) == 0]
    # sources = sorted(sources, key=lambda n: -pos[n][1])
    # print(f"Found {len(sources)} sources of motifs:")
    # last_complexity = -1
    # for source in sources:
    #     interactions_joined = source.split("::")[1]
    #     if interactions_joined:
    #         complexity = interactions_joined.count("_") + 1
    #     else:
    #         complexity = 0
    #     if complexity != last_complexity:
    #         print()
    #         print(f" Complexity {complexity}")
    #         print(f"====================")
    #         last_complexity = complexity
    #     print(f"  {source}")
    # print()

    # Restrict this to motifs that were exhaustively sampled and any successors with
    # high enough oscillation probability (Q_tilde)
    highlight_edges = [
        (n1, n2)
        for n1, n2 in complexity_graph.edges
        if n1 in exhausted_motifs
        and (n2 in exhausted_motifs or complexity_graph.nodes[n2]["Q_tilde"] > 0.4)
    ]
    complexity_graph = complexity_graph.edge_subgraph(highlight_edges)
    print(
        f"Restricting the complexity graph to {len(complexity_graph.nodes)} nodes that"
        f" were exhaustively sampled and their successors that are either exhausted or "
        f" have suffiicent Q_tilde."
    )

    node_qtildes = np.array([q for _, q in complexity_graph.nodes(data="Q_tilde")])
    ranks = np.argsort(-node_qtildes).argsort()

    print("Top Q-tilde values:")
    argsort = np.argsort(-node_qtildes)
    top_qtildes = node_qtildes[argsort][:20]
    top_nodes = np.array(complexity_graph.nodes)[argsort][:20]
    print(*[f"\t{q:.3f}  <-- {n}" for n, q in zip(top_nodes, top_qtildes)], sep="\n")

    # Get node positions
    print(f"Computing node positions...")
    pos: dict[str, tuple[float, float]] = simplenetwork_complexity_layout(
        complexity_graph, ctree.grammar
    )

    ### Plot the complexity graph
    print(f"Plotting complexity graph...")

    # log10_terminal_samples = np.log10(motif_terminal_samples)
    # if np.unique(log10_terminal_samples).size == 1:
    #     node_sizes = 1 + node_size_scale * log10_terminal_samples
    # else:
    #     node_sizes = (
    #         1 + node_size_scale * log10_terminal_samples / log10_terminal_samples.max()
    #     )
    # node_colors = [
    #     "tab:green" if n in exhausted_motifs else "dimgray"
    #     for n in complexity_graph.nodes
    # ]

    ############################################

    # # Color nodes by Q-hat ranking
    # node_qhats = np.array([motif_qhat[n] for n in complexity_graph.nodes])
    # ranks = np.argsort(-node_qhats).argsort()
    # node_labels = dict(zip(complexity_graph.nodes, [f"{q:.2f}" for q in node_qhats]))
    # cmap = plt.cm.get_cmap("viridis_r")
    # node_colors = cmap(ranks / ranks.max())

    cmap = plt.colormaps.get_cmap("viridis")
    node_colors = cmap(node_qtildes / node_qtildes.max())

    # node_sizes = 5 + node_size_scale * node_qtildes / node_qtildes.max()
    # node_qtilde_map = dict(zip(complexity_graph.nodes, node_qtildes))

    # node_labels = dict(zip(complexity_graph.nodes, [f"{q:.3f}" for q in node_qtildes]))

    ############################################

    # Outline nodes that were sampled to exhaustion
    node_edgecolors = [
        "black" if n in exhausted_motifs else "none" for n in complexity_graph.nodes
    ]

    # # Highlight edges outgoing from exhausted motifs to either another exhausted
    # # motif or a motif with Q_tilde > 0.4
    # highlight_edges = [
    #     (n1, n2)
    #     for n1, n2 in complexity_graph.edges
    #     if n1 in exhausted_motifs
    #     and (n2 in exhausted_motifs or node_qtilde_map[n2] > 0.4)
    # ]

    # # Highlight edges between nodes with Q_tilde > 0.4
    # highlight_edges = [
    #     (n1, n2)
    #     for n1, n2 in complexity_graph.edges
    #     if node_qtilde_map[n1] > 0.4 and node_qtilde_map[n2] > 0.4
    # ]

    print(
        f"Found {len(exhausted_motifs)} motifs with >= {n_visits_exhaustive} samples:"
    )
    print(*(f"\t{m}" for m in exhausted_motifs), sep="\n")

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        complexity_graph,
        pos,
        width=1.0,
        edge_color="gray",
        # node_size=node_sizes,
        node_size=node_size_scale,
        arrows=False,
        ax=ax,
    )
    # nx.draw_networkx_edges(
    #     complexity_graph,
    #     pos,
    #     width=0.5,
    #     edgelist=highlight_edges,
    #     edge_color="black",
    #     node_size=node_sizes,
    #     # node_size=node_size_scale,
    #     arrows=False,
    #     ax=ax,
    # )
    nx.draw_networkx_nodes(
        complexity_graph,
        pos,
        edgecolors=node_edgecolors,
        node_color=node_colors,
        # node_color="gray",
        # node_size=node_sizes,
        node_size=node_size_scale,
        linewidths=1.0,
        ax=ax,
    )

    # ############################################
    # # Label each node with its qhat value
    # nx.draw_networkx_labels(complexity_graph, pos, labels=node_labels, font_size=6)
    # ############################################

    plt.axis("off")

    # Plot a guide for the number of interactions in each level in the tree
    x_min, x_max = plt.xlim()
    if guide_position == "right":
        x_guide = x_max + 0.05 * (x_max - x_min)
    elif guide_position == "left":
        x_guide = x_min - 0.05 * (x_max - x_min)
    else:
        raise ValueError(f"Invalid position for guide labels: {guide_position}")

    # Plot a guide for the complexity of each level in the tree
    edge0 = next(iter(complexity_graph.edges))
    dy = pos[edge0[0]][1] - pos[edge0[1]][1]
    depth_min = min(c.count("_") for c in complexity_graph.nodes) + 1
    depth_max = max(c.count("_") for c in complexity_graph.nodes) + 1
    y_max = max(pos[n][1] for n in complexity_graph.nodes)

    plt.text(x_guide, y_max + dy, "Complexity", ha="center", va="center")
    yval = y_max
    for depth in range(depth_min, depth_max + 1):
        plt.text(x_guide, yval, str(depth), ha="center", va="center")
        yval -= dy

    # Plot colorbar for Q_tilde values
    # Horizontal at the top-right of the plot
    norm = colors.Normalize(vmin=0.0, vmax=node_qtildes.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Plot a colorbar at the top-right of the plot, inside the plot area
    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        location="top",
        shrink=0.2,
        aspect=4,
        ticks=[0.0, 0.25, 0.5],
        format="%.2f",
        panchor=(0.6, 0.75),
    )
    # Put the label below the colorbar
    cbar.set_label(r"$\tilde{Q}$")

    # # Plot markers for a range of visit numbers for the terminal states (log-scale)
    # y_marker = y_max + dy / 2
    # x_min, x_max = plt.xlim()
    # dx = 0.05 * (x_max - x_min)
    # x_marker = x_guide + 4 * dx
    # plt.text(
    #     x_marker + 2 * dx,
    #     y_marker + dy / 2,
    #     r"$n_\mathrm{samples}$",
    #     ha="center",
    #     va="center",
    #     fontsize="medium",
    # )
    # for log10_samples in range(5):
    #     size = 1 + node_size_scale * log10_samples / log10_terminal_samples.max()
    #     plt.scatter(x_marker, y_marker, s=size, color="dimgray", edgecolors="none")
    #     plt.text(
    #         x_marker,
    #         y_marker - dy / 3,
    #         rf"$10^{{{log10_samples}}}$",
    #         ha="center",
    #         va="center",
    #         fontsize="small",
    #     )
    #     x_marker += dx

    if save:
        today = datetime.now().strftime("%y%m%d")
        fname = save_dir / f"{today}_complexity_graph{suffix}.{fmt}"
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(fname, dpi=dpi, bbox_inches="tight")

    print("Saving network diagrams of the top Q-tilde values...")

    for qtilde, motif in zip(top_qtildes, top_nodes):
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        plot_network(
            *ctree.grammar.parse_genotype(motif, nonterminal_ok=True),
            ax=ax,
            **_network_kwargs,
        )

        plt.title(rf"$\tilde{{Q}} = {qtilde:.3f}$", fontsize=8)

        if save:
            today = datetime.now().strftime("%y%m%d")
            fname = save_dir / f"{today}_motif_{motif}.{fmt}"
            print(f"Writing to: {fname.resolve().absolute()}")
            plt.savefig(fname, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    # data_dir = Path("data/oscillation/mcts/mcts_bootstrap_long_231022_173227")

    # stats_csv = data_dir / "231026_circuit_pattern_tests_depth4.csv"
    # # stats_csv = data_dir / "231027_circuit_pattern_tests_depth6.csv"
    # # stats_csv = data_dir / "231027_circuit_pattern_tests_depth9.csv"
    # # tree_gml = data_dir / "231026_merged_search_graph.gml.gz"
    # tree_gml = data_dir / "5/oscillation_mcts_bootstrap_5000000_tree.gml"
    # tree_json = data_dir / "231026_merged_search_graph.json"

    data_dir = Path(
        "data/aws_exhaustion_exploration2.00/231104-19-32-24_5tf_exhaustion"
        "_mutationrate0.5_batch1_max_interactions15_exploration2.000"
    )
    # stats_csv = data_dir / "analysis/231107_circuit_pattern_tests_depth9.csv"
    # stats_csv = data_dir / "analysis/231108_circuit_pattern_tests_depth12.csv"
    stats_csv = data_dir / "analysis/231109022226_circuit_pattern_tests_depth12.csv"

    tree_gml = data_dir.joinpath(
        "backups/tree-28047823-dd31-4723-9dc1-f00ae6545013_2023-11-07_02-00-36.gml.gz"
    )
    config_json = data_dir.joinpath(
        "backups/tree-28047823-dd31-4723-9dc1-f00ae6545013_2023-11-04_12-32-24.json"
    )
    suffix = "_5tf_step2500k_with_exhaustion"

    save_dir = Path("figures/oscillation/complexity_graphs")
    save_dir.mkdir(exist_ok=True)

    main(
        stats_csv=stats_csv,
        tree_gml=tree_gml,
        config_json=config_json,
        n_visits_exhaustive=10_000,
        min_samples=10,
        suffix=suffix,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
