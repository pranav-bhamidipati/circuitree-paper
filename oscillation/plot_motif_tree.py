from datetime import datetime
from math import ceil
from typing import Literal, Optional
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from circuitree import SimpleNetworkGrammar
import networkx as nx
from graph_utils import simplenetwork_complexity_layout

from oscillation import OscillationTree


def main(
    stats_csv: Path,
    tree_gml: Path,
    config_json: Path,
    n_visits_exhaustive: int = np.inf,
    node_size_scale: float = 30.0,
    min_samples: int = 0,
    figsize: tuple[float | int] = (9.0, 5.0),
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
    stats_df = pd.read_csv(Path(stats_csv)).dropna()
    stats_df = stats_df.sort_values(
        ["complexity", "odds_ratio"], ascending=(True, False)
    )
    stats_df["is_motif"] = (
        stats_df["significant"] & stats_df["overrepresented"] & stats_df["sufficient"]
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

    ### Plot the complexity graph
    print(f"Getting the complexity graph of motifs...")

    # The subgraph of successful nodes
    complexity_graph = ctree.to_complexity_graph(successes=motif_terminals)
    print(
        f"Complexity graph has {len(complexity_graph.nodes)} nodes "
        f"and {len(complexity_graph.edges)} edges"
    )

    # Get node positions
    print(f"Computing node positions...")
    pos: dict[str, tuple[float, float]] = simplenetwork_complexity_layout(
        complexity_graph, ctree.grammar
    )

    # Print a report of all sources (motifs with no predecessors in the graph)
    sources = [n for n in complexity_graph.nodes if complexity_graph.in_degree(n) == 0]
    sources = sorted(sources, key=lambda n: -pos[n][1])
    print(f"Found {len(sources)} sources of motifs:")
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

    ### Plot the complexity graph
    print(f"Plotting complexity graph...")

    # Highlight nodes that were sampled to exhaustion
    exhausted_motifs = set(
        n
        for n, v in zip(motif_nonterminals, motif_terminal_samples)
        if v >= n_visits_exhaustive
    )
    # log10_terminal_samples = np.log10(motif_terminal_samples)
    # if np.unique(log10_terminal_samples).size == 1:
    #     node_sizes = 1 + node_size_scale * log10_terminal_samples
    # else:
    #     node_sizes = (
    #         1 + node_size_scale * log10_terminal_samples / log10_terminal_samples.max()
    #     )
    node_colors = [
        "tab:green" if n in exhausted_motifs else "dimgray"
        for n in complexity_graph.nodes
    ]
    print(
        f"Found {len(exhausted_motifs)} motifs with >= {n_visits_exhaustive} samples:"
    )
    print(*(f"\t{m}" for m in exhausted_motifs), sep="\n")

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        complexity_graph,
        pos,
        width=0.5,
        edge_color="lightgray",
        # node_size=node_sizes,
        node_size=node_size_scale,
        arrows=False,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        complexity_graph,
        pos,
        edgecolors="none",
        # node_size=node_sizes,
        node_size=node_size_scale,
        node_color=node_colors,
        linewidths=0.0,
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
    stats_csv = data_dir / "analysis/231108_circuit_pattern_tests_depth12.csv"

    tree_gml = data_dir.joinpath(
        "backups/tree-28047823-dd31-4723-9dc1-f00ae6545013_2023-11-07_02-00-36.gml.gz"
    )
    config_json = data_dir.joinpath(
        "backups/tree-28047823-dd31-4723-9dc1-f00ae6545013_2023-11-04_12-32-24.json"
    )
    suffix = "_5tf_step2500k_with_exhaustion"

    save_dir = Path("figures/oscillation")

    main(
        stats_csv=stats_csv,
        tree_gml=tree_gml,
        config_json=config_json,
        n_visits_exhaustive=10_000,
        # min_samples=100,
        suffix=suffix,
        save=True,
        save_dir=save_dir,
        # fmt="pdf",
    )
