from datetime import datetime
from itertools import chain
from typing import Iterable, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import colormaps
from circuitree import SimpleNetworkGrammar
from circuitree.viz import plot_network
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from tqdm import tqdm
from graph_utils import (
    compute_Q_tilde,
    get_minimum_spanning_arborescence,
    make_complexity_graph,
    prune_bad_branches_inplace,
    simplenetwork_complexity_layout,
)

from oscillation import OscillationTree

# 1a. Load the dataframe of circuit patterns and p-values
# 1b. Load the tree of circuit patterns
# 2. Perform a Bonferroni correction on the p-values
# 3. Given a significance threshold, find the patterns that are significant
# 4. Get the subgraph of the tree that contains only the significant patterns (is it connected?)
# 5. Plot the subgraph, with the circuit diagram for each pattern


def get_complexity(pattern: str) -> int:
    return pattern.count("_") + 1


def plot_tree_with_circuits(
    motif_tree: nx.DiGraph,
    pos: dict[str, tuple[float, float]],
    motifs: Iterable[str],
    grammar: SimpleNetworkGrammar,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    edges = nx.draw_networkx_edges(
        motif_tree,
        pos,
        width=0.5,
        edge_color="darkgray",
        node_size=1000,
        arrowsize=0.01,
        ax=ax,
    )
    try:
        edges.set_zorder(1)
    except AttributeError:
        for edge in edges:
            edge.set_zorder(1)

    # colormap = colormaps.get_cmap("tab10")
    # cmap_vals = colormap(np.linspace(0, 1, colormap.N))
    # cmap_light_vals = cmap_vals
    # colormap_light = ListedColormap(cmap_light_vals)

    # All black colormap
    gray = colormaps.get_cmap("gray")(0.7)
    black = colormaps.get_cmap("gray")(0.0)
    colormap = ListedColormap([black] * 10)
    colormap_light = ListedColormap([gray] * 10)

    for node, (x, y) in tqdm(pos.items()):
        if node in motifs:
            kw = dict(colormap=colormap, color=black)
        else:
            kw = dict(colormap=colormap_light, color=gray, ec=gray)

        plot_network(
            *grammar.parse_genotype(node, nonterminal_ok=True),
            center=(x, y),
            plot_labels=False,
            **kw,
            **_network_kwargs,
        )


_network_kwargs = dict(
    padding=0.5,
    lw=1,
    node_shrink=0.4,
    offset=0.8,
    auto_shrink=0.9,
    width=0.005,
)


def main(
    stats_csv: Path,
    tree_gml: Path,
    tree_json: Path,
    plot_network_diagrams: bool = False,
    make_tree: bool = False,
    significance_threshold: float = 0.05,
    # aspect: float = 3.0,
    scale: float = 20.0,
    figsize: tuple[float | int] = (7.0, 4.0),
    save: bool = False,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
    dpi: int = 300,
    suffix: str = "",
):
    # Load the search tree and identify successful circuits
    tree = OscillationTree.from_file(
        Path(tree_gml), Path(tree_json), grammar_cls=SimpleNetworkGrammar
    )
    components = "".join(tree.grammar.components)
    successful_circuits = set(
        n
        for n in tree.graph.nodes
        if tree.grammar.is_terminal(n) and tree.is_success(n)
    )
    print(f"# oscillators found: {len(successful_circuits)}")

    # Load the dataframe of circuit patterns, p-values odds ratios, etc.
    stats_df = pd.read_csv(Path(stats_csv))
    stats_df["complexity"] = stats_df["pattern"].apply(get_complexity)
    stats_df = stats_df.set_index("pattern")

    # Find the patterns that are overrepresented and sufficient for oscillation (motifs)
    stats_df["significant"] = stats_df["p_corrected"] < significance_threshold
    stats_df["favorable"] = stats_df["odds_ratio"] > 1.0
    circuit_codes = [f"*{components}::{pat}" for pat in stats_df.index.values]
    stats_df["oscillator"] = [c in successful_circuits for c in circuit_codes]

    # ########################################
    # pattern_is_oscillator = []
    # for pat in stats_df.index.values:
    #     circuit = f"{components}::{pat}"
    #     attrs = tree.graph.nodes[circuit]
    #     pattern_is_oscillator.append(attrs["reward"] / attrs["visits"] > 0.01)
    # stats_df["oscillator"] = pattern_is_oscillator
    # ########################################

    motifs = stats_df.loc[
        stats_df["significant"] & stats_df["favorable"] & stats_df["oscillator"], :
    ].index.tolist()
    motif_circuits = [f"*{components}::{pat}" for pat in motifs]
    motif_nonterminals = set([mc[1:] for mc in motif_circuits])

    print(f"Found {len(motifs)} significant motifs:")
    print(stats_df.loc[motifs, :])

    # Get the subgraph of the search tree that leads to motifs and is favorable
    print(f"Getting subgraph of search graph with motifs...")
    if make_tree:
        # Get minimum spanning arborescence
        motif_graph = get_minimum_spanning_arborescence(tree=tree)

        # Remove branches that don't lead to motifs
        prune_bad_branches_inplace(
            motif_graph, root_node=tree.root, good_leaves=set(motifs)
        )

    else:
        # Keep the whole search graph leading to motifs
        nodes_to_keep = set()
        for mnt in tqdm(motif_nonterminals):
            nodes_leading_to_motif = chain.from_iterable(
                nx.bfs_layers(tree.graph.reverse(), mnt)
            )
            nodes_to_keep.update(nodes_leading_to_motif)
        motif_graph = tree.graph.subgraph(nodes_to_keep)

        # # Generate a complexity graph from the motifs
        # motif_graph = make_complexity_graph(
        #     tree.graph, tree.grammar, from_circuits=motif_circuits
        # )

    # Plot the subgraph, with the circuit diagram for each pattern
    print(f"Getting layout for graph...")

    pos: dict[str, tuple[float, float]] = simplenetwork_complexity_layout(
        motif_graph, tree.grammar
    )

    # ########################################
    # complexity_graph = make_complexity_graph(
    #     tree.graph, tree.grammar, from_circuits=successful_circuits
    # )
    # pos: dict[str, tuple[float, float]] = simplenetwork_complexity_layout(
    #     complexity_graph, tree.grammar
    # )
    # pos: dict[str, tuple[float, float]] = simplenetwork_complexity_layout(
    #     complexity_graph, tree.grammar
    # )

    # compute_Q_tilde(tree.graph, tree.root, inplace=True, in_edges=False)
    # for n1, n2 in tree.graph.edges:
    #     tree.graph.edges[n1, n2]["Q_loss"] = np.abs(
    #         tree.graph.nodes[n1]["Q_tilde"] - tree.graph.nodes[n2]["Q_tilde"]
    #     )
    # motif_graph = get_minimum_spanning_arborescence(tree=tree, weight="Q_loss")

    # # Remove branches that don't lead to successful circuits
    # prune_bad_branches_inplace(
    #     motif_graph,
    #     root_node=tree.root,
    #     good_leaves=successful_circuits,
    #     # good_leaves=set(motifs),
    # )
    # pos: dict[str, tuple[float, float]] = simplenetwork_complexity_layout(
    #     motif_graph, tree.grammar
    # )
    # ########################################

    # Print a report of all sources in the largest connected component
    conn_components = nx.weakly_connected_components(motif_graph)
    conn_components = sorted(conn_components, key=len, reverse=True)
    sources = [n for n in conn_components[0] if motif_graph.in_degree(n) == 0]
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

    print(f"Plotting graph of motifs...")
    fig, ax = plt.subplots(figsize=figsize)

    if plot_network_diagrams:
        # Scale xy positions to the size of circuit diagrams
        xmax = scale * (figsize[0] / figsize[1])
        ymax = scale
        xs, ys = np.array(list(pos.values())).T
        xs = xmax * (xs - xs.min()) / (xs.max() - xs.min())
        ys = ymax * (ys - ys.min()) / (ys.max() - ys.min())
        pos = dict(zip(pos.keys(), zip(xs, ys)))

        plot_tree_with_circuits(
            motif_graph,
            pos,
            motif_nonterminals,
            tree.grammar,
            ax=ax,
        )

        if save:
            today = datetime.now().strftime("%y%m%d")
            fname = save_dir / f"{today}_motif_tree{suffix}_with_networks.{fmt}"
            print(f"Saved figure to {fname}")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")

    else:
        # results_csv = Path("data/oscillation/230717_motifs.csv")
        # results_df = pd.read_csv(results_csv)
        # true_oscillators = set(
        #     results_df["state"].loc[results_df["p_oscillation"] >= 0.01]
        # )
        # oscillators = set(successful_circuits)

        # max_qval = max(motif_graph.nodes[n]["Q_tilde"] for n in oscillators)
        # normalize = Normalize(0.0, max_qval)
        # cmap = plt.cm.get_cmap("viridis")
        # gray = plt.cm.get_cmap("gray")(0.5)
        # blue = plt.cm.get_cmap("Blues")(1.0)
        # node_sizes = []
        # node_colors = []
        # false_positives = []
        # for n, qval in motif_graph.nodes("Q_tilde"):
        #     if n in motif_nonterminals:
        #         node_sizes.append(20)
        #         node_colors.append(blue)
        #     elif n in oscillators:
        #         clr = cmap(normalize(qval))
        #         node_sizes.append(20)
        #         node_colors.append(clr)
        #         if n not in true_oscillators:
        #             false_positives.append(n)
        #     else:
        #         node_sizes.append(2)
        #         node_colors.append(gray)

        node_colors = [
            "b" if n in motif_nonterminals else "k" for n in motif_graph.nodes
        ]
        nx.draw_networkx_edges(
            motif_graph,
            pos,
            width=0.5,
            edge_color="gray",
            arrows=False,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            motif_graph,
            pos,
            edgecolors="none",
            # node_color="k",
            node_size=25,
            node_color=node_colors,
            # node_size=node_sizes,
            ax=ax,
        )

        # # Plot a red dotted circle around false positives
        # for fp in false_positives:
        #     plt.scatter(*pos[fp], s=45, c="none", edgecolors="r", linewidths=0.5)

        if save:
            today = datetime.now().strftime("%y%m%d")
            fname = save_dir / f"{today}_motif_tree{suffix}.{fmt}"
            print(f"Saved figure to {fname}")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")

        ...


if __name__ == "__main__":
    data_dir = Path("data/oscillation/mcts/mcts_bootstrap_long_231022_173227")
    
    stats_csv = data_dir / "231026_circuit_pattern_tests_depth4.csv"
    # stats_csv = data_dir / "231027_circuit_pattern_tests_depth6.csv"
    # stats_csv = data_dir / "231027_circuit_pattern_tests_depth9.csv"
    # tree_gml = data_dir / "231026_merged_search_graph.gml.gz"
    tree_gml = data_dir / "5/oscillation_mcts_bootstrap_5000000_tree.gml"
    tree_json = data_dir / "231026_merged_search_graph.json"
    save_dir = Path("figures/oscillation")

    main(
        stats_csv=stats_csv,
        tree_gml=tree_gml,
        tree_json=tree_json,
        # make_tree=True,
        # plot_network_diagrams=True,
        significance_threshold=0.05,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
        dpi=300,
        suffix="_3tf_depth4_justmotifs_dag",
        # suffix="_3tf_depth4",
        # suffix="_3tf_depth6",
        # suffix="_3tf_depth9",
        # suffix="_3tf_depth9__",
    )
