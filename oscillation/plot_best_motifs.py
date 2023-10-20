# Load in a search graph
# Compute Q-tilde for each edge and node
# Compute Delta(Q-tilde) and IG(Q-tilde) for each edge in relation to its parent
# ?? Error bars ??
# Plot IG(Q-tilde) vs Delta(Q-tilde) for each edge


from datetime import datetime
import json
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Optional
import warnings

from circuitree.viz import plot_network
from circuitree.models import SimpleNetworkGrammar
from circuitree.modularity import information_gain
from graph_utils import compute_Q_tilde, merge_search_graphs

_network_kwargs = dict(
    padding=0.5,
    lw=1,
    node_shrink=0.7,
    offset=0.8,
    auto_shrink=0.9,
    width=0.005,
    plot_labels=False,
)


def main(
    graph_gml: Path,
    root_node: Optional[Any] = None,
    attrs_json: Optional[Path] = None,
    save: bool = False,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
    dpi: int = 300,
    figsize: tuple = (5, 3),
    network_figsize: tuple = (3, 3),
    network_kwargs: Optional[dict] = None,
    n_motifs: int = 8,
    palette: str = "muted",
    multiple: str = "fill",
    legend_title: str = "Motif combination",
    bins=100,
):
    # Load the search graph, a DAG with a single root, from one or more GML files
    if isinstance(graph_gml, Path):
        G: nx.DiGraph = nx.read_gml(graph_gml)
    else:
        print(f"Merging {len(graph_gml)} search graphs...")
        search_graphs = [nx.read_gml(gml) for gml in graph_gml]
        G: nx.DiGraph = merge_search_graphs(search_graphs)
        print(f"Done. Merged graph has {len(G.nodes)} nodes.")

    # Get the root node
    if root_node is None:
        with Path(attrs_json).open("r") as f:
            root_node = json.load(f)["root"]

    # Compute Q-tilde for each edge and node
    compute_Q_tilde(G, root_node=root_node)

    # Compute Delta P and Delta H for each edge and store
    edges = []
    dPs = []
    dHbars = []
    Qroot = G.nodes[root_node]["Q_tilde"]
    for *edge, Qj in G.edges(data="Q_tilde"):
        Qi = G.nodes[edge[0]]["Q_tilde"]
        dP = Qj - Qi
        if dP == 0:
            continue
        dHbar = information_gain(Qj, Qroot)
        edges.append(tuple(edge))
        dPs.append(dP)
        dHbars.append(dHbar)

    dP_dict = dict(zip(edges, dPs))

    # tell seaborn to shut up
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    # Plot Delta P vs Delta H
    # sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    p = sns.scatterplot(x=dPs, y=dHbars, s=5)
    plt.tight_layout()

    # # Plot Delta P on the x axis with jitter
    # sns.set_theme(style="whitegrid")
    # fig = plt.figure(figsize=figsize)
    # p = sns.stripplot(x=dPs, jitter=0.3)
    # plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_IG_vs_dP_3tf_1mil.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)

    ...

    # Given a number of actions M to choose from, compute the probability of success for each action combination
    # This is just the Q-tilde of each node layer M
    # Should exclude edges that do termination
    layer_num = []
    nodes = []
    p_success = []
    visits = []
    for i, layer in enumerate(nx.bfs_layers(G, root_node)):
        nonterminals = [n for n in layer if G.out_degree(n) > 0]
        nodes.extend(nonterminals)
        # rewards = [G.nodes[n]["Q_tilde"] for n in nonterminals]
        p_success.extend([G.nodes[n]["Q_tilde"] for n in nonterminals])
        visits.extend([G.nodes[n]["visits"] for n in nonterminals])
        layer_num.extend([i] * len(nonterminals))
    df = pd.DataFrame(dict(node=nodes, p_success=p_success, visits=visits))
    df["layer"] = pd.Categorical(
        layer_num, categories=sorted(set(layer_num)), ordered=True
    )
    # Standard error of the mean of Bernoulli is sqrt(p(1-p) / N)
    df["std_dev"] = np.sqrt(df["p_success"] * (1 - df["p_success"]) / df["visits"])

    # Plot the distribution of Q-tilde on the x axis and the # layers on the y axis
    # sns.set_theme(style="whitegrid")
    # df = df.sort_values("p_success", ascending=False).head(100)
    df["log_Qratio"] = np.log10(best_motifs_in_layer_df["p_success"] / Qroot)
    fig = plt.figure(figsize=figsize)
    plt.scatter(df["log_Qratio"], df["layer"], s=1)
    plt.errorbar(
        df["log_Qratio"],
        df["layer"],
        xerr=df["std_dev"],
        fmt="none",
        ecolor="gray",
        # zorder=0,
        lw=0.5,
    )
    plt.tight_layout()

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}__Qtilde_vs_layer_3tf_1mil.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)

    # Get the top N nodes with the highest Q-tilde as motif candidates
    top_Qs = df.sort_values("p_success", ascending=False).head(n_motifs)
    motifs = top_Qs["node"].tolist()
    motif_Qs = top_Qs["p_success"].tolist()

    grammar = SimpleNetworkGrammar(
        components=["A", "B", "C"], interactions=["activates", "inhibits"]
    )
    network_kwargs = _network_kwargs | (network_kwargs or dict())
    for i, (motif, q) in enumerate(zip(motifs, motif_Qs)):
        fig = plt.figure(figsize=network_figsize)
        plt.title(f"Q={q:.3f}", size=10)
        plot_network(
            *grammar.parse_genotype(motif, nonterminal_ok=True), **network_kwargs
        )
        plt.xlim(-1.7, 1.7)
        plt.ylim(-1.2, 2.0)

        if save:
            today = datetime.today().strftime("%y%m%d")
            fpath = Path(save_dir).joinpath(f"{today}_3tf_1mil_motif_{i+1}.{fmt}")
            print(f"Writing to: {fpath.resolve().absolute()}")
            plt.savefig(fpath, dpi=dpi)

    # Plot the best motif at each layer
    best_motifs_in_layer_df = df.loc[df.groupby("layer")["p_success"].idxmax()]
    best_motifs_in_layer = best_motifs_in_layer_df["node"].tolist()
    best_motif_Qs = best_motifs_in_layer_df["p_success"].tolist()
    for i, (motif, q) in enumerate(zip(best_motifs_in_layer, best_motif_Qs)):
        fig = plt.figure(figsize=network_figsize)
        plt.title(f"Q={q:.3f}", size=10)
        plot_network(
            *grammar.parse_genotype(motif, nonterminal_ok=True), **network_kwargs
        )
        plt.xlim(-1.7, 1.7)
        plt.ylim(-1.2, 2.0)

        if save:
            today = datetime.today().strftime("%y%m%d")
            fpath = Path(save_dir).joinpath(f"{today}_3tf_1mil_{i}_edges.{fmt}")
            print(f"Writing to: {fpath.resolve().absolute()}")
            plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    # 5 million MCTS iterations - all 12 replicates
    graph_gml = Path(
        "data/oscillation/mcts/bootstrap_long_230928_000832/231006_merged_tree_iter5000000.gml"
    )
    graph_attrs_json = graph_gml.parent.joinpath(
        "0/oscillation_mcts_bootstrap_5000000_tree.json"
    )

    # graph_gml = list(
    #     Path(
    #         "data/oscillation/mcts/bootstrap_long_230928_000832",
    #     ).glob(
    #         "*/oscillation_mcts_bootstrap_5000000_tree.gml",
    #     )
    # )
    # graph_attrs_json = graph_gml[0].parent.joinpath(
    #     "oscillation_mcts_bootstrap_5000000_tree.json"
    # )

    save_dir = Path("figures/oscillation")

    main(
        graph_gml,
        attrs_json=graph_attrs_json,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
