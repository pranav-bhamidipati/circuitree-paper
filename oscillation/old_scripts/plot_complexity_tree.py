from itertools import chain, islice
from circuitree.viz import plot_network
from circuitree import CircuitGrammar
from datetime import date, datetime
from functools import partial
import json
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import networkx as nx
import numpy as np
from pathlib import Path
import pandas as pd
from seaborn.palettes import color_palette
from typing import Any, Callable, Iterable, Literal, Mapping, Optional

from tqdm import tqdm

from graph_utils import (
    compute_Q_tilde,
    merge_search_graphs,
    _get_mean_reward,
    make_complexity_tree_mst,
    complexity_layout,
    n_connected_components_in_circuit,
    prune_complexity_tree,
)


_edge_kwargs = dict(
    width=0.5,
    edge_color="black",
    alpha=1.0,
    edge_cmap=None,
)

_node_kwargs = dict(
    node_shape="o",
    node_size=2.0,
    # node_size=5,
    node_color="dimgray",
    edgecolors="dimgray",
    alpha=1.0,
    cmap=None,
)


_network_kwargs = dict(
    padding=0.5,
    lw=1,
    node_shrink=0.7,
    offset=0.8,
    auto_shrink=0.9,
    width=0.005,
    plot_labels=False,
)


def _color_by_log_Q_tilde(n: Any, tree: nx.DiGraph, cmap: str = "viridis", norm=None):
    log10_Q_tilde = np.log10(tree.nodes[n]["Q_tilde"])
    return to_rgba(plt.get_cmap(cmap)(norm(log10_Q_tilde)))


_node_ranks = dict()


def _color_by_rank(n: Any, tree: nx.DiGraph, cmap: Optional[str] = None):
    if not _node_ranks:
        terminal_nodes = []
        key = lambda n: _get_mean_reward(tree, n)
        terminal_nodes = sorted(
            (n for n in tree.nodes if tree.out_degree(n) == 0), key=key, reverse=True
        )
        for i, n in enumerate(terminal_nodes):
            _node_ranks[n] = i
    rank = _node_ranks[n]
    return to_rgba(color_palette(cmap, n_colors=len(_node_ranks))[rank])


def plot_complexity_tree(
    complexity_tree: nx.DiGraph,
    pos: Mapping[Any, tuple[float, float]] = None,
    min_depth: int = 0,
    edge_kwargs: Optional[Mapping[str, Any]] = None,
    node_kwargs: Optional[Mapping[str, Any]] = None,
    reverse_x: bool = False,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    fig_kwargs: Optional[Mapping[str, Any]] = None,
    plot_Q: bool = True,
    terminal_size: float = 1.0,
    color_func: Optional[Callable] = None,
    guide_position: Literal["left", "right"] = "right",
):
    if pos is None:
        pos = complexity_layout(complexity_tree)

    if reverse_x:
        pos = {k: (-x, y) for k, (x, y) in pos.items()}
    x_min = min(x for x, y in pos.values())
    x_max = max(x for x, y in pos.values())

    # Plotting options
    edge_kwargs = _edge_kwargs | (edge_kwargs or {})
    node_kwargs = _node_kwargs | (node_kwargs or {})

    # Create a figure and axis if none are provided
    if fig is None:
        fig_kwargs = fig_kwargs or {}
        fig, ax = plt.subplots(**fig_kwargs)
    elif ax is None:
        ax = fig.gca()

    if plot_Q:
        n_nodes = len(complexity_tree.nodes)
        kw = dict(
            node_size=[node_kwargs["node_size"]] * n_nodes,
            node_color=[node_kwargs["node_color"]] * n_nodes,
        )
        for i, n in enumerate(complexity_tree.nodes):
            # if complexity_tree.out_degree(n) > 0:
            #     continue
            kw["node_size"][i] = terminal_size
            if color_func is not None:
                kw["node_color"][i] = color_func(n, complexity_tree)

        node_kwargs = node_kwargs | kw

    nx.draw_networkx_edges(
        complexity_tree.to_undirected(as_view=True), pos=pos, ax=ax, **edge_kwargs
    )
    nx.draw_networkx_nodes(complexity_tree, pos=pos, ax=ax, **node_kwargs)

    plt.axis("off")

    # Plot a guide for the number of interactions in each level in the tree
    if guide_position == "right":
        x_depth = x_max + 0.05 * (x_max - x_min)
    elif guide_position == "left":
        x_depth = x_min - 0.05 * (x_max - x_min)
    else:
        raise ValueError(f"Invalid position for guide labels: {guide_position}")

    level_yvals = sorted(set(y for x, y in pos.values()))[::-1]
    dy = np.diff(-np.array(level_yvals)).min()
    plt.text(x_depth, level_yvals[0] + dy, "# actions", ha="center", va="top")
    for i, y in enumerate(level_yvals):
        plt.text(x_depth, y, i + min_depth, ha="center", va="center")

    # # Plot a colormap for the mean reward at each node
    # if colorbar:
    #     axins = inset_axes(
    #         ax,
    #         width="15%",
    #         height="5%",
    #         loc="upper right",
    #     )
    #     norm = plt.Normalize(vmin=0.0, vmax=vmax)
    #     cbar = plt.colorbar(
    #         ScalarMappable(norm, cmap=cmap),
    #         # ax=ax,
    #         cax=axins,
    #         shrink=0.2,
    #         orientation="horizontal",
    #         location="top",
    #         anchor=(0.8, 0.0),
    #         ticks=[0, round(vmax / 2, 1), 2 * round(vmax / 2, 1)],
    #     )
    #     cbar.ax.set_xlabel(r"$Q$")
    #     cbar.ax.tick_params(labelsize=8)

    return (fig, ax), pos


def main(
    search_graph_gml: Path | Iterable[Path],
    search_graph_json: Path,
    grammar: CircuitGrammar,
    success_cutoff: float = 0.01,
    n_best: int = 100,
    reverse_x: bool = False,
    plot_n_motifs: int = 5,
    edge_kwargs: Optional[Mapping[str, Any]] = None,
    node_kwargs: Optional[Mapping[str, Any]] = None,
    network_kwargs: Optional[Mapping[str, Any]] = None,
    plot_Q: bool = True,
    terminal_size: float = 25.0,
    color_func: Optional[Callable] = None,
    color_by_rank: bool = True,
    rank_cmap: str = "viridis_r",
    figsize: tuple = (10, 4),
    save_dir: Optional[Path] = None,
    save: bool = False,
    fmt: str = "png",
    dpi: int = 300,
    **kwargs,
):
    edge_kwargs = edge_kwargs or {}
    node_kwargs = node_kwargs or {}

    search_graph_json = Path(search_graph_json)
    with search_graph_json.open() as f:
        json_dict = json.load(f)
        root_node = json_dict["root"]

    # Load the search graph, a DAG with a single root, from one or more GML files
    if isinstance(search_graph_gml, Path):
        G: nx.DiGraph = nx.read_gml(search_graph_gml)
    else:
        print(f"Merging {len(search_graph_gml)} search graphs...")
        search_graphs = [nx.read_gml(gml) for gml in search_graph_gml]
        G: nx.DiGraph = merge_search_graphs(search_graphs)
        print(f"Done. Merged graph has {len(G.nodes)} nodes.")

    # Check that the root node is valid
    if G.in_degree(root_node) != 0:
        raise ValueError("The specified root node is invalid.")

    # ##### for testing
    # # Use the LCB loss as a weight for the shortest paths
    # for e in G.edges:
    #     G.edges[e]["lcb_loss"] = lcb_loss(*e, G.edges[e], G)
    # weight_func = "lcb_loss"

    # Build the complexity tree, which is the union of the shortest paths from the root
    # to all states with mean reward above the cutoff

    # complexity_tree, tree_paths = make_complexity_tree_dijkstra(
    complexity_tree, terminal_states = make_complexity_tree_mst(
        G,
        root_node=root_node,
        grammar=grammar,
        success_cutoff=success_cutoff,
        n_best=n_best,
    )
    print(f"Complexity tree has {len(terminal_states)} terminal states.")

    # Color terminal nodes by their rank
    if color_by_rank:
        # color_func = partial(_color_by_rank, cmap=rank_cmap)
        Qs = np.array(list(nx.get_node_attributes(complexity_tree, "Q_tilde").values()))
        log10Qs = np.log10(Qs)
        norm = mpl.colors.Normalize(vmin=min(log10Qs), vmax=max(log10Qs))
        color_func = partial(_color_by_log_Q_tilde, cmap=rank_cmap, norm=norm)
    elif color_func is None:
        node_kwargs["node_color"] = "gray"
        color_func = lambda n, tree: "black"

    fig, ax = plt.subplots(figsize=figsize)
    complexity_kwargs = dict()
    _, pos = plot_complexity_tree(
        complexity_tree=complexity_tree,
        fig=fig,
        ax=ax,
        edge_kwargs=edge_kwargs,
        node_kwargs=node_kwargs,
        reverse_x=reverse_x,
        plot_Q=plot_Q,
        terminal_size=terminal_size,
        color_func=color_func,
        **complexity_kwargs,
        **kwargs,
    )
    xylim = ax.axis()

    ############################################################

    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_complexity_tree.{fmt}"
        fpath = Path(save_dir).joinpath(fname)
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)

    ############################################################

    # Compute p-values on each non-terminal edge
    # For each parent node
    #   Compute theta_parent = reward(parent) / visits(parent)
    #   For each child node, compute
    #     n = reward(node)
    #     N = visits(node)
    #     theta_child = n / N
    #     p_value = P(n, N | theta_parent) = 1 - binom.cdf(n, N, theta_parent)

    from scipy.stats import binom

    significance_level = 0.05
    bonferroni_factor = sum(1 for _, child in G.edges if not grammar.is_terminal(child))
    log_bonferroni_factor = np.log(bonferroni_factor)
    log_significance_level = np.log(significance_level)

    ############################################################

    # Bayesian approaches over all assembly states!!

    # Attempt 2: Compare the posterior distributions between reachable and non-reachable sets
    from graph_utils import bayesian_p_value_of_motif

    sample_size = 10_000
    largest_motif = 6  # the largest motif to consider, in number of interactions

    terminal_set = set(n for n in G.nodes if grammar.is_terminal(n))
    layers_with_motifs = list(nx.bfs_layers(G, root_node))[1 : largest_motif + 1]
    motif_candidates = list(chain.from_iterable(layers_with_motifs))
    p_motif = {}
    iterator = tqdm(motif_candidates, desc="Computing p-values")
    for state in iterator:
        if state in terminal_set or state == root_node:
            continue
        ######## NEED TO CHECK THAT THIS IS WORKING AS INTENDED - FOUND NO MOTIFS
        p_val = bayesian_p_value_of_motif(
            G, grammar=grammar, state=state, U=terminal_set, sample_size=sample_size
        )
        p_motif[state] = p_val

    n_significant = sum(p < significance_level for p in p_motif.values())
    print(f"Found {n_significant} significant motifs.")

    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_motif_p_values.csv"
        fpath = Path(save_dir).joinpath(fname)
        print(f"Writing to: {fpath.resolve().absolute()}")
        pd.Series(p_motif).to_csv(fpath)

    ...

    0 / 0

    ############################################################

    # # Attempt 1: Compute the posterior distribution of Q for each node
    # # For each circuit, compute the posterior distribution of the mean reward Q
    # # Then compute the p-value, which is the Bonferroni-corrected probability
    # #           P(Q > Q_threshold) * correction_factor
    # # For a terminated circuits, we use a uniform Beta prior (1, 1), which is conjugate
    # # to the binomial likelihood.
    # # For non-terminated (incomplete) circuits, we take an analogous approach to other
    # # motif-finding algorithms, where we consider the set of all possible terminated
    # # circuits that contain this motif. The posterior of the incomplete circuit is the
    # # mean of the posteriors in this set.
    # # Result: only the successful terminal states (and parents where those states were
    # # inevitable) were significant. This is because the requirement for the set of all
    # # reachable states to be net successful is too stringent.

    # from graph_utils import Q_posterior
    # from tqdm import tqdm
    # from scipy.special import logsumexp

    # significant_nodes = []
    # n_nodes = G.number_of_nodes()
    # posterior_log_cdf = {}
    # for node in tqdm(
    #     nx.dfs_postorder_nodes(G, root_node), desc="Computing p-values", total=n_nodes
    # ):
    #     ### CDF evaluated at Q_threshold is the posterior probability that Q lies below
    #     ### Q_threshold. As with other Bayesian approaches, we do not apply a correction
    #     ### for multiple hypothesis testing.
    #     if grammar.is_terminal(node):
    #         # Compute the log-CDF of the posterior distribution of Q
    #         posterior_dist = Q_posterior(
    #             G.nodes[node]["reward"], G.nodes[node]["visits"]
    #         )
    #         log_cdf = posterior_dist.logcdf(Q_threshold)
    #     else:
    #         # Compute the log(mean CDF) over the posteriors of circuits containing this motif
    #         descendant_log_cdfs = []
    #         for descendant in nx.dfs_postorder_nodes(G, node):
    #             if grammar.is_terminal(descendant):
    #                 descendant_log_cdfs.append(posterior_log_cdf[descendant])

    #         # Compute Log(Mean of CDFs) as Log(Sum of CDFs) - Log(N)
    #         log_n_descendants = np.log(len(descendant_log_cdfs))
    #         log_cdf = logsumexp(descendant_log_cdfs) - log_n_descendants

    #     posterior_log_cdf[node] = log_cdf
    #     log_p_val = log_cdf
    #     G.nodes[node]["log_p_val"] = log_cdf

    #     if log_p_val < log_significance_level:
    #         significant_nodes.append(node)

    # print(f"Found {len(significant_nodes)} significant nodes.")

    ...

    ############################################################

    # Frequentist approach over edges - for each (parent, child) edge, compares each
    # child's sampling history to the parent's unbiased reward (Q_tilde) as a null
    # model. This finds moves that are significantly exploited by MCTS, but it does not
    # really give us motifs. It is more a reflection of how sampling is done.

    # # Compute Q_tilde for each node
    # compute_Q_tilde(G, root_node, inplace=True, in_edges=False)

    # # Convert the signficance level to a threshold we can apply to the log(CDF)
    # significant_edges = []
    # for parent, child in G.edges:
    #     # if grammar.is_terminal(child):
    #     #     continue
    #     theta_parent = G.nodes[parent]["Q_tilde"]
    #     n = G.nodes[child]["reward"]
    #     N = G.nodes[child]["visits"]
    #     log_p_uncorrected = binom.logsf(n, N, theta_parent)
    #     log_p_bonferroni = log_p_uncorrected + log_bonferroni_factor
    #     if log_p_bonferroni < log_significance_level:
    #         significant_edges.append((parent, child))
    #     G.edges[parent, child]["log_p_uncorrected"] = log_p_uncorrected
    #     G.edges[parent, child]["log_p_bonferroni"] = log_p_bonferroni

    # # Make complexity tree from significant edges
    # pval_graph = G.edge_subgraph(significant_edges).copy()
    # print(
    #     f"Found {len(pval_graph.nodes)} significant nodes and {len(significant_edges)} significant edges."
    # )

    # # Remove individual circuits with more than one connected components
    # invalid_circuits = []
    # for node in pval_graph.nodes:
    #     ncomp = n_connected_components_in_circuit(node, grammar=grammar)
    #     if ncomp > 1:
    #         invalid_circuits.append(node)
    # print(f"Removing {len(invalid_circuits)} invalid circuits.")
    # pval_graph.remove_nodes_from(invalid_circuits)

    # # Remove nodes that are not reachable from the root
    # unreachable_nodes = set(pval_graph.nodes)
    # for layer in nx.bfs_layers(pval_graph, root_node):
    #     unreachable_nodes -= set(layer)
    # print(f"Removing {len(unreachable_nodes)} unreachable nodes.")
    # pval_graph.remove_nodes_from(unreachable_nodes)

    ############################################################

    # # To compute a minimum spanning arborescence, we weight each edge by the loss
    # # function Loss(parent -> child) = (1 - Q_tilde(parent)) * (1 - Q_tilde(child))
    # for n1, n2 in pval_graph.edges:
    #     pval_graph.edges[n1, n2]["Q_loss"] = (1 - pval_graph.nodes[n1]["Q_tilde"]) * (
    #         1 - pval_graph.nodes[n2]["Q_tilde"]
    #     )

    # pval_tree = nx.minimum_spanning_arborescence(
    #     pval_graph, attr="Q_loss", preserve_attrs=True
    # )

    # # Copy node attributes from the original graph
    # nx.set_node_attributes(complexity_tree, pval_graph.nodes)

    # pval_terminal_states = [n for n in pval_tree.nodes if grammar.is_terminal(n)]
    # print(
    #     f"Complexity tree from p-values has {len(pval_terminal_states)} terminal states."
    # )

    ...

    # # Trim the tree to show only the best (top X) terminal states
    # best_states = prune_complexity_tree(
    #     pval_tree,
    #     root_node=root_node,
    #     grammar=grammar,
    #     n_best=n_best,
    #     success_cutoff=success_cutoff,
    # )

    ############################################################

    0 / 0

    # Plot the complexity graph induced by the edges that are significant
    graph = pval_tree
    # graph = pval_graph

    pos2 = complexity_layout(graph)
    fig, ax = plt.subplots(figsize=figsize)
    plot_complexity_tree(
        complexity_tree=graph,
        pos=pos2,
        fig=fig,
        ax=ax,
        edge_kwargs=edge_kwargs,
        node_kwargs=node_kwargs | dict(node_size=20.0),
        plot_Q=False,
        # reverse_x=True,
        # terminal_size=terminal_size,
        # color_func=color_func,
        **kwargs,
    )
    # nx.draw_networkx_labels(graph, pos2, ax=ax, font_size=8)

    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_complexity_graph_from_pvals_bayesian.{fmt}"
        fpath = Path(save_dir).joinpath(fname)
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)

    # Plot smaller subtree with labels so you can see which nodes are which
    branches = sorted(pval_tree.successors(root_node), key=lambda n: pos2[n][0])
    for i, branch in enumerate(branches):
        subtree = nx.dfs_tree(pval_tree, branch)

        if subtree.number_of_nodes() == 1:
            print(f"Found singleton branch {i}: {branch}")
            continue

        subtree_pos = complexity_layout(subtree)
        fig, ax = plt.subplots(figsize=figsize)
        plot_complexity_tree(
            complexity_tree=subtree,
            pos=subtree_pos,
            fig=fig,
            ax=ax,
            edge_kwargs=edge_kwargs,
            node_kwargs=node_kwargs | dict(node_size=20.0),
            plot_Q=False,
            # reverse_x=True,
            # terminal_size=terminal_size,
            # color_func=color_func,
            **kwargs,
        )
        nx.draw_networkx_labels(subtree, subtree_pos, ax=ax, font_size=8)

        if save:
            today = date.today().strftime("%y%m%d")
            fname = f"{today}_complexity_tree_branch_{i}.{fmt}"
            fpath = Path(save_dir).joinpath(fname)
            print(f"Writing to: {fpath.resolve().absolute()}")
            plt.savefig(fpath, dpi=dpi)

    ############################################################

    # ### trying to just use reward and visits
    # # Compute mean reward on each node's sampling history
    # for node in pval_graph.nodes:
    #     mean_reward = pval_graph.nodes[node]["reward"] / max(
    #         1, pval_graph.nodes[node]["visits"]
    #     )
    #     pval_graph.nodes[node]["mean_reward"] = mean_reward

    # # To compute a minimum spanning arborescence, we weight each edge by the loss
    # # function Loss(parent -> child) = (1 - Q_tilde(parent)) * (1 - Q_tilde(child))
    # for n1, n2 in pval_graph.edges:
    #     pval_graph.edges[n1, n2]["Q_loss"] = (1 - pval_graph.nodes[n1]["Q_tilde"]) * (
    #         1 - pval_graph.nodes[n2]["Q_tilde"]
    #     )

    # pval_tree = nx.minimum_spanning_arborescence(
    #     pval_graph, attr="Q_loss", preserve_attrs=True
    # )
    # pval_terminal_states = [n for n in pval_tree.nodes if grammar.is_terminal(n)]
    # print(
    #     f"Complexity tree from p-values has {len(pval_terminal_states)} terminal states."
    # )

    ############################################################

    # # Find motifs in the complexity tree using pagerank algorithm. Ignore terminal nodes
    # ranking = nx.pagerank(complexity_tree, weight="Q_tilde")
    # ranked_motifs = sorted(ranking, key=ranking.get, reverse=True)
    # ranked_motifs = [m for m in ranked_motifs if complexity_tree.out_degree(m) > 0]
    # best_motifs = ranked_motifs[:plot_n_motifs]

    # # Plot a dashed black circle around the best pageranked nodes
    # dx = (xylim[1] - xylim[0]) * 0.01
    # dy = (xylim[3] - xylim[2]) * 0.025
    # for i, m in enumerate(best_motifs):
    #     plt.scatter(*pos[m], s=130, c="none", edgecolors="black", linewidths=0.5)

    #     # Add a label with the motif ranking
    #     plt.text(
    #         pos[m][0] + dx,
    #         pos[m][1] + dy,
    #         f"#{i + 1}",
    #         ha="left",
    #         va="bottom",
    #         fontsize=10,
    #     )
    # ax.axis(xylim)

    # if save:
    #     today = date.today().strftime("%y%m%d")
    #     fname = f"{today}_complexity_tree_annotated.{fmt}"
    #     fpath = Path(save_dir).joinpath(fname)
    #     print(f"Writing to: {fpath.resolve().absolute()}")
    #     plt.savefig(fpath, dpi=dpi)

    # # Print the best motifs and their circuit string representations
    # print()
    # print("Motif candidates:")
    # print("  Rank  |  Q_tilde  |  Circuit ")
    # print("--------+--------+--------------")
    # for i, m in enumerate(best_motifs):
    #     Q_tilde = complexity_tree.nodes[m]["Q_tilde"]
    #     print(f"  {i + 1:2d}    |  {Q_tilde:.3f}  |  {m} ")

    # print()
    # print("Plotting motif candidates.")

    # # Plot network diagram for each motif
    # network_kwargs = _network_kwargs | (network_kwargs or dict())
    # for i, m in enumerate(best_motifs):
    #     fig = plt.figure(figsize=figsize)
    #     plt.title(f"Motif #{i + 1}", size=10)
    #     plot_network(*grammar.parse_genotype(m, nonterminal_ok=True), **network_kwargs)
    #     plt.xlim(-1.7, 1.7)
    #     plt.ylim(-1.2, 2.0)

    #     if save:
    #         today = datetime.today().strftime("%y%m%d")
    #         fpath = Path(save_dir).joinpath(
    #             f"{today}_complexity_tree_motif_{i+1}.{fmt}"
    #         )
    #         print(f"Writing to: {fpath.resolve().absolute()}")
    #         plt.savefig(fpath, dpi=dpi)

    ############################################################

    # # Plot a green arrow pointing to the best pageranked nodes
    # dx = (xylim[1] - xylim[0]) * 0.02
    # dy = (xylim[3] - xylim[2]) * 0.05
    # for rank, n in enumerate(best_ranked):
    #     plt.annotate(
    #         f"#{rank + 1}: {n}",
    #         xy=pos[n],
    #         xytext=(pos[n][0] + dx, pos[n][1] - dy),
    #         arrowprops=dict(facecolor="green", shrink=0.05),
    #         fontsize=12,
    #     )

    # # Print specific nodes with labels
    # best_branches = sorted(
    #     (n for n in complexity_tree.nodes if complexity_tree.out_degree(n) >= 4),
    #     key=complexity_tree.out_degree,
    #     reverse=True,
    # )
    # terminal_nodes = sorted(
    #     tree_paths, key=lambda n: _get_mean_reward(complexity_tree, n), reverse=True
    # )

    # AI_nodes = [n for n in complexity_tree.nodes if grammar.has_motif(n, "ABa_BAi")]
    # AAI_nodes = [
    #     n for n in complexity_tree.nodes if grammar.has_motif(n, "ABa_BCa_CAi")
    # ]
    # III_nodes = [
    #     n for n in complexity_tree.nodes if grammar.has_motif(n, "ABi_BCi_CAi")
    # ]

    # motifs = {
    #     "AIpar": "ABC::AAa_ABa_BAi",
    #     "III": "ABC::ABi_BCi_CAi",
    #     "AIpar+III": "ABC::AAa_ABa_ACi_BAi_CBi",
    # }
    # motif_nodes = {}
    # for name, motif in motifs.items():
    #     motif_nodes[name] = [
    #         n for n in complexity_tree.nodes if grammar.has_motif(n, motif)
    #     ]

    # fig, ax = plt.subplots(figsize=figsize)
    # plt.scatter(*zip(*pos.values()), s=2, c="gray")
    # plt.scatter(*zip(*[pos[n] for n in AI_nodes]), s=50, c="lightsalmon")
    # plt.scatter(*zip(*[pos[n] for n in AAI_nodes]), s=30, c="lightgreen")
    # plt.scatter(*zip(*[pos[n] for n in III_nodes]), s=10, c="lightblue")
    # # plt.scatter(*zip(*[pos[n] for n in best_branches]), s=10, c="red")
    # plt.scatter(*zip(*[pos[n] for n in terminal_nodes]), s=2, c="blue")

    # # Plot a circle around important motifs, with a legend for each
    # for name, motif in motifs.items():
    #     if motif in complexity_tree.nodes:
    #         plt.scatter(*pos[motif], s=100, c="none", edgecolors="black")
    #         plt.text(*pos[motif], name, ha="left", va="bottom", fontsize=10)
    #     else:
    #         print(f"Motif {motif} not found in complexity tree.")


if __name__ == "__main__":
    from circuitree.models import SimpleNetworkGrammar

    # # 100k MCTS iterations
    # search_graph_gml = Path(
    #     "data/oscillation/mcts/mcts_bootstrap_short_231002_221756/2"
    #     "/oscillation_mcts_bootstrap_100000_tree.gml"
    # )
    # graph_attrs_json = search_graph_gml.parent.joinpath(
    #     "oscillation_mcts_bootstrap_1000_tree.json"
    # )

    # # 5 million MCTS iterations - single replicate (replicate # 0)
    # search_graph_gml = Path(
    #     "data/oscillation/mcts/bootstrap_long_230928_000832"
    #     "/0/oscillation_mcts_bootstrap_5000000_tree.gml"
    # )
    # graph_attrs_json = search_graph_gml.parent.joinpath(
    #     "oscillation_mcts_bootstrap_5000000_tree.json"
    # )

    # 5 million MCTS iterations - all 12 replicates
    search_graph_gml = Path(
        "data/oscillation/mcts/bootstrap_long_230928_000832"
        "/231006_merged_tree_iter5000000.gml"
    )
    graph_attrs_json = search_graph_gml.parent.joinpath(
        "0/oscillation_mcts_bootstrap_5000000_tree.json"
    )
    save_dir = Path("figures/oscillation")

    grammar = SimpleNetworkGrammar(
        components=["A", "B", "C"], interactions=["activates", "inhibits"]
    )

    main(
        search_graph_gml=search_graph_gml,
        search_graph_json=graph_attrs_json,
        grammar=grammar,
        reverse_x=False,
        save_dir=save_dir,
        save=True,
        fmt="pdf",
        n_best=100,
        plot_n_motifs=0,
        rank_cmap="viridis",
    )
