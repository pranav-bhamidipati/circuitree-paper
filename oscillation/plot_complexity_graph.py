from circuitree import depth_from_root
from circuitree.viz import complexity_layout
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Optional

from models.oscillation.oscillation import OscillationTree


def main(
    gml: Path,
    attrs_json: Path,
    reward_threshold: float = 0.01,
    motif: Optional[str] = None,
    motif_color: str = "tab:red",
    figsize: tuple = (8, 5),
    # max_edge_width: int | float = 6,
    root_n_edges: int = 0,
    cmap: str = "viridis",
    colorbar: bool = True,
    vmax: float = 0.8,
    save_dir: Optional[Path] = None,
    save: bool = False,
    fmt: str = "png",
    dpi: int = 300,
):
    gml = Path(gml)
    attrs_json = Path(attrs_json)

    tree = OscillationTree.from_file(gml, attrs_json)
    # C = tree.to_complexity_graph(reward_threshold=reward_threshold)
    # pos = nx.nx_agraph.graphviz_layout(C, prog="dot")

    C, pos = complexity_layout(tree, reward_threshold=reward_threshold)

    # # Normalize edge weights by depth
    # depth = depth_from_root(C, source_tree=tree.graph, root=tree.graph.root)
    # nnodes_at_depth = Counter(depth.values())
    # edge_visits_at_depth = Counter()
    # for e1, e2, visits in C.edges.data("visits"):
    #     edge_visits_at_depth[depth[e1]] += visits

    # I want to visualize this as a network flow problem.
    # The width of each edge is proportional to the number of visits flowing through.
    # Howqever, what's happening is that the edge with the most visits is an edge that
    # is reached from a path *outside* the complexity graph. This means that the flow
    # relationship is broken (i.e. this node is receiving an influx of flow from outside
    # the complexity graph).
    # To make this work I need to reference the source tree and compute the component of
    # edge flow that is a conserved flow through the complexity graph.

    # width_per_visit = {d: total_edge_width / v for d, v in edge_visits_at_depth.items()}

    # edge_weights = []
    # for e1, e2, visits in C.edges.data("visits"):
    #     width = visits * width_per_visit[depth[e1]]
    #     edge_weights.append(width)

    # max_edge_visits = max(v for *e, v in C.edges.data("visits"))
    # width_per_visit = max_edge_width / max_edge_visits
    # edge_weights = [v * width_per_visit for *e, v in C.edges(data="visits")]

    ...

    fig, ax = plt.subplots(figsize=figsize)

    # Add a background shading for a motif of interest
    # if motif:
    #     motif_color = plt.get_cmap(motif_cmap)(np.arange(len(motifs)))

    mean_reward = np.array(
        [d["reward"] / max(d["visits"], 1) for *_, d in C.nodes(data=True)]
    )
    nx.draw_networkx_edges(
        C.to_undirected(as_view=True),
        pos=pos,
        ax=ax,
        alpha=0.5,
        width=0.5,
        # width=edge_widths,
    )
    nx.draw_networkx_nodes(
        C.to_undirected(as_view=True),
        pos=pos,
        ax=ax,
        node_shape="s",
        node_size=10,
        node_color=mean_reward,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )
    plt.axis("off")

    # Plot a guide for the number of interactions in each level in the tree
    level_yvals = sorted(set(y for x, y in pos.values()))
    depth = depth_from_root(C, source_tree=tree.graph, root=tree.graph.root)
    min_depth = root_n_edges + min(depth.values())
    dy = level_yvals[1] - level_yvals[0]
    min_x = min(x for x, y in pos.values())
    max_x = max(x for x, y in pos.values())
    x = min_x - 0.05 * (max_x - min_x)
    plt.text(x, level_yvals[-1] + dy * 0.6, "# interactions", ha="left", va="center")
    for i, y in enumerate(level_yvals[::-1]):
        plt.text(x, y, i + min_depth, ha="center", va="center")

    if motif is not None:
        motif_nodes = [n for n in C.nodes if tree.has_motif(n, motif)]
        motif_pos = np.array([pos[n] for n in motif_nodes])
        plt.scatter(*motif_pos.T, s=40, c=motif_color, alpha=0.5, zorder=0)

    # Plot a colormap for the mean reward at each node
    if colorbar:
        axins = inset_axes(
            ax,
            width="15%",
            height="5%",
            loc="upper right",
        )
        norm = plt.Normalize(vmin=0.0, vmax=vmax)
        cbar = plt.colorbar(
            ScalarMappable(norm, cmap=cmap),
            # ax=ax,
            cax=axins,
            shrink=0.2,
            orientation="horizontal",
            location="top",
            anchor=(0.8, 0.0),
            ticks=[0, round(vmax / 2, 1), 2 * round(vmax / 2, 1)],
        )
        cbar.ax.set_xlabel(r"$Q$")
        cbar.ax.tick_params(labelsize=8)

    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_complexity_graph.{fmt}"
        fpath = Path(save_dir).joinpath(fname)
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    gml_path = Path(
        "data/oscillation/230727_oscillation_aggregated_tree_batchsize10.gml"
    )
    json_path = Path(
        "data/oscillation/230727_oscillation_aggregated_tree_batchsize10.json"
    )
    save_dir = Path("./figures/oscillation")
    main(
        gml=gml_path,
        attrs_json=json_path,
        colorbar=False,
        save_dir=save_dir,
        save=True,
        # motif="ABi_BAi_BCi_CAi",
    )
