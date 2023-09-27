from datetime import date
from typing import Optional
from circuitree.viz import plot_network
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pathlib import Path

import numpy as np

from oscillation import OscillationTree

_network_kwargs = dict(
    padding=0.5,
    lw=1,
    node_shrink=0.7,
    offset=0.8,
    auto_shrink=0.9,
    width=0.005,
)


def main(
    figsize: tuple = (2.0, 2.0),
    save: bool = False,
    save_dir: Optional[Path] = None,
    fmt: str = "svg",
    dpi: int = 300,
    tree_kwargs: dict = {},
    aspect: float = 1.0,
    scale: float = 8.0,
    **kwargs,
):
    # Plotting parameters
    network_kwargs = _network_kwargs | kwargs

    root = "ABC::ABi"

    # node_successors = {root: [f"{root}_BAa", f"{root}_BCi"], "BAa": [], "BCi": ["CAi", "CAa"]}

    tree = OscillationTree(
        root=root,
        components=["A", "B", "C"],
        interactions=["activates", "inhibits"],
        **tree_kwargs,
    )

    tree.graph.add_edge(root, f"{root}_CBa")
    abi_bci = f"{root}_BCi"
    tree.graph.add_edge(root, abi_bci)
    repressilator = f"{abi_bci}_CAi"
    tree.graph.add_edge(abi_bci, repressilator)
    tree.graph.add_edge(abi_bci, f"{abi_bci}_ACa")

    fig_1a_tree = plt.figure(figsize=figsize)
    pos: dict[str, tuple[float, float]] = graphviz_layout(tree.graph, prog="dot")

    # Shrink xy positions to the size of circuit diagrams
    xmax = scale * aspect
    ymax = scale
    xs, ys = np.array(list(pos.values())).T
    xs = xmax * (xs - xs.min()) / (xs.max() - xs.min())
    ys = ymax * (ys - ys.min()) / (ys.max() - ys.min())
    pos = dict(zip(pos.keys(), zip(xs, ys)))

    # nodes = nx.draw_networkx_nodes(
    #     tree.graph, pos, node_size=5000, node_color="white", edgecolors="w"
    # )
    edges = nx.draw_networkx_edges(
        tree.graph, pos, width=3, edge_color="gray", node_size=1000
    )

    # nodes.set_zorder(0)
    for edge in edges:
        edge.set_zorder(1)

    # nx.draw(
    #     tree.graph,
    #     pos=pos,
    #     with_labels=False,
    #     node_color="white",  # Transparent nodes
    #     edgecolors="w",
    #     node_size=5000,  # very large
    #     edge_color="k",
    #     width=3,
    # )

    for g, (x, y) in pos.items():
        plot_network(
            *tree.parse_genotype(g, nonterminal_ok=True),
            center=(x, y),
            plot_labels=False,
            **network_kwargs,
        )

    plt.tight_layout()

    if save:
        today = date.today().strftime("%y%m%d")
        fname = f"{today}_fig1_topologies.{fmt}"
        fpath = Path(save_dir).joinpath(fname).resolve().absolute()
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi)
    plt.close()

    fig1a_sim_node = plt.figure(figsize=(1.0, 1.0))
    sim_node = f"{repressilator}_BBa_CCi"
    plot_network(
        *tree.parse_genotype(sim_node, nonterminal_ok=True),
        plot_labels=False,
        **network_kwargs,
    )
    plt.tight_layout()

    if save:
        fname = f"{today}_fig1_simulated_circuit.{fmt}"
        fpath = Path(save_dir).joinpath(fname).resolve().absolute()
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi)
    plt.close()


if __name__ == "__main__":
    save_dir = Path("figures/oscillation")
    main(
        save=True,
        save_dir=save_dir,
    )
