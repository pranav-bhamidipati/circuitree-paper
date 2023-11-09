from graph_utils import get_minimum_spanning_arborescence
from datetime import datetime
from pathlib import Path
import json
from typing import Iterable, Optional
import networkx as nx
import matplotlib.pyplot as plt

from circuitree import SimpleNetworkGrammar
import numpy as np
import pandas as pd
from oscillation import OscillationTree

_edge_kwargs = dict(
    edge_color="darkgray",
    width=0.5,
    arrows=False,
    # arrowsize=0,
    node_size=0,
    # connectionstyle="bar,angle=180,fraction=-0.2",
)


def main(
    data_dir: Path,
    attrs_json: Path,
    results_csv: Path,
    plot_steps: Iterable[int],
    Qthresh: float = 0.01,
    gml_glob_str: str = "*_{}_tree.gml",
    figsize: tuple = (4, 4),
    suffix: str = "",
    save_dir: Optional[Path] = None,
    save: bool = True,
    fmt: str = "png",
    dpi: int = 300,
):
    # Get which states are oscillators
    results_df = pd.read_csv(results_csv, index_col=0)
    oscillators = set(results_df.loc[results_df["p_oscillation"] >= Qthresh, "state"])

    gmls = []
    for step in plot_steps:
        gml = next(data_dir.glob(gml_glob_str.format(step)))
        gmls.append(gml)

    with attrs_json.open("r") as f:
        attrs = json.load(f)

    # Account for a possible bug in the saved json file where the
    # _non_serializable_attrs key is not removed before saving
    attrs["grammar"].pop("_non_serializable_attrs", None)
    # write back to file
    with attrs_json.open("w") as f:
        json.dump(attrs, f)

    today = datetime.now().strftime("%y%m%d")

    min_trees = []
    node_positions = []
    for step, gml in zip(plot_steps, gmls):
        tree = OscillationTree.from_file(
            gml, attrs_json, grammar_cls=SimpleNetworkGrammar
        )
        for e in tree.graph.edges:
            tree.graph.edges[e]["w"] = np.exp(-tree.graph.edges[e]["visits"])
        msa = get_minimum_spanning_arborescence(
            tree,
            weight="w",
        )

        pos = nx.nx_agraph.graphviz_layout(msa, prog="dot")
        node_positions.append(pos)
        min_trees.append(msa)

    # Align x and y value ranges based on depth
    ref_positions = node_positions[-1]
    xlim = [
        min(x for x, _ in ref_positions.values()),
        max(x for x, _ in ref_positions.values()),
    ]
    ylim = [
        min(y for _, y in ref_positions.values()),
        max(y for _, y in ref_positions.values()),
    ]
    ny = len(set(y for _, y in ref_positions.values()))
    if ny < 2:
        raise ValueError(
            "The number of y values is less than 2. Cannot align plot dimensions."
        )
    dy = (ylim[1] - ylim[0]) / (ny - 1)

    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    xlim = [xlim[0] - 0.1 * xrange, xlim[1] + 0.1 * xrange]
    ylim = [ylim[0] - 0.1 * yrange, ylim[1] + 0.1 * yrange]

    for _pos in node_positions:
        xyvals = np.array(list(_pos.values()))
        xvals, yvals = xyvals.T

        # Y values are set based on depth
        unique_ys, depth = np.unique(-yvals, return_inverse=True)
        yvals = ylim[1] - depth * dy

        # Rescale x values to be between xlim[0] and xlim[1]
        xval_max = xvals.max()
        xval_min = xvals.min()
        xcenter = np.mean(xlim)
        if xval_max == xval_min:
            xvals = np.ones_like(xvals) * xcenter
        else:
            xvals = xlim[0] + xrange * (xvals - xval_min) / (xval_max - xval_min)

        # Center x values within a layer
        for d, _ in enumerate(unique_ys):
            mask = depth == d
            xvals_d = xvals[mask]
            shift = (xvals_d.max() + xvals_d.min()) / 2 - xcenter
            xvals[mask] -= shift

        _pos.update(dict(zip(_pos.keys(), zip(xvals, yvals))))

    for step, _pos, _msa in zip(plot_steps, node_positions, min_trees):
        fig, ax = plt.subplots(figsize=figsize)
        edges = nx.draw_networkx_edges(_msa, _pos, ax=ax, **_edge_kwargs)
        edges.set_zorder(0)
        # for edge in edges:
        #     edge.set_zorder(1)

        # Highlight oscillators
        oscillator_xys = np.array([_pos[node] for node in oscillators if node in _pos])
        if oscillator_xys.size > 0:
            plt.scatter(*oscillator_xys.T, color="tab:orange", s=7, zorder=5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_axis_off()

        if save:
            fpath = Path(save_dir).joinpath(f"{today}_dendrogram_{step}{suffix}.{fmt}")
            print(f"Writing to: {fpath.resolve().absolute()}")
            plt.savefig(fpath, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    results_csv = Path("data/oscillation/231102_exhaustive_results.csv")

    data_dir = Path(
        "data/oscillation/mcts/mcts_bootstrap_short_exploration2.00_231103_140501/0"
    )
    attrs_json = next(data_dir.glob("*.json"))
    save_dir = Path("figures/oscillation/dendrograms")
    save_dir.mkdir(exist_ok=True)

    plot_steps = [1_000, 10_000, 50_000, 100_000]
    # plot_steps = [1_000, 100_000]

    main(
        results_csv=results_csv,
        data_dir=data_dir,
        attrs_json=attrs_json,
        plot_steps=plot_steps,
        save_dir=save_dir,
        save=True,
        suffix="_3tf_short",
        fmt="pdf",
    )
