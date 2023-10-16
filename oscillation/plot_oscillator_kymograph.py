from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.sparse import csr_matrix


def get_depth(state: str) -> int:
    _, interactions_joined = state.split("::")
    if interactions_joined:
        return interactions_joined.count("_") + 1
    else:
        return 0


def get_Qval(state: str, graph: nx.DiGraph) -> float:
    if state in graph.nodes:
        return graph.nodes[state]["reward"] / max(1, graph.nodes[state]["visits"])
    else:
        return 0.0


def main(
    data_dir: Path,
    figsize: tuple[float, float] = (5, 5),
    save: bool = False,
    save_dir: Optional[Path] = None,
    root_node: str = "ABC::",
    fmt: str = "png",
    dpi: int = 300,
):
    # Load in the visited states
    #   - they are stored in txt files, one per chunk
    #   - sort the txt files by extracting the step number
    #   - store the max step number
    #   - concat to make a long dataframe with columns for step #, name, and chunk #
    #   - aggregate by chunk #, making a new column for the fraction of samples for each terminal

    # Load sampling data
    csv_to_step = {}
    for csv in Path(data_dir).glob("*samples.csv"):
        step = int(csv.name.split("_")[-2])
        csv_to_step[csv] = step
    csvs = sorted(csv_to_step, key=csv_to_step.get)

    # Load search graph
    last_step = max(csv_to_step.values())
    gml = next(data_dir.glob(f"*{last_step}_tree.gml"))
    search_graph = nx.read_gml(gml)

    # Build a dataframe of states and their number of samples in each chunk of time-steps
    dfs = []
    for i, (csv, step) in enumerate(csv_to_step.items()):
        sample_counter = Counter(pd.read_csv(csv, names=["state"])["state"].tolist())
        # n_steps = sample_counter.total()
        df = pd.DataFrame(
            {
                "state": sample_counter.keys(),
                "n_samples": sample_counter.values(),
                "step": step,
                "chunk": i,
            }
        )
        dfs.append(df)
    data = pd.concat(dfs)

    # Extract the name, depth, and Q value of terminal nodes from the search graph
    #   - Get the search graph at the last step number
    #   - Load the graph with nx.read_gml
    #   - Iterate over nodes with enumerate(bfs_layers) and store the name, depth, and Q
    #   - Make the state column a pd.Categorical ordered by depth, then Q
    unique_states = data["state"].unique()
    depths = {}
    Qvals = {}
    for state in unique_states:
        Qval = get_Qval(state, search_graph)
        if Qval < 0.01:
            continue
        depth = get_depth(state)
        depths[state] = depth
        Qvals[state] = Qval

    data = data.loc[data["state"].isin(Qvals)]

    data["depth"] = data["state"].map(depths)
    data["Q"] = data["state"].map(Qvals)

    argsort_states = np.lexsort(
        [
            -np.array(list(depths.values()), dtype=int),
            np.array(list(Qvals.values()), dtype=float),
        ]
    )
    state_ordering = dict(zip(depths.keys(), argsort_states))
    data["order"] = data["state"].map(state_ordering)
    data = data.sort_values(["chunk", "order"])

    samples_2d = csr_matrix(
        (data["n_samples"].values, (data["order"].values, data["chunk"].values))
    ).toarray()

    # plot_data = sample_data.pivot(index="step", columns="state", values="n_samples")

    # Plot a kymograph of sampling
    #   - x-axis is sampling time, y-axis is terminal node
    #   - cmap is the fraction of samples going to that node
    fig = plt.figure(figsize=figsize)

    hmap = sns.heatmap(data=samples_2d, vmin=0.0)

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_sampling_kymograph.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)

    ...


if __name__ == "__main__":
    data_dir = Path("data/oscillation/mcts/mcts_bootstrap_short_231013_175157/0")
    save_dir = Path("figures/oscillation")

    main(
        data_dir=data_dir,
        save=True,
        save_dir=save_dir,
    )
