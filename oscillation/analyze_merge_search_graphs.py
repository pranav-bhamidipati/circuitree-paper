from datetime import datetime
import networkx as nx
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
from graph_utils import merge_search_graphs


def main(
    graph_gmls: Iterable[Path],
    save_dir: Path,
    compress: bool = False,
    progress: bool = False,
    processes: int = 1,
):
    gmls = [Path(gml) for gml in graph_gmls]
    if processes == 1:
        print(f"Reading {len(gmls)} search graphs...")
        if progress:
            search_graphs = [nx.read_gml(gml) for gml in tqdm(gmls)]
        else:
            search_graphs = [nx.read_gml(gml) for gml in gmls]
    else:
        from multiprocessing import Pool

        if progress:
            pbar = tqdm(total=len(gmls), desc="Reading search graphs")

        search_graphs = []
        with Pool(processes) as pool:
            for graph in pool.imap_unordered(nx.read_gml, gmls):
                search_graphs.append(graph)
                if progress:
                    pbar.update(1)

    print(f"Merging {len(gmls)} search graphs...")
    G: nx.DiGraph = merge_search_graphs(search_graphs)
    print(f"Done.")

    today = datetime.now().strftime("%y%m%d")
    fpath = Path(save_dir) / f"{today}_merged_search_graph.gml"
    if compress:
        fpath = fpath.with_suffix(".gml.gz")
    print(f"Writing to: {fpath.resolve().absolute()}")
    nx.write_gml(G, fpath)


if __name__ == "__main__":
    data_dir = Path("data/oscillation/mcts/mcts_bootstrap_long_231022_173227")
    # graph_gmls = data_dir.glob("*/*_100000_tree.gml")
    graph_gmls = data_dir.glob("*/*_5000000_tree.gml")
    save_dir = data_dir
    main(
        graph_gmls=graph_gmls,
        save_dir=save_dir,
        compress=True,
        progress=True,
        processes=2,
    )
