from datetime import datetime
from itertools import chain, islice
from typing import Optional
from circuitree import SimpleNetworkGrammar
import pandas as pd
from oscillation import OscillationTree
from pathlib import Path
import networkx as nx

import json


def main(
    search_graph_gml: Path,
    search_graph_json: Path,
    to_depth: int,
    sample_size: int = 100_000,
    significance_threshold: int = 0.05,
    confidence_level: float = 0.95,
    sampling_method: str = "enumeration",
    nprocs: int = 1,
    nprocs_testing: int = 1,
    null_chunksize: Optional[int] = None,
    succ_chunksize: Optional[int] = None,
    max_iter: Optional[int] = None,
    barnard_ok: bool = True,
    save: bool = False,
    save_dir: Path = None,
    progress: bool = False,
):
    print(
        "Loading search graph from file. This can take many minutes for large graphs..."
    )
    tree = OscillationTree.from_file(
        Path(search_graph_gml),
        Path(search_graph_json),
        grammar_cls=SimpleNetworkGrammar,
    )

    print("Identifying successful oscillators...")
    successful_circuits = set(
        n
        for n in tree.graph.nodes
        if tree.grammar.is_terminal(n) and tree.is_success(n)
    )
    print(f"# oscillators found: {len(successful_circuits)}")

    # Enumerate all patterns up to a certain depth, keeping any that are oscillators
    layers = islice(nx.bfs_layers(tree.graph, tree.root), to_depth + 1)
    pattern_complexity = {}
    for depth, layer in enumerate(layers):
        for state in layer:
            if tree.grammar.is_terminal(state) or state == tree.root:
                continue
            terminal_state = tree._do_action(state, "*terminate*")
            if terminal_state in successful_circuits:
                pattern = state.split("::")[1]
                pattern_complexity[pattern] = depth
    patterns = list(pattern_complexity.keys())

    print(
        f"Computing frequencies for {len(patterns)} patterns up to depth {to_depth}..."
    )

    null_kwargs = {}
    succ_kwargs = {}
    max_iter_kw = {}
    if null_chunksize is not None:
        null_kwargs["chunksize"] = null_chunksize
    if succ_chunksize is not None:
        succ_kwargs["chunksize"] = succ_chunksize
    if max_iter is not None:
        max_iter_kw["max_iter"] = max_iter

    df: pd.DataFrame = tree.test_pattern_significance(
        patterns,
        n_samples=sample_size,
        progress=progress,
        sampling_method=sampling_method,
        nprocs_sampling=nprocs,
        nprocs_testing=nprocs_testing,
        confidence=confidence_level,
        null_kwargs=null_kwargs,
        succ_kwargs=succ_kwargs,
        barnard_ok=barnard_ok,
        **max_iter_kw,
    )
    df["complexity"] = df["pattern"].map(pattern_complexity)

    # Find the patterns that are sufficient for oscillation and significantly
    # overrepresented (motifs)
    df["sufficient"] = True
    df["significant"] = df["p_corrected"] < significance_threshold
    df["overrepresented"] = df["odds_ratio"] > 1.0

    if save:
        save_dir = Path(save_dir)
        today = datetime.now().strftime("%y%m%d%H%M%S")
        fname = save_dir / f"{today}_circuit_pattern_tests_depth{to_depth}.csv"
        print(f"Writing to: {fname.resolve().absolute()}")
        df.to_csv(fname)

    else:
        return df


if __name__ == "__main__":
    # graph_path = Path(
    #     # "data/oscillation/mcts/mcts_bootstrap_short_231020_175449/231022_merged_search_graph"
    #     "data/oscillation/mcts/mcts_bootstrap_long_231022_173227/231026_merged_search_graph"
    # )
    # graph_gml = graph_path.with_suffix(".gml.gz")
    # graph_json = graph_path.with_suffix(".json")

    graph_dir = Path(
        # "data/aws_exhaustion_exploration2.00"
        "data/oscillation/mcts"
        # "/231104-19-32-24_5tf_exhaustion_mutationrate0.5_batch1_max_interactions15_exploration2.000"
        "/231114-06-48-21_5tf_exhaustion_mutationrate0.5_batch1_max_interactions15_exploration2.000"
        "/backups"
    )
    graph_gml = graph_dir.joinpath(
        # "tree-28047823-dd31-4723-9dc1-f00ae6545013_2023-11-07_02-00-36.gml.gz"
        "tree-42e2e9e5-db08-4d22-bd50-81e96e36c5a4_2023-11-15_13-44-19.gml.gz"
    )
    graph_json = graph_dir.joinpath(
        # "tree-28047823-dd31-4723-9dc1-f00ae6545013_2023-11-04_12-32-24.json"
        "tree-42e2e9e5-db08-4d22-bd50-81e96e36c5a4_2023-11-13_23-48-21.json"
    )

    save_dir = graph_dir.parent.joinpath("analysis")
    save_dir.mkdir(exist_ok=True)

    main(
        search_graph_gml=graph_gml,
        search_graph_json=graph_json,
        to_depth=9,
        # to_depth=12,
        # sample_size=500,
        sample_size=100_000,
        # sample_size=500_000,
        sampling_method="rejection",
        nprocs=186,
        nprocs_testing=186,
        null_chunksize=100,
        succ_chunksize=1,
        max_iter=10_000_000_000,
        # null_chunksize=100,
        # succ_chunksize=20,
        progress=True,
        barnard_ok=False,
        save=True,
        save_dir=save_dir,
    )
