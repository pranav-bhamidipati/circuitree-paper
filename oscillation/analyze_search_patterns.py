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
    save: bool = False,
    save_dir: Path = None,
    progress: bool = False,
):
    # Hack around a dumb bug in the original code for grammar objects. Attributes that
    # are not supposed to be serialized to JSON are stored in the object attribute
    # "_non_serializable_attrs". Originally, this attribute itself was accidentally
    # serialized. This is a hack to remove it before loading.
    search_graph_json = Path(search_graph_json)
    with open(search_graph_json, "r") as f:
        attrs = json.load(f)
        attrs["grammar"].pop("_non_serializable_attrs", None)
    search_graph_json = search_graph_json.with_suffix(".json.tmp")
    with open(search_graph_json, "w") as f:
        json.dump(attrs, f)

    tree = OscillationTree.from_file(
        Path(search_graph_gml), search_graph_json, grammar_cls=SimpleNetworkGrammar
    )
    search_graph_json.unlink()

    # Identify successful circuits
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

    print(f"Computing frequencies for {len(patterns)} patterns...")

    null_kwargs = {}
    succ_kwargs = {}
    if null_chunksize is not None:
        null_kwargs["chunksize"] = null_chunksize
    if succ_chunksize is not None:
        succ_kwargs["chunksize"] = succ_chunksize

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
    )
    df["complexity"] = df["pattern"].map(pattern_complexity)

    # Find the patterns that are sufficient for oscillation and significantly
    # overrepresented (motifs)
    df["sufficient"] = True
    df["significant"] = df["p_corrected"] < significance_threshold
    df["overrepresented"] = df["odds_ratio"] > 1.0

    if save:
        save_dir = Path(save_dir)
        today = datetime.now().strftime("%y%m%d")
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
        "data/aws_exhaustion_exploration2.00"
        "/231104-19-32-24_5tf_exhaustion_mutationrate0.5_batch1_max_interactions15_exploration2.000"
        "/backups"
    )
    graph_gml = graph_dir.joinpath(
        "tree-28047823-dd31-4723-9dc1-f00ae6545013_2023-11-06_10-33-54.gml.gz"
    )
    graph_json = graph_dir.joinpath(
        "tree-28047823-dd31-4723-9dc1-f00ae6545013_2023-11-04_12-32-24.json"
    )

    save_dir = graph_dir.parent.joinpath("analysis")
    save_dir.mkdir(exist_ok=True)

    main(
        search_graph_gml=graph_gml,
        search_graph_json=graph_json,
        to_depth=9,
        sample_size=10_000,
        sampling_method="rejection",
        nprocs=13,
        nprocs_testing=13,
        # null_chunksize=2_000,
        # succ_chunksize=2_000,
        null_chunksize=100,
        succ_chunksize=20,
        progress=True,
        save=True,
        save_dir=save_dir,
    )
