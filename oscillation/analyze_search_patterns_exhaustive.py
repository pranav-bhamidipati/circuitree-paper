from datetime import datetime
from itertools import chain, islice
from circuitree import SimpleNetworkGrammar
import pandas as pd
from oscillation import OscillationTree
from pathlib import Path
import networkx as nx

import json


def main(
    results_csv: Path,
    config_json: Path,
    to_depth: int,
    sample_size: int = 100_000,
    sampling_method: str = "enumeration",
    significance_threshold: int = 0.05,
    confidence_level: float = 0.95,
    nprocs: int = 1,
    nprocs_testing: int = 1,
    save: bool = False,
    save_dir: Path = None,
    progress: bool = False,
):
    tree = OscillationTree.from_file(
        None, config_json, grammar_cls=SimpleNetworkGrammar
    )

    # Load the results of an exhaustive search and Identify successful circuits
    results_df = pd.read_csv(results_csv)
    Qs = dict(results_df[["state", "p_oscillation"]].values)
    successful_circuits = set(n for n, q in Qs.items() if q >= tree.Q_threshold)
    print(f"# oscillators found: {len(successful_circuits)}")

    # Fill the tree with the results of the exhaustive search
    tree.grow_tree()
    for n in tree.graph.nodes:
        tree.graph.nodes[n]["visits"] = 1
        if n in successful_circuits:
            tree.graph.nodes[n]["reward"] = 1.0
        else:
            tree.graph.nodes[n]["reward"] = 0.0

    # Find all patterns that are sufficient for oscillation
    layers = islice(nx.bfs_layers(tree.graph, tree.root), to_depth + 1)
    pattern_complexity = {}
    terminal_map = {}
    for depth, layer in enumerate(layers):
        for state in layer:
            if tree.grammar.is_terminal(state) or state == tree.root:
                continue
            terminal_state = tree._do_action(state, "*terminate*")
            if terminal_state in successful_circuits:
                pattern = state.split("::")[1]
                pattern_complexity[pattern] = depth
                terminal_map[pattern] = terminal_state
    patterns = list(pattern_complexity.keys())

    ...

    print(
        f"Computing frequencies for {len(patterns)} patterns up to depth {to_depth}..."
    )

    df: pd.DataFrame = tree.test_pattern_significance(
        patterns,
        n_samples=sample_size,
        progress=progress,
        sampling_method=sampling_method,
        nprocs_sampling=nprocs,
        nprocs_testing=nprocs_testing,
        confidence=confidence_level,
    )
    df["complexity"] = df["pattern"].map(pattern_complexity)

    # Store the p_oscillation column of the results df in the pattern df
    df["Q"] = df["pattern"].map(terminal_map).map(Qs)

    # Find the patterns that are sufficient for oscillation and significantly
    # overrepresented (motifs)
    df["sufficient"] = True
    df["significant"] = df["p_corrected"] < significance_threshold
    df["overrepresented"] = df["odds_ratio"] > 1.0

    if save:
        today = datetime.now().strftime("%y%m%d")
        fname = Path(save_dir).joinpath(
            f"{today}_circuit_pattern_tests_exhaustive_search_depth{to_depth}.csv"
        )
        print(f"Writing to: {fname.resolve().absolute()}")
        df.to_csv(fname)

    else:
        return df


if __name__ == "__main__":
    results_csv = Path("data/oscillation/230717_motifs.csv")
    config_json = Path(
        "data/oscillation/mcts/mcts_bootstrap_long_231022_173227/231026_merged_search_graph.json"
    )
    save_dir = Path("data/oscillation/")
    main(
        results_csv=results_csv,
        config_json=config_json,
        to_depth=9,
        sample_size=100_000,
        sampling_method="enumeration",
        nprocs=13,
        nprocs_testing=13,
        progress=True,
        save=True,
        save_dir=save_dir,
    )
