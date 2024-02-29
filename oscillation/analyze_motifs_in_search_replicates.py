from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from analyze_motifs_in_search_graph import main as analyze_patterns


def main(
    data_dir: Path,
    step: int,
    to_depth: int,
    sample_size: int = 100_000,
    confidence_level: float = 0.95,
    gml_glob_str: str = "*_{}_tree.gml",
    json_glob_str: str = "*_{}_tree.json",
    sampling_method: str = "enumeration",
    nprocs: int = 1,
    nprocs_testing: int = 1,
    save: bool = False,
    save_dir: Path = None,
    progress: bool = False,
):
    dfs = []
    for rep_dir in Path(data_dir).iterdir():
        if not rep_dir.is_dir():
            continue
        search_graph_gml = next(Path(rep_dir).glob(gml_glob_str.format(step)), None)
        search_graph_json = next(Path(rep_dir).glob(json_glob_str.format(step)), None)
        if search_graph_gml is None or search_graph_json is None:
            continue

        print(f"Analyzing circuit pattern frequencies for replicate: {rep_dir.name}")

        rep_df = analyze_patterns(
            search_graph_gml=search_graph_gml,
            search_graph_json=search_graph_json,
            to_depth=to_depth,
            sample_size=sample_size,
            confidence_level=confidence_level,
            sampling_method=sampling_method,
            nprocs=nprocs,
            nprocs_testing=nprocs_testing,
            save=False,
            progress=progress,
        )
        rep_df["replicate"] = rep_dir.name
        dfs.append(rep_df)

    if not dfs:
        raise ValueError(
            f"No matches for the expression '*/{gml_glob_str}' in directory: {data_dir}"
        )
    df = pd.concat(dfs)

    if save:
        today = datetime.now().strftime("%y%m%d")
        fname = Path(save_dir).joinpath(
            f"{today}_circuit_pattern_tests_reps{len(dfs)}_depth{to_depth}.csv"
        )
        print(f"Writing to: {fname.resolve().absolute()}")
        df.to_csv(fname)
    else:
        return df


if __name__ == "__main__":
    # data_dir = Path("data/oscillation/mcts/mcts_bootstrap_long_231022_173227")
    # step = 5_000_000

    # data_dir = Path("data/oscillation/mcts/mcts_bootstrap_short_231020_175449")
    data_dir = Path(
        "data/oscillation/mcts/mcts_bootstrap_short_exploration2.00_231103_140501"
    )
    step = 100_000

    save_dir = data_dir
    main(
        data_dir=data_dir,
        step=step,
        to_depth=9,
        sample_size=10_000,
        nprocs=12,
        nprocs_testing=12,
        progress=True,
        save=True,
        save_dir=save_dir,
    )
