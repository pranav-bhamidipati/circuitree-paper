from datetime import datetime
from functools import cache, partial
from pathlib import Path
from more_itertools import powerset
import numpy as np
import pandas as pd
from tqdm import tqdm

from circuitree.models import SimpleNetworkGrammar


def _get_which_motif_combo_sampled(
    sample: str,
    motif_combination_strings: list[str],
    grammar: SimpleNetworkGrammar,
):
    motif_set = [m for m in motifs if grammar.has_pattern(sample, motifs[m])]
    motif_combination_string = "+".join(motif_set) or "(none)"

    motif_combinations_counter = np.zeros(len(motif_combination_strings), dtype=bool)
    for i, mc in enumerate(motif_combination_strings):
        if mc == motif_combination_string:
            motif_combinations_counter[i] = True
            return motif_combinations_counter


def _get_n_motifs_present(
    samples: list[str],
    motif_combination_strings: list[str],
    grammar: SimpleNetworkGrammar,
):
    motif_combinations_counter = np.zeros(len(motif_combination_strings), dtype=int)
    mc_index = {mc: i for i, mc in enumerate(motif_combination_strings)}
    for sample in samples:
        motif_set = [m for m in motifs if grammar.has_pattern(sample, motifs[m])]
        motif_combination_string = "+".join(motif_set) or "(none)"
        motif_combinations_counter[mc_index[motif_combination_string]] += 1
    return motif_combinations_counter


def _load_from_replicate_dir(
    replicate_idx_and_dir: tuple[int, Path],
    step_globstr: str,
    delim: str,
    which_in_split: int,
    motif_combination_strings: list[str],
    grammar: SimpleNetworkGrammar,
):
    @cache
    def get_motif_combination(sample):
        motif_set = [m for m in motifs if grammar.has_pattern(sample, motifs[m])]
        return "+".join(motif_set) or "(none)"

    replicate_idx, replicate_dir = replicate_idx_and_dir
    step_files = list(replicate_dir.glob(step_globstr))
    steps = []
    motif_samples = []
    for f in step_files:
        step = int(f.stem.split(delim)[which_in_split])
        sample_df = pd.read_csv(f)
        steps.extend(range(step - len(sample_df), step))
        if "selected_node" in sample_df.columns:
            # Get the selected node at each step
            step_samples = sample_df["selected_node"].values
        else:
            # Remove header and read the first column as a list of selected nodes
            step_samples = sample_df.iloc[1:, 0].to_list()

        motif_samples.extend(map(get_motif_combination, step_samples))

        # n_motifs_sampled_at_step = _get_n_motifs_present(
        #     step_samples, motif_combination_strings, grammar
        # )
        # motif_samples.append(n_motifs_sampled_at_step)

    # motif_samples = np.array(motif_samples)

    replicate_df = pd.DataFrame(
        {
            "replicate": replicate_idx,
            "step": steps,
            "motifs": motif_samples,
            # **{mc: n for mc, n in zip(motif_combination_strings, motif_samples.T)},
        }
    )
    return replicate_df


def main(
    results_dir: Path,
    motifs: dict[str, str],
    nprocs: int = 1,
    save_dir: Path = None,
    save: bool = False,
    components=["A", "B", "C"],
    interactions=["activates", "inhibits"],
    step_globstr="*_samples.csv",
    delim="_",
    which_in_split=-2,
):
    grammar = SimpleNetworkGrammar(components=components, interactions=interactions)

    motif_combinations = []
    motif_combination_strings = []
    for ms in powerset(motifs.keys()):
        mc_string = "+".join(ms) or "(none)"
        motif_combinations.append(set(ms))
        motif_combination_strings.append(mc_string)

    results_dir = Path(results_dir)
    replicate_dirs = sorted(
        (d for d in results_dir.iterdir() if d.is_dir()), key=lambda x: int(x.name)
    )
    load_one_replicate = partial(
        _load_from_replicate_dir,
        step_globstr=step_globstr,
        delim=delim,
        which_in_split=which_in_split,
        motif_combination_strings=motif_combination_strings,
        grammar=grammar,
    )

    dfs = []
    if nprocs == 1:
        for i, d in enumerate(tqdm(replicate_dirs, desc=f"Loading replicates")):
            replicate_df = load_one_replicate((i, d))
            dfs.append(replicate_df)
    else:
        from multiprocessing import Pool

        pbar = tqdm(total=len(replicate_dirs), desc=f"Loading replicates")
        with Pool(nprocs) as p:
            for replicate_df in p.imap_unordered(
                load_one_replicate, enumerate(replicate_dirs)
            ):
                pbar.update()
                dfs.append(replicate_df)

    df = pd.concat(dfs)
    df = df.sort_values(["replicate", "step"]).reset_index(drop=True)

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_motif_counts.csv")
        print(f"Writing to: {fpath.resolve().absolute()}")
        df.to_csv(fpath, index=False)


if __name__ == "__main__":
    results_dir = Path(
        "data/oscillation/mcts/mcts_bootstrap_1mil_iters_exploration2.00_231204_142301"
    )

    save_dir = results_dir

    motifs = {"AI": "ABa_BAi", "III": "ABi_BCi_CAi"}

    main(
        results_dir=results_dir,
        save_dir=save_dir,
        motifs=motifs,
        # nprocs=12,
        save=True,
    )
