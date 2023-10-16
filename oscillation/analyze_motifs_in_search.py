from datetime import datetime
from pathlib import Path
from more_itertools import powerset
import numpy as np
import pandas as pd
from tqdm import tqdm

from oscillation import OscillationGrammar


def _get_n_motifs_present(
    samples: list[str],
    motif_combination_strings: list[str],
    grammar: OscillationGrammar,
):
    motif_combinations_counter = np.zeros(len(motif_combination_strings), dtype=int)
    mc_index = {mc: i for i, mc in enumerate(motif_combination_strings)}
    for sample in samples:
        motif_set = [m for m in motifs if grammar.has_motif(sample, motifs[m])]
        motif_combination_string = "+".join(motif_set) or "(none)"
        motif_combinations_counter[mc_index[motif_combination_string]] += 1
    return motif_combinations_counter


def main(
    results_dir: Path,
    motifs: dict[str, str],
    save_dir: Path = None,
    save: bool = False,
    components=["A", "B", "C"],
    interactions=["activates", "inhibits"],
    step_globstr="*_samples.csv",
    delim="_",
    which_in_split=-2,
):
    grammar = OscillationGrammar(components=components, interactions=interactions)

    motif_combinations = []
    motif_combination_strings = []
    for ms in powerset(motifs.keys()):
        mc_string = "+".join(ms) or "(none)"
        motif_combinations.append(set(ms))
        motif_combination_strings.append(mc_string)

    results_dir = Path(results_dir)
    replicate_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    replicates = []
    steps = []
    motif_samples = []

    for i, d in enumerate(tqdm(replicate_dirs, desc=f"Loading replicates")):
        step_files = list(d.glob(step_globstr))
        for f in step_files:
            step = int(f.stem.split(delim)[which_in_split])
            replicates.append(i)
            steps.append(step)
            step_samples = pd.read_csv(f, header=None).iloc[:, 0].to_list()
            n_motifs_sampled_at_step = _get_n_motifs_present(
                step_samples, motif_combination_strings, grammar
            )
            motif_samples.append(n_motifs_sampled_at_step)

    motif_samples = np.array(motif_samples)
    df = pd.DataFrame(
        {
            "replicate": replicates,
            "step": steps,
            **{mc: n for mc, n in zip(motif_combination_strings, motif_samples.T)},
        }
    )
    df = df.sort_values(["replicate", "step"]).reset_index(drop=True)

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_motif_counts.csv")
        print(f"Writing to: {fpath.resolve().absolute()}")
        df.to_csv(fpath, index=False)


if __name__ == "__main__":
    results_dir = Path("data/oscillation/mcts/mcts_bootstrap_short_231002_221756")
    save_dir = results_dir

    motifs = {"AI": "ABa_BAi", "AAI": "ABa_BCa_CAi", "III": "ABi_BCi_CAi"}

    main(
        results_dir=results_dir,
        save_dir=save_dir,
        motifs=motifs,
        save=True,
    )
