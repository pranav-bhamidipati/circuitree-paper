from typing import Iterable, Mapping
import numpy as np
import pandas as pd
from pathlib import Path

from models.oscillation.oscillation import OscillationTree


def main(
    table_loc: Path,
    save_loc: Path,
    motifs: Mapping[str, str],
):
    tree = OscillationTree(
        components=["A", "B", "C"], interactions=["activates", "inhibits"], root="ABC::"
    )

    df = pd.read_parquet(table_loc, columns=["state", "reward"])
    p_oscillation = df.groupby("state")["reward"].agg(lambda s: np.mean(s.abs() > 0.4))
    p_oscillation.name = "p_oscillation"
    
    motif_columns = []
    for name, motif in motifs.items():
        motif_col = df.groupby("state")["state"].agg(lambda s: tree.has_motif(s.iloc[0], motif))
        motif_col.name = name
        motif_columns.append(motif_col)

    agg_df = pd.concat([p_oscillation] + motif_columns, axis=1).reset_index()
    
    target = save_loc.with_suffix(".csv")
    print(f"Writing to: {target}")
    agg_df.to_csv(target)

    ...

if __name__ == "__main__":
    table_loc = Path("data/oscillation/230717_transposition_table_hpc.parquet")
    save_loc = Path("data/oscillation/230717_motifs.csv")
    motifs = {
        "AI": "ABa_BAi",
        "AAI": "ABa_BCa_CAi",
        "III": "ABi_BCi_CAi",
        "toggle": "ABi_BAi",
    }

    main(
        table_loc=table_loc,
        save_loc=save_loc,
        motifs=motifs,
        
    )
