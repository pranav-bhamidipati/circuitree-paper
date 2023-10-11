from datetime import datetime
import h5py
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from gillespie import PARAM_NAMES

from tf_network import TFNetworkModel


def main(
    transposition_table_parquet: Path,
    data_dir: Path,
    ACF_thresh: float = -0.4,
    dt_seconds: float = 20.0,
    save: bool = False,
    save_dir: Path = None,
):
    # Load transposition table, which contains all the input information for each
    #   simulation - parameters, random seeds, initial conditions, states, etc.
    ttable = pd.read_parquet(transposition_table_parquet).reset_index(drop=True)

    hdfs = list(data_dir.glob("*.hdf5"))
    idx_in_transposition_table = []
    frequencies = []
    ACF_mins = []
    for hdf in tqdm(hdfs):
        with h5py.File(str(hdf), "r") as f:
            y_ts = np.array(f["y_t"])
        metadata: pd.DataFrame = pd.read_hdf(hdf, key="metadata").reset_index(drop=True)
        state = metadata["state"].iloc[0]

        # Calculate the frequency of oscillations and the minimum of the ACF
        t = np.arange(y_ts.shape[1]) * dt_seconds
        freqs, Amins = TFNetworkModel(state).get_acf_minima_and_results(t, y_ts)
        oscillating_rows = Amins < ACF_thresh
        if oscillating_rows.sum() == 0:
            continue
        freqs = freqs[oscillating_rows]
        Amins = Amins[oscillating_rows]
        y_ts = y_ts[oscillating_rows]
        metadata = metadata.loc[pd.Series(oscillating_rows), :]

        # Find the matching rows in the transposition table
        match_attrs = ["reward", *PARAM_NAMES, "A_0", "B_0", "C_0"]
        matching_indices = -np.ones(metadata.shape[0], dtype=int)
        state_ttable: pd.DataFrame = ttable.loc[ttable["state"] == state]
        for i, row in enumerate(metadata.itertuples(index=False)):
            mask = pd.Series(True, index=state_ttable.index)
            for attr in match_attrs:
                val = getattr(row, attr)
                if isinstance(val, float):
                    mask = mask & np.isclose(state_ttable[attr], val)
                else:
                    mask = mask & (state_ttable[attr] == val)
                if mask.sum() == 1:
                    break
            else:
                if mask.any():
                    raise ValueError(
                        "Multiple matching rows found in transposition table"
                    )
                else:
                    raise ValueError("No matching row found in transposition table")
            matching_indices[i] = mask.loc[mask].index.item()

        # Append results
        idx_in_transposition_table.append(matching_indices)
        frequencies.append(freqs)
        ACF_mins.append(Amins)
        ...

    idx_in_transposition_table = np.concatenate(idx_in_transposition_table)
    frequencies = np.concatenate(frequencies)
    ACF_mins = np.concatenate(ACF_mins)
    data = pd.DataFrame(
        {
            "state": ttable.loc[idx_in_transposition_table, "state"],
            "index_in_transposition_table": idx_in_transposition_table,
            "frequency_per_min": 60.0 * frequencies,
            "ACF_min": ACF_mins,
        }
    )

    if save:
        today = datetime.today().strftime("%y%m%d")
        fpath = Path(save_dir).joinpath(f"{today}_oscillator_frequencies.csv")
        print(f"Writing to: {fpath.resolve().absolute()}")
        data.to_csv(fpath, index=False)


if __name__ == "__main__":
    ttable_parquet = Path("data/oscillation/230717_transposition_table_hpc.parquet")
    data_dir = Path("data/oscillation/bfs_230710_hpc/extras")

    save_dir = data_dir.parent

    main(
        transposition_table_parquet=ttable_parquet,
        data_dir=data_dir,
        save=True,
        save_dir=save_dir,
    )
