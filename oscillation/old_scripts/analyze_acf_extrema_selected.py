from typing import Optional
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm


def main(
    genotype: [str],
    data_dir: Path,
    save_dir: Path,
    n_top: Optional[int] = None,
    t: Optional[np.ndarray] = np.arange(0, 40000, 20.0),
    save: bool = True,
    key: Optional[str] = None,
):
    g_nonterm = genotype.lstrip("*")
    run_glob = f"state_{g_nonterm}_samples*.hdf5"

    hdfs = list(Path(data_dir).glob(run_glob))
    if not hdfs:
        raise ValueError(
            f"No HDF5 files found in {data_dir} matching pattern {run_glob}"
        )
    with h5py.File(hdfs[0], "r") as f:
        _, nt, n_species = f["y_t"][...].shape

    if n_top is None:
        n_top = len(hdfs)
    n_top = min(n_top, len(hdfs))

    dfs = []
    for h in tqdm(hdfs, desc="Loading data"):
        _df = pd.read_hdf(h, key=key)
        _df["index_in_file"] = np.arange(len(_df))
        _df["file"] = h.name
        dfs.append(_df)
    df: pd.DataFrame = pd.concat(dfs)

    # Pick the best oscillating runs
    top_runs = df.sort_values("reward").head(n_top)
    top_run_data = np.zeros((n_top, n_species, nt), dtype=np.float64)
    for i, (fpath, idx) in enumerate(top_runs[["file", "index_in_file"]].values):
        h = data_dir.joinpath(fpath).resolve().absolute()
        with h5py.File(h, "r") as f:
            top_run_data[i] = f["y_t"][idx].T

    if save:
        fname = f"top_oscillating_runs_{g_nonterm}.hdf5"
        fpath = save_dir.joinpath(fname).resolve().absolute()
        print(f"Writing to: {fpath}")
        with h5py.File(fpath, "w") as f:
            f.create_dataset("t", data=t)
            f.create_dataset("y_t", data=top_run_data)
            f.create_dataset("states", data=top_runs.state.tolist())
        top_runs.to_hdf(fpath, key="metadata", mode="a")


if __name__ == "__main__":
    data_dir = Path("data/oscillation/bfs_230710_hpc/extras")
    save_dir = Path("data/oscillation/bfs_230710_hpc")

    main(
        genotype="*ABC::AAa_ABa_BAi_CBi",
        data_dir=data_dir,
        save_dir=save_dir,
    )
