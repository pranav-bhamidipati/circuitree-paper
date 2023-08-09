from collections import defaultdict
from functools import partial
import h5py
from pathlib import Path
from typing import Iterable
import numpy as np

import pandas as pd
from tqdm import tqdm


def consolidate_hdfs(
    files: Iterable[Path],
    param_table: pd.DataFrame,
    out_file: Path,
    param_names: Iterable[str] = [
        "k_on",
        "k_off_1",
        "k_off_2",
        "km_unbound",
        "km_act",
        "km_rep",
        "km_act_rep",
        "kp",
        "gamma_m",
        "gamma_p",
    ],
):
    """Consolidate HDF5 files into one HDF5 file."""
    metadata_dfs = []
    y_ts = []
    for f in files:
        f_metadata = pd.read_hdf(f, key="metadata")
        sample_numbers = []
        for i, row in f_metadata.iterrows():
            # Search for the first sample number that matches this parameter set.
            # Each parameter in each parameter set should be distinct because they were
            # sampled using a Latin hypercube. However, due to floating point precision,
            # we may not find the desired match on the first try.
            for param_name in param_names:
                param_val = row[param_name]
                sample_num = np.where(param_table[param_name] == param_val)[0]
                found_sample = len(sample_num) == 1
                if found_sample:
                    break
            if not found_sample:
                raise ValueError(
                    f"Could not find sample number for {param_name}={param_val}"
                )

            sample_numbers.append(sample_num)
        f_metadata["sample_number"] = sample_numbers
        metadata_dfs.append(f_metadata[["sample_number", "seed", "state", "reward"]])

        with h5py.File(f, "r") as f_in:
            y_ts.append(f_in["y_t"][...])

    metadata = pd.concat(metadata_dfs, ignore_index=True)
    y_t = np.concatenate(y_ts, axis=0)

    metadata.to_hdf(out_file, key="metadata", mode="w", format="table")
    with h5py.File(out_file, "a") as f_out:
        f_out.create_dataset("y_t", data=y_t)


def run_one_genotype(genotype_and_files, param_table):
    genotype_file, input_files = genotype_and_files
    consolidate_hdfs(input_files, param_table, genotype_file)


def main(
    data_dir: Path,
    params_csv: Path,
    delim: str = "-",
    progress: bool = True,
    n_workers: int = 1,
):
    """Consolidate files with oscillation data into one HDF5 file per genotype."""

    param_table = pd.read_csv(params_csv)

    file_to_genotype = {f: f.stem.split(delim)[0] for f in data_dir.glob("*.hdf5")}
    genotype_to_files = defaultdict(list)
    for f, g in file_to_genotype.items():
        gfile = data_dir.joinpath(f"{g}.hdf5")
        genotype_to_files[gfile].append(f)

    iterator = genotype_to_files.items()
    if progress:
        pbar = tqdm(total=len(genotype_to_files))

    run_one = partial(run_one_genotype, param_table=param_table)

    if n_workers == 1:
        for args in iterator:
            run_one(args)
            if progress:
                pbar.update(1)
    else:
        from multiprocessing import Pool

        with Pool(n_workers) as pool:
            for _ in pool.imap_unordered(run_one, iterator):
                if progress:
                    pbar.update(1)


if __name__ == "__main__":
    data_dir = Path("data/oscillation/bfs_230710_hpc/extras")
    params_csv = Path("data/oscillation/param_sets_queue_10000.csv")

    main(
        data_dir=data_dir,
        params_csv=params_csv,
        # n_workers=14,
    )
