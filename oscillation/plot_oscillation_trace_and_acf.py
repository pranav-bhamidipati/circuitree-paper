from circuitree import SimpleNetworkGrammar
from typing import Iterable
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from gillespie import PARAM_NAMES
from plot_acf_extrema_single import plot_network_quantity_and_acorr
from tf_network import autocorrelate, binomial9_kernel, compute_lowest_minimum

_network_kwargs = dict(
    fontsize=6,
    padding=0.5,
    lw=1,
    node_shrink=0.7,
    offset=0.8,
    auto_shrink=0.9,
    width=0.005,
    plot_labels=False,
)


def plot_traces_and_acfs(
    genotypes: list[str],
    y_ts: Iterable[np.ndarray],
    t: np.ndarray,
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
    figsize: tuple[float, float] = (3.0, 3.5),
):
    # Generate plots
    for genotype, y_t in zip(genotypes, y_ts):
        # Filter the data with a 9-point binomial filter, then compute autocorrelation
        filtered9 = np.apply_along_axis(binomial9_kernel, -1, y_t)[:, 4:-4]
        acorrs_f9 = np.apply_along_axis(autocorrelate, -1, filtered9)

        t_f9 = t[4:-4]
        all_minima_idx, all_minima = zip(
            *[compute_lowest_minimum(a) for a in acorrs_f9]
        )
        which_lowest = np.argmin(all_minima)
        where_minimum = all_minima_idx[which_lowest]
        minimum_val = all_minima[which_lowest]
        corr_time = t_f9[where_minimum]

        interaction_code = genotype.split("::")[1]
        plot_network_quantity_and_acorr(
            genotype=genotype,
            t=t_f9,
            y_t=filtered9,
            acorr=acorrs_f9,
            corr_time=corr_time,
            minimum_val=minimum_val,
            plot_dir=save_dir,
            figsize=figsize,
            save=save,
            dpi=dpi,
            fmt=fmt,
            suffix=f"_{interaction_code}",
            annotate_acf_min=False,
            **_network_kwargs,
        )


def plot_matching_samples_from_hdf5(
    genotypes: list[str],
    hdf: Path,
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
    figsize: tuple[float, float] = (3.0, 3.5),
):
    y_ts = []
    with h5py.File(hdf, "r") as f:
        t = np.array(f["t_secs"][...])
        for genotype in genotypes:
            components, *_ = SimpleNetworkGrammar.parse_genotype(genotype)
            n_components = len(components)
            y_t = np.array(f[genotype])

            # Isolate TF species
            y_ts.append(y_t[:, n_components : 2 * n_components].T)

    plot_traces_and_acfs(
        genotypes=genotypes,
        y_ts=y_ts,
        t=t,
        save=save,
        save_dir=save_dir,
        fmt=fmt,
        dpi=dpi,
        figsize=figsize,
    )


def plot_matching_samples_from_ttable_data(
    transposition_table_parquet: Path,
    data_dir: Path,
    genotypes: list[str],
    ACF_thresh: float = -0.4,
    dt_seconds: float = 20.0,
    figsize: tuple[float, float] = (3.0, 3.5),
    save: bool = False,
    save_dir: Path = None,
    fmt: str = "png",
    dpi: int = 300,
):
    import warnings

    warnings.filterwarnings(action="ignore", category=FutureWarning)

    print("Loading transposition table...")
    param_name = PARAM_NAMES[0]
    ttable = pd.read_parquet(transposition_table_parquet).reset_index(drop=True)

    # Isolate oscillating samples
    ttable = ttable.loc[ttable["reward"] < ACF_thresh]

    # Find parameter sets that oscillate in both topologies
    ttable = ttable.loc[ttable["state"].isin(genotypes)]
    for param in PARAM_NAMES:
        ttable[param] = ttable[param].astype("category")
    states_per_param_set = ttable.groupby(param_name).size()
    if states_per_param_set.max() < len(genotypes):
        repr_genotypes = ", ".join(genotypes)
        raise ValueError(
            f"None of the parameter sets in the transposition table generated "
            f"oscillations for all circuits: {repr_genotypes}"
        )
    param_vals_in_all = np.array(
        states_per_param_set.loc[states_per_param_set == len(genotypes)].index
    )
    ttable = ttable.loc[ttable[param_name].isin(param_vals_in_all)]

    # Drop unused param sets
    ttable[param_name] = ttable[param_name].cat.remove_unused_categories()

    # Sort them from best to worst oscillation
    best_acf_by_param_val = ttable.groupby(param_name)["reward"].max().sort_values()
    param_vals_ordered = np.array(best_acf_by_param_val.index)
    param_ordering = dict(zip(param_vals_ordered, range(len(param_vals_ordered))))
    ttable["param_order"] = ttable[param_name].map(param_ordering).astype(int)

    # Get the raw data files for each genotype
    print("Finding data for runs with matching parameters...")
    hdfs_per_genotype = [
        list(data_dir.glob(f"state_{g}_samples*.hdf5")) for g in genotypes
    ]

    # Iterate over parameter values, from best to worst
    found_matches = False
    matching_hdfs = []
    matching_indices = []
    for param_val in tqdm(param_vals_ordered):
        # Try to find the hdf5 files for all topologies with this parameter value
        found_match = False
        for g_hdfs in hdfs_per_genotype:
            found_match = False
            for g_hdf in g_hdfs:
                metadata: pd.DataFrame = pd.read_hdf(g_hdf, key="metadata").reset_index(
                    drop=True
                )
                where_matches = np.isclose(metadata[param_name], param_val) & (
                    metadata["reward"] < ACF_thresh
                )
                if where_matches.any():
                    matching_hdfs.append(g_hdf)
                    matching_indices.append(np.where(where_matches)[0][0])
                    found_match = True
                    break
            if not found_match:
                break
        if found_match:
            found_matches = True
            break

    if not found_matches:
        repr_genotypes = ", ".join(genotypes)
        raise ValueError(
            f"No hdf5 files found with matching parameter values for the circuits: "
            f"{repr_genotypes}"
        )

    param_set = ttable.loc[ttable["param_order"] == param_ordering[param_val]].iloc[0]
    print()
    print("Using the parameter set:")
    for param in PARAM_NAMES:
        print(f"\t{param}: {param_set[param]:.3e}")
    print()

    # Load the protein dynamics
    y_ts = []
    for hdf, idx in zip(matching_hdfs, matching_indices):
        with h5py.File(hdf, "r") as f:
            y_t = np.array(f["y_t"])[idx].T
        y_ts.append(y_t)
    t = np.arange(y_ts[0].shape[-1]) * dt_seconds

    # Generate plots
    plot_traces_and_acfs(
        genotypes=genotypes,
        y_ts=y_ts,
        t=t,
        save=save,
        save_dir=save_dir,
        fmt=fmt,
        dpi=dpi,
        figsize=figsize,
    )


if __name__ == "__main__":
    pass
