# Specify the AI+PAR and AI+PAR+suppressor circuits
# Read in oscillating samples from both topologies, keeping the first pair with the same parameter values
# Plot the circuit diagrams
# Plot the protein dynamics
# Plot the autocorrelation function

import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from circuitree.viz import plot_network
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


def main(
    tranposition_table_parquet: Path,
    data_dir: Path,
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

    ai_par = "*ABC::AAa_ABa_BAi"
    ai_par_supp = "*ABC::AAa_ABa_BAi_CBi"
    genotypes = [ai_par, ai_par_supp]

    print("Loading transposition table...")
    ttable = pd.read_parquet(tranposition_table_parquet).reset_index(drop=True)

    # Isolate oscillating samples
    ttable = ttable.loc[ttable["reward"] < ACF_thresh]

    # Find parameter sets that oscillate in both topologies
    ttable = ttable.loc[ttable["state"].isin([ai_par, ai_par_supp])]
    for param in PARAM_NAMES:
        ttable[param] = ttable[param].astype("category")
    states_per_param_set = ttable.groupby(PARAM_NAMES[0]).size()
    param_vals_in_both = np.array(
        states_per_param_set.loc[states_per_param_set == 2].index
    )
    ttable = ttable.loc[ttable[PARAM_NAMES[0]].isin(param_vals_in_both)]

    # Sort them from best to worst oscillation
    max_acf = ttable.groupby(PARAM_NAMES[0])["reward"].max().sort_values()
    param_vals_ordered = np.array(max_acf.index)
    param_ordering = dict(zip(param_vals_ordered, range(len(param_vals_ordered))))
    ttable["param_order"] = ttable[PARAM_NAMES[0]].map(param_ordering).astype(int)

    ...

    ai_par_hdfs = list(data_dir.glob("state_ABC::AAa_ABa_BAi_samples*.hdf5"))
    ai_par_supp_hdfs = list(data_dir.glob("state_ABC::AAa_ABa_BAi_CBi_samples*.hdf5"))

    # Iterate over parameter values, from best to worst
    for param_val in param_vals_ordered:
        # Try to find the hdf5 files for both topologies
        for ai_par_hdf in ai_par_hdfs:
            metadata: pd.DataFrame = pd.read_hdf(
                ai_par_hdf, key="metadata"
            ).reset_index(drop=True)
            where_matches = np.isclose(metadata[PARAM_NAMES[0]], param_val) & (
                metadata["reward"] < ACF_thresh
            )
            if where_matches.any():
                ai_par_matching_idx = np.where(where_matches)[0][0]
                break
        else:
            # This is an efficient but cursed way to do a nested loop break
            # If the for loop is broken, the else statement is skipped
            # If the for loop is not broken, the else statement is executed
            # So in this case, if the appropriate file was not found, skip
            # to the next parameter value
            continue

        for ai_par_supp_hdf in ai_par_supp_hdfs:
            metadata: pd.DataFrame = pd.read_hdf(
                ai_par_supp_hdf, key="metadata"
            ).reset_index(drop=True)
            where_matches = np.isclose(metadata[PARAM_NAMES[0]], param_val) & (
                metadata["reward"] < ACF_thresh
            )
            if where_matches.any():
                ai_par_supp_matching_idx = np.where(where_matches)[0][0]
                break
        else:
            # Again, efficient but cursed, but efficient
            continue
        # If we found both files, we can stop looking
        break
    else:
        # This else block is executed if the outside for loop is not broken
        raise ValueError(
            f"No hdf5 files found with matching parameter values for the circuits "
            f"{ai_par} and {ai_par_supp}"
        )

    # Load the protein dynamics
    with h5py.File(ai_par_hdf, "r") as f:
        ai_par_y_t = np.array(f["y_t"])[ai_par_matching_idx].T

    with h5py.File(ai_par_supp_hdf, "r") as f:
        ai_par_supp_y_t = np.array(f["y_t"])[ai_par_supp_matching_idx].T

    t = np.arange(ai_par_y_t.shape[1]) * dt_seconds
    y_ts = np.array([ai_par_y_t, ai_par_supp_y_t])

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
            figsize=figsize,
            plot_dir=save_dir,
            save=save,
            dpi=dpi,
            fmt=fmt,
            suffix=f"_{interaction_code}",
            annotate_acf_min=False,
            **_network_kwargs,
        )


if __name__ == "__main__":
    ttable_parquet = Path("data/oscillation/230717_transposition_table_hpc.parquet")
    data_dir = Path("data/oscillation/bfs_230710_hpc/extras")

    save_dir = Path("figures/oscillation/ai_suppressor_comparison")
    save_dir.mkdir(exist_ok=True)

    main(
        tranposition_table_parquet=ttable_parquet,
        data_dir=data_dir,
        save=True,
        save_dir=save_dir,
        fmt="pdf",
    )
