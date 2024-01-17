from typing import Optional
from circuitree import SimpleNetworkGrammar
from datetime import datetime
import h5py
from multiprocessing import Pool
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

from graph_utils import simplenetwork_adjacency_matrix, simplenetwork_n_interactions


def edge_substitution_cost(edge1_data, edge2_data):
    """Return the absolute difference in edge weights as the cost of substituting
    edge1 with edge2."""
    return np.abs(edge1_data["weight"] - edge2_data["weight"])


def compute_distance(args):
    """Compute the graph edit distance between two graphs."""
    i, j, adj_a, adj_b = args
    a = nx.from_numpy_array(adj_a, create_using=nx.DiGraph)
    b = nx.from_numpy_array(adj_b, create_using=nx.DiGraph)
    return (
        i,
        j,
        nx.similarity.graph_edit_distance(a, b, edge_subst_cost=edge_substitution_cost),
    )


def main(
    data_dir: Path,
    param_sets_csv: Path,
    n_procs: int = 1,
    Q_thresh: float = 0.01,
    min_visits: int = 1,
    max_interactions: Optional[int] = None,
    save: bool = False,
    save_dir: Path = None,
    suffix: str = "",
):
    """Compute the pairwise graph edit distance between all sampled oscillators
    and save the distance matrix to an HDF5 file.

    Parameters
    ----------
    data_dir : Path
        Path to directory containing the sampled oscillators.
    param_sets_csv : Path
        Path to CSV file containing the sampled parameter sets.
    n_procs : int, optional
        Number of processes to use for parallelization, by default 1.
    Q_thresh : float, optional
        Minimum value of Q_hat to consider a state an oscillator, by default 0.01.
    min_visits : int, optional
        Minimum number of visits to consider a state sufficiently sampled, by default 100.
    max_interactions : int, optional
        Maximum number of interactions to consider in the grammar, by default None.
    save : bool, optional
        Whether to save the distance matrix and metadata to an HDF5 file, by default False.
    save_dir : Path, optional
        Path to directory to save the HDF5 file, by default None.
    suffix : str, optional
        Suffix to append to the HDF5 file name, by default "".
    """

    # Suppress warnings from pandas
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    ttable_parquet = sorted(Path(data_dir).glob("*.parquet"))[-1]
    params_df = pd.read_csv(param_sets_csv)
    ttable_df = pd.read_parquet(ttable_parquet)

    if "visit" in ttable_df.columns:
        ttable_df = ttable_df.rename(columns={"visit": "param_index"})

    ttable_df = pd.merge(
        ttable_df,
        params_df.loc[:, ["param_index", "component_mutation"]],
        on="param_index",
    )
    ttable_df["oscillated"] = ttable_df["autocorr_min"] < -0.4
    ttable_df["has_mutation"] = ttable_df["component_mutation"] != "(none)"
    ttable_df["oscillated_with_mutation"] = (
        ttable_df["oscillated"] & ttable_df["has_mutation"]
    )

    # Get a summary of results for each state
    agg_df = (
        ttable_df.groupby("state")
        .agg(
            visits=pd.NamedAgg("state", "count"),
            n_osc=pd.NamedAgg("oscillated", "sum"),
            n_mut=pd.NamedAgg("has_mutation", "sum"),
            n_osc_with_mut=pd.NamedAgg("oscillated_with_mutation", "sum"),
        )
        .reset_index()
    )
    agg_df["Q_hat"] = agg_df["n_osc"] / agg_df["visits"]
    agg_df["Q_hat_std_err"] = np.sqrt(
        agg_df["Q_hat"] * (1 - agg_df["Q_hat"]) / agg_df["visits"]
    )

    # Isolate putative oscillators with at least a certain number of visits
    agg_df = agg_df.loc[
        (agg_df["Q_hat"] > Q_thresh) & (agg_df["visits"] > min_visits)
    ].reset_index(drop=True)
    agg_df["state"] = agg_df["state"].cat.remove_unused_categories()

    # Compute Q_hat with and without mutation, and std. error
    agg_df["Q_with_mutation"] = agg_df["n_osc_with_mut"] / agg_df["n_mut"]
    agg_df["Q_with_mut_std_err"] = np.sqrt(
        agg_df["Q_with_mutation"] * (1 - agg_df["Q_with_mutation"]) / agg_df["n_mut"]
    )
    agg_df["Q_wout_mutation"] = (agg_df["n_osc"] - agg_df["n_osc_with_mut"]) / (
        agg_df["visits"] - agg_df["n_mut"]
    )
    agg_df["Q_wout_mut_std_err"] = np.sqrt(
        agg_df["Q_wout_mutation"]
        * (1 - agg_df["Q_wout_mutation"])
        / (agg_df["visits"] - agg_df["n_mut"])
    )

    # Compute the ratio of Q_hat w/w/o mutation - this is the estimated fault tolerance
    agg_df["Q_mut_ratio"] = agg_df["Q_with_mutation"] / agg_df["Q_wout_mutation"]

    # Compute complexity of each state (number of interactions)
    agg_df["complexity"] = agg_df["state"].map(simplenetwork_n_interactions)

    print(f"Number of samples: {ttable_df.shape[0]}")
    print(f"Number of sampled states: {ttable_df['state'].nunique()}")
    print(
        f"Number of oscillators with at least {min_visits} samples: {agg_df.shape[0]}"
    )
    print(f"The most-sampled state has {agg_df['visits'].max()} samples")

    components, *_ = SimpleNetworkGrammar.parse_genotype(agg_df["state"].values[0])
    grammar = SimpleNetworkGrammar(
        components=list(components),
        interactions=["activates", "inhibits"],
        max_interactions=max_interactions,
        root=f"*{components}::",
    )

    n_components = len(grammar.components)
    n_interactions = n_components**2
    adj_matrices = []
    data = np.zeros((len(agg_df), n_interactions), dtype=int)
    for i, s in enumerate(agg_df["state"]):
        adj_mat = simplenetwork_adjacency_matrix(s, grammar, directed=True, signed=True)
        adj_matrices.append(adj_mat)
        data[i, :] = adj_mat.flatten()
    data = data.T

    dist_matrix = np.zeros((len(agg_df), len(agg_df)))
    pbar = tqdm(total=len(agg_df) * (len(agg_df) - 1) // 2)
    args = [
        (i, j, adj_matrices[i], adj_matrices[j])
        for i, j in zip(*np.tril_indices(len(agg_df), k=1))
    ]
    if n_procs == 1:
        for i, j, dist in map(compute_distance, args):
            dist_matrix[i, j] = dist
            pbar.update()
    else:
        with Pool(n_procs) as pool:
            for i, j, dist in pool.imap_unordered(
                compute_distance, args, chunksize=1000
            ):
                dist_matrix[i, j] = dist
                pbar.update()
    pbar.close()

    dist_matrix = dist_matrix + dist_matrix.T

    if save:
        today = datetime.today().strftime("%y%m%d")
        hdf_file = Path(save_dir).joinpath(f"{today}_pairwise_dist{suffix}.hdf5")
        with h5py.File(hdf_file, "w") as f:
            f.create_dataset("distance_matrix", data=dist_matrix)

        print(f"Saving distance matrix and metadata to {hdf_file}")
        agg_df.to_hdf(hdf_file, key="metadata", mode="a", format="table")


if __name__ == "__main__":
    data_dir = Path(
        "data/aws_exhaustion_exploration2.00/231118-02-23-28_5tf_exhaustion100"
        "_mutationrate0.5_batchsize1_max_interactions15_exploration2.000/backups"
    )
    param_sets_csv = Path(
        "data/oscillation/231104_param_sets_10000_5tf_pmutation0.5.csv"
    )
    min_visits = 100
    suffix = f"_5TF_FT_oscs_min_visits{min_visits}"

    save_dir = Path("data/oscillation/FT_oscillator_pairwise_dist")
    save_dir.mkdir(exist_ok=True)

    main(
        data_dir=data_dir,
        param_sets_csv=param_sets_csv,
        n_procs=13,
        min_visits=min_visits,
        save=True,
        save_dir=save_dir,
        suffix=suffix,
    )
